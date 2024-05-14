import argparse
import logging
import math
import os
import random
from itertools import chain

import torch
import numpy as np
from datasets import load_dataset
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from transformers import (
    CONFIG_MAPPING,
    MODEL_MAPPING,
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    default_data_collator,
)
from copy import deepcopy

logger = get_logger(__name__)

MODEL_CONFIG_CLASSES = list(MODEL_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)

class CustomSamplerWithoutReplacement(torch.utils.data.Sampler):
    def __init__(self, length, shuffle_end=1):
        assert length >= shuffle_end
        self.length = length
        self.shuffle_end = shuffle_end
        self.indices = list(range(1, shuffle_end))

    def __iter__(self):
        yield 0
        random.shuffle(self.indices)
        for idx in self.indices:
            yield idx
        for idx in range(self.shuffle_end, self.length):
            yield idx

    def __len__(self):
        return self.length


def parse_args():

    parser = argparse.ArgumentParser(description="Finetune large language models on causal language modeling tasks")
    parser.add_argument("--dataset_name", type=str, default=None, help="The name of the dataset to use (via the datasets library).")
    parser.add_argument("--dataset_config_name", type=str, default=None, help="The configuration name of the dataset to use (via the datasets library).")
    parser.add_argument("--train_file", type=str, default=None, help="A csv or a json file containing the training data.")
    parser.add_argument("--model_name_or_path", type=str, help="Path to pretrained model or model identifier from huggingface.co/models.", required=False)
    parser.add_argument("--revision", type=str, default='main', help="Model Branch")
    parser.add_argument("--config_name", type=str, default=None, help="Pretrained config name or path if not the same as model_name")
    parser.add_argument("--tokenizer_name", type=str, default=None, help="Pretrained tokenizer name or path if not the same as model_name")
    parser.add_argument("--use_slow_tokenizer", action="store_true", help="If passed, will use a slow tokenizer (not backed by the Tokenizers library).")
    parser.add_argument("--per_device_train_batch_size", type=int, default=8, help="Batch size (per device) for the training dataloader.")
    parser.add_argument("--learning_rate", type=float, default=0.0001, help="Initial learning rate (after the potential warmup period) to use.")
    parser.add_argument("--weight_decay", type=float, default=0.0, help="Weight decay to use.")
    parser.add_argument("--num_train_epochs", type=int, default=1, help="Total number of training epochs to perform.")
    parser.add_argument("--max_train_steps", type=int, default=None, help="Total number of training steps to perform. If provided, overrides num_train_epochs.")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--output_dir", type=str, default=None, help="Where to store the final model.")
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    parser.add_argument("--model_type", type=str, default=None, help="Model type to use if training from scratch.", choices=MODEL_TYPES)
    parser.add_argument("--block_size", type=int, default=None, help="The training dataset will be truncated to blocks of this size (after tokenization) for training.")
    parser.add_argument("--preprocessing_num_workers", type=int, default=None, help="The number of processes to use for the preprocessing.")
    parser.add_argument("--overwrite_cache", action="store_true", help="Overwrite the cached training and evaluation sets")
    parser.add_argument("--checkpointing_steps", type=str, default=None, help="Whether the various states should be saved at the end of every n steps, or 'epoch' for each epoch.")
    parser.add_argument("--resume_from_checkpoint", type=str, default=None, help="If the training should continue from a checkpoint folder.")
    parser.add_argument("--save_prefix", type=str, default='', help="Informative string prefix for saving purposes.")
    parser.add_argument("--use_pretrained_weights", action=argparse.BooleanOptionalAction, help="Whether to use pretrained weights.")
    parser.set_defaults(use_pretrained_weights=True)

    parser.add_argument("--eval_every_step", action=argparse.BooleanOptionalAction, help="Whether to eval every step.")
    parser.set_defaults(eval_every_step=True)

    parser.add_argument("--use_validation", action=argparse.BooleanOptionalAction, help="Whether to eval on validation set.")
    parser.set_defaults(use_validation=False)

    parser.add_argument("--seen_file", type=str, default=None, help="A csv or a json file containing the seen examples.")
    parser.add_argument("--per_device_eval_batch_size", type=int, default=8, help="Batch size (per device) for the evaluation dataloader.")
    parser.add_argument("--eval_freq", type=int, default=1, help="Number of epochs before every recall experiment.")
    parser.add_argument("--save_freq", type=int, default=10, help="Number of epochs before every moodel and optimizer save.")

    parser.add_argument("--num-grad-steps", type=int, default=1, help="Number of gradient updates for each data point.")
    parser.add_argument("--num-data-samples", type=int, default=1, help="Number of tasks to interleave.")
    parser.add_argument("--num-eval-data-samples", type=int, default=100, help="Number of tasks to interleave.")
    parser.add_argument("--start-shuffle-data-samples", type=int, default=-1, help="Number of tasks to interleave.")

    args = parser.parse_args()

    if args.start_shuffle_data_samples == -1:
        args.start_shuffle_data_samples = args.num_data_samples - 1

    # Sanity checks
    if args.dataset_name is None and args.train_file is None:
        raise ValueError("Need either a dataset name or a training file.")
    else:
        if args.train_file is not None:
            extension = args.train_file.split(".")[-1]
            assert extension in ["csv", "json", "txt"], "`train_file` should be a csv, json or txt file."

    return args


def main():
    args = parse_args()
    print(args)

    eval_sample = 0

    accelerator = Accelerator(gradient_accumulation_steps=args.gradient_accumulation_steps)
    
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()

    if args.seed is not None:
        set_seed(args.seed)

    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)
    accelerator.wait_for_everyone()

    raw_datasets = load_dataset(args.dataset_name, args.dataset_config_name)
    
    raw_datasets.pop('test')
    subset_task_index = random.sample(range(len(raw_datasets['train'])), args.num_data_samples)
    # subset_task_index = range(args.num_data_samples)
    raw_datasets['train'] = raw_datasets['train'].select(subset_task_index)
    if args.use_validation:
        subset_task_index_eval = random.sample(range(len(raw_datasets['validation'])), args.num_eval_data_samples)
        raw_datasets['validation'] = raw_datasets['validation'].select(subset_task_index_eval)
    else:
        raw_datasets.pop('validation')

    eval_datasets = deepcopy(raw_datasets)
    eval_datasets['train'] = eval_datasets['train'].select([eval_sample])

    # Load pretrained model and tokenizer
    if args.config_name:
        config = AutoConfig.from_pretrained(args.config_name)
    elif args.model_name_or_path:
        config = AutoConfig.from_pretrained(args.model_name_or_path)
    else:
        config = CONFIG_MAPPING[args.model_type]()
        logger.warning("You are instantiating a new config instance from scratch.")

    if args.tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name, use_fast=not args.use_slow_tokenizer, model_max_length=2048)
    elif args.model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=not args.use_slow_tokenizer, model_max_length=2048)
        if args.model_name_or_path.startswith("gpt2") or args.model_name_or_path.startswith("EleutherAI"):
            tokenizer.pad_token = tokenizer.eos_token
    else:
        raise ValueError()

    if args.model_name_or_path and args.use_pretrained_weights:
        if 'pythia' in args.model_name_or_path:
            model_author, model_name = args.model_name_or_path.split('/')
            model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path, from_tf=bool(".ckpt" in args.model_name_or_path), config=config, revision=args.revision, cache_dir=f"./{model_name}/{args.revision}")
        else:
            model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path, from_tf=bool(".ckpt" in args.model_name_or_path), config=config)
    else:
        logger.info("Training new model from scratch")
        model = AutoModelForCausalLM.from_config(config)

    model.resize_token_embeddings(len(tokenizer))

    column_names = raw_datasets["train"].column_names
    text_column_name = eval_text_column_name = "article"

    eval_column_names = eval_datasets["train"].column_names

    test_text_column_name = 'highlights'

    if args.block_size is None:
        block_size = tokenizer.model_max_length
    else:
        block_size = args.block_size
        if args.block_size > tokenizer.model_max_length:
            logger.warning(
                f"The block_size passed ({args.block_size}) is larger than the maximum length for the model"
                f"({tokenizer.model_max_length}). Using block_size={tokenizer.model_max_length}."
            )
            block_size = tokenizer.model_max_length
    print('Block size:', block_size)

    def tokenize_function(examples):
        return tokenizer(examples[text_column_name], padding='max_length', truncation=True, max_length=block_size)

    def tokenize_function_eval(examples):
        return tokenizer(examples[eval_text_column_name], truncation=True, max_length=block_size)

    def tokenize_function_test(examples):
        return tokenizer(examples[test_text_column_name], truncation=True, max_length=block_size)

    with accelerator.main_process_first():
        tokenized_datasets = raw_datasets.map(
            tokenize_function,
            batched=True,
            num_proc=args.preprocessing_num_workers,
            remove_columns=column_names,
            load_from_cache_file=not args.overwrite_cache,
            desc="Running tokenizer on dataset",
        )

    with accelerator.main_process_first():
        eval_tokenized_datasets = eval_datasets.map(
            tokenize_function_eval,
            batched=True,
            num_proc=args.preprocessing_num_workers,
            remove_columns=eval_column_names,
            load_from_cache_file=not args.overwrite_cache,
            desc="Running tokenizer on dataset",
        )

    with accelerator.main_process_first():
        test_tokenized_datasets = eval_datasets.map(
            tokenize_function_test,
            batched=True,
            num_proc=args.preprocessing_num_workers,
            remove_columns=eval_column_names,
            load_from_cache_file=not args.overwrite_cache,
            desc="Running tokenizer on dataset",
        )

    def preprocess_function(examples):
        examples["labels"] = examples["input_ids"].copy()
        examples["labels"] = [[(l if l != tokenizer.pad_token_id else -100) for l in label] for label in examples["labels"]]
        return examples
    
    with accelerator.main_process_first():
        lm_datasets = tokenized_datasets.map(
            preprocess_function,
            batched=True,
            num_proc=args.preprocessing_num_workers,
            load_from_cache_file=not args.overwrite_cache,
            desc=f"Not grouping text.",
        )

    with accelerator.main_process_first():
        eval_lm_datasets = eval_tokenized_datasets.map(
            preprocess_function,
            batched=True,
            num_proc=args.preprocessing_num_workers,
            load_from_cache_file=not args.overwrite_cache,
            desc=f"Not grouping text.",
        )

    with accelerator.main_process_first():
        test_lm_datasets = test_tokenized_datasets.map(
            preprocess_function,
            batched=True,
            num_proc=args.preprocessing_num_workers,
            load_from_cache_file=not args.overwrite_cache,
            desc=f"Not grouping text.",
        )

    train_dataset = lm_datasets["train"]
    # train_dataloader = DataLoader(train_dataset, shuffle=False, collate_fn=default_data_collator, batch_size=args.per_device_train_batch_size)
    train_sampler = CustomSamplerWithoutReplacement(len(train_dataset), shuffle_end=args.start_shuffle_data_samples)
    train_dataloader = DataLoader(train_dataset, collate_fn=default_data_collator, batch_size=args.per_device_train_batch_size, sampler=train_sampler)

    eval_dataset = eval_lm_datasets["train"]
    eval_dataloader = DataLoader(eval_dataset, shuffle=False, collate_fn=default_data_collator, batch_size=args.per_device_eval_batch_size)

    test_dataset = test_lm_datasets['train']
    test_dataloader = DataLoader(test_dataset, shuffle=False, collate_fn=default_data_collator, batch_size=args.per_device_eval_batch_size)


    for index in random.sample(range(len(train_dataset)), 1):
        logger.info(f"Sample {index} of the training set (decoded): {tokenizer.decode(train_dataset[index]['input_ids'], skip_special_tokens=True)}.")
    for index in random.sample(range(len(eval_dataset)), 1):
        logger.info(f"Sample {index} of the training set (decoded): {tokenizer.decode(eval_dataset[index]['input_ids'], skip_special_tokens=True)}.")
    
    no_decay = ["bias", "layer_norm.weight"]
    optimizer_grouped_parameters = [
        {"params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], "weight_decay": args.weight_decay},
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
    ]

    optimizer = torch.optim.SGD(optimizer_grouped_parameters, lr=args.learning_rate)

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps) * args.num_grad_steps
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    # Prepare everything with our `accelerator`.
    model, optimizer, train_dataloader, eval_dataloader, test_dataloader = accelerator.prepare(model, optimizer, train_dataloader, eval_dataloader, test_dataloader)

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps) * args.num_grad_steps
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch

    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # Figure out how many steps we should save the Accelerator states
    checkpointing_steps = args.checkpointing_steps
    if checkpointing_steps is not None and checkpointing_steps.isdigit():
        checkpointing_steps = int(checkpointing_steps)

    # Train!
    total_batch_size = args.per_device_train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.per_device_train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    
    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(args.max_train_steps), disable=not accelerator.is_local_main_process)
    completed_steps = 0
    starting_epoch = 0

    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint is not None or args.resume_from_checkpoint != "":
            accelerator.print(f"Resumed from checkpoint: {args.resume_from_checkpoint}")
            accelerator.load_state(args.resume_from_checkpoint)
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            dirs = [f.name for f in os.scandir(os.getcwd()) if f.is_dir()]
            dirs.sort(key=os.path.getctime)
            path = dirs[-1]  # Sorts folders by date modified, most recent checkpoint is the last
        # Extract `epoch_{i}` or `step_{i}`
        training_difference = os.path.splitext(path)[0]

        if "epoch" in training_difference:
            starting_epoch = int(training_difference.replace("epoch_", "")) + 1
            resume_step = None
        else:
            # need to multiply `gradient_accumulation_steps` to reflect real steps
            resume_step = int(training_difference.replace("step_", "")) * args.gradient_accumulation_steps
            starting_epoch = resume_step // len(train_dataloader)
            resume_step -= starting_epoch * len(train_dataloader)

    # update the progress_bar if load from checkpoint
    progress_bar.update(starting_epoch * num_update_steps_per_epoch)
    completed_steps = starting_epoch * num_update_steps_per_epoch

    train_losses = []
    train_losses_all = []

    eval_losses = []
    eval_losses_all = []

    test_losses = []
    test_losses_all = []

    # Initial Eval
    model.eval()
    with torch.no_grad():
        for _, batch in enumerate(eval_dataloader):
            outputs = model(**batch)
            loss = outputs.loss
            eval_losses.append(loss.detach().unsqueeze(0))
            eval_losses_all.append(loss.detach().unsqueeze(0))
        eval_losses_ckpt = torch.cat(eval_losses)
        eval_losses_ckpt = eval_losses_ckpt.cpu().numpy()
        logger.info(f"Mean eval loss: {np.mean(eval_losses_ckpt)}")

        for _, batch in enumerate(test_dataloader):
            outputs = model(**batch)
            loss = outputs.loss
            test_losses.append(loss.detach().unsqueeze(0))
            test_losses_all.append(loss.detach().unsqueeze(0))
        test_losses_ckpt = torch.cat(test_losses)
        test_losses_ckpt = test_losses_ckpt.cpu().numpy()
        logger.info(f"Mean TEST loss: {np.mean(test_losses_ckpt)}")

    for epoch in range(starting_epoch, args.num_train_epochs):
        for step, batch in enumerate(train_dataloader):
            
            # if step >= max(epoch, 2) * 10:
            #     continue

            if args.resume_from_checkpoint and epoch == starting_epoch:
                if resume_step is not None and step < resume_step:
                    if step % args.gradient_accumulation_steps == 0:
                        progress_bar.update(1)
                        completed_steps += 1
                    continue
            
            model.train()
            # We need to skip steps until we reach the resumed step
            for _ in range(args.num_grad_steps):

                with accelerator.accumulate(model):
                    assert model.training
                    outputs = model(**batch)
                    loss = outputs.loss
                    # keep track of the loss at each epoch
                    train_losses.append(loss.detach().unsqueeze(0))
                    train_losses_all.append(loss.detach().unsqueeze(0))
                    optimizer.zero_grad()
                    accelerator.backward(loss)
                    optimizer.step()
                    # optimizer.zero_grad()

                # Checks if the accelerator has performed an optimization step behind the scenes
                if accelerator.sync_gradients:
                    progress_bar.update(1)
                    completed_steps += 1

            if args.eval_every_step:
                model.eval()
                with torch.no_grad():
                    eval_losses = []
                    for eval_step, eval_batch in enumerate(eval_dataloader):
                        eval_outputs = model(**eval_batch)
                        eval_loss = eval_outputs.loss
                        eval_losses.append(eval_loss.detach().unsqueeze(0))
                        eval_losses_all.append(eval_loss.detach().unsqueeze(0))

                    test_losses = []
                    for test_step, test_batch in enumerate(test_dataloader):
                        test_outputs = model(**test_batch)
                        test_loss = test_outputs.loss
                        test_losses.append(test_loss.detach().unsqueeze(0))
                        test_losses_all.append(test_loss.detach().unsqueeze(0))
                    
                    logger.info(f"Step {step}, Eval Loss {eval_losses[0]}")

            if isinstance(checkpointing_steps, int):
                if completed_steps % checkpointing_steps == 0:
                    output_dir = f"step_{completed_steps}"
                    if args.output_dir is not None:
                        output_dir = os.path.join(args.output_dir, output_dir)

                    # save model and tokenizer
                    accelerator.wait_for_everyone()
                    unwrapped_model = accelerator.unwrap_model(model)
                    unwrapped_model.save_pretrained(output_dir, is_main_process=accelerator.is_main_process, save_function=accelerator.save)
                    if accelerator.is_main_process:
                        tokenizer.save_pretrained(output_dir)

                    # save train_losses
                    train_losses_ckpt = torch.cat(train_losses)
                    train_losses_ckpt = train_losses_ckpt.cpu().numpy()

                    train_losses_all_ckpt = torch.cat(train_losses_all)
                    train_losses_all_ckpt = train_losses_all_ckpt.cpu().numpy()

                    logger.info(f"Mean train loss: {np.mean(train_losses_ckpt)}")

                    save_path = os.path.join(output_dir, 'train_losses.npz')
                    np.savez(save_path, train_losses=train_losses_all_ckpt, completed_steps=completed_steps)

                    # re-initialize losses
                    train_losses = []

            if completed_steps >= args.max_train_steps:
                break

        if args.checkpointing_steps == "epoch":
            output_dir = f"epoch_{epoch}"
            if args.output_dir is not None:
                output_dir = os.path.join(args.output_dir, output_dir)
                
            os.makedirs(output_dir, exist_ok=True)
            if epoch == 0 or (epoch+1) % args.save_freq == 0:
                accelerator.save_state(output_dir)

            # save train_losses
            train_losses_ckpt = torch.cat(train_losses)
            train_losses_ckpt = train_losses_ckpt.cpu().numpy()
            logger.info(f"Mean train loss: {np.mean(train_losses_ckpt)}")

            save_path = os.path.join(output_dir, args.save_prefix + '_results.npz')
            np.savez(save_path, train_losses_ckpt=train_losses_ckpt, completed_steps=completed_steps)

        # if epoch == 0 or (epoch+1) % args.eval_freq == 0:
        #     for step, batch in enumerate(test_dataloader):
        #         outputs = model(**batch)
        #         loss = outputs.loss
        #         test_losses.append(loss.detach().unsqueeze(0))
        #     test_losses_ckpt = torch.cat(test_losses)
        #     test_losses_ckpt = test_losses_ckpt.cpu().numpy()
        #     logger.info(f"Mean test loss: {np.mean(test_losses_ckpt)}")

    if args.output_dir is not None:
        output_dir = os.path.join(args.output_dir, f'final')
        # save model and tokenizer
        accelerator.wait_for_everyone()
        unwrapped_model = accelerator.unwrap_model(model)
        unwrapped_model.save_pretrained(output_dir, is_main_process=accelerator.is_main_process, save_function=accelerator.save)
        if accelerator.is_main_process:
            tokenizer.save_pretrained(output_dir)

        # save train_losses
        train_losses_ckpt = torch.cat(train_losses)
        train_losses_ckpt = train_losses_ckpt.cpu().numpy()
        logger.info(f"Final mean train loss: {np.mean(train_losses_ckpt)}")

        eval_losses_all_ckpt = torch.cat(eval_losses_all)
        eval_losses_all_ckpt = eval_losses_all_ckpt.cpu().numpy()

        test_losses_all_ckpt = torch.cat(test_losses_all)
        test_losses_all_ckpt = test_losses_all_ckpt.cpu().numpy()

        # save results
        save_path = os.path.join(output_dir, args.save_prefix + '_results.npz')
        np.savez(save_path, train_losses_ckpt=train_losses_ckpt, eval_losses_ckpt=eval_losses_all_ckpt, test_losses_ckpt=test_losses_all_ckpt, completed_steps=completed_steps)


if __name__ == "__main__":
    main()
