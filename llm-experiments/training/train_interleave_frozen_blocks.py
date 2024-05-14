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
    GPTNeoXForCausalLM,
)
from copy import deepcopy

logger = get_logger(__name__)

MODEL_CONFIG_CLASSES = list(MODEL_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)


def parse_args():

    parser = argparse.ArgumentParser(description="Finetune large language models on causal language modeling tasks")
    parser.add_argument("--dataset_name", type=str, default=None, required=True, help="The name of the dataset to use (via the datasets library).")
    parser.add_argument("--dataset_config_name", type=str, default=None, help="The configuration name of the dataset to use (via the datasets library).")
    parser.add_argument("--model_name_or_path", type=str, help="Path to pretrained model or model identifier from huggingface.co/models.", required=False)
    parser.add_argument("--revision", type=str, default='main', help="Model Branch")
    parser.add_argument("--config_name", type=str, default=None, help="Pretrained config name or path if not the same as model_name")
    parser.add_argument("--tokenizer_name", type=str, default=None, help="Pretrained tokenizer name or path if not the same as model_name")
    parser.add_argument("--use_slow_tokenizer", action="store_true", help="If passed, will use a slow tokenizer (not backed by the Tokenizers library).")
    parser.add_argument("--per_device_train_batch_size", type=int, default=8, help="Batch size (per device) for the training dataloader.")
    parser.add_argument("--learning_rate", type=float, default=0.0001, help="Initial learning rate (after the potential warmup period) to use.")
    parser.add_argument("--weight_decay", type=float, default=0.0, help="Weight decay to use.")
    parser.add_argument("--num_train_epochs", type=int, default=1, help="Total number of training epochs to perform.")
    parser.add_argument("--output_dir", type=str, default=None, help="Where to store the final model.")
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    parser.add_argument("--model_type", type=str, default=None, help="Model type to use if training from scratch.", choices=MODEL_TYPES)
    parser.add_argument("--block_size", type=int, default=None, help="The training dataset will be truncated to blocks of this size (after tokenization) for training.")
    parser.add_argument("--preprocessing_num_workers", type=int, default=None, help="The number of processes to use for the preprocessing.")
    parser.add_argument("--overwrite_cache", action="store_true", help="Overwrite the cached training and evaluation sets")
    parser.add_argument("--save_prefix", type=str, default='', help="Informative string prefix for saving purposes.")
    parser.add_argument("--use_pretrained_weights", action=argparse.BooleanOptionalAction, help="Whether to use pretrained weights.")
    parser.set_defaults(use_pretrained_weights=True)

    parser.add_argument("--eval_every_step", action=argparse.BooleanOptionalAction, help="Whether to eval every step.")
    parser.set_defaults(eval_every_step=True)

    parser.add_argument("--use_validation", action=argparse.BooleanOptionalAction, help="Whether to eval on validation set.")
    parser.set_defaults(use_validation=False)

    parser.add_argument("--per_device_eval_batch_size", type=int, default=8, help="Batch size (per device) for the evaluation dataloader.")
    parser.add_argument("--eval_freq", type=int, default=1, help="Number of epochs before every recall experiment.")
    parser.add_argument("--save_freq", type=int, default=10, help="Number of epochs before every moodel and optimizer save.")

    parser.add_argument("--num-grad-steps", type=int, default=1, help="Number of gradient updates for each data point.")
    parser.add_argument("--num-data-samples", type=int, default=1, help="Number of tasks to interleave.")
    parser.add_argument("--num-eval-data-samples", type=int, default=100, help="Number of tasks to interleave.")

    parser.add_argument("--num-frozen-blocks", type=int, default=0, help="Number of transformer blocks to freeze.")
    parser.add_argument("--num-frozen-blocks-rev", type=int, default=0, help="Number of transformer blocks to freeze (from the end)")

    args = parser.parse_args()

    return args


def main():
    args = parse_args()
    print(args)

    # eval_sample = [0]
    eval_sample = range(args.num_data_samples)

    accelerator = Accelerator()

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
    
    if 'test' in raw_datasets.keys():
        raw_datasets.pop('test')
    print("Length of Training Set", raw_datasets['train'])

    subset_task_index = random.sample(range(len(raw_datasets['train'])), args.num_data_samples)
    # subset_task_index = range(args.num_data_samples)

    raw_datasets['train'] = raw_datasets['train'].select(subset_task_index)
    if args.use_validation:
        subset_task_index_eval = random.sample(range(len(raw_datasets['validation'])), args.num_eval_data_samples)
        raw_datasets['validation'] = raw_datasets['validation'].select(subset_task_index_eval)
    elif 'validation' in raw_datasets.keys():
        raw_datasets.pop('validation')

    eval_datasets = deepcopy(raw_datasets)
    eval_datasets['train'] = eval_datasets['train'].select(eval_sample)

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
            model = GPTNeoXForCausalLM.from_pretrained(args.model_name_or_path, from_tf=bool(".ckpt" in args.model_name_or_path), config=config, revision=args.revision, cache_dir=f"./{model_name}/{args.revision}")
        else:
            model = AutoModelForCausalLM.from_predtrained(args.model_name_or_path, from_tf=bool(".ckpt" in args.model_name_or_path), config=config)
    else:
        logger.info("Training new model from scratch")
        if 'pythia' in args.model_name_or_path:
            model = GPTNeoXForCausalLM(config)
        else:
            model = AutoModelForCausalLM.from_config(config)

    model.resize_token_embeddings(len(tokenizer))

    # EXPERIMENTAL
    if not args.use_pretrained_weights:
        with torch.no_grad():
            for name, param in model.named_parameters():
                if 'norm' not in name and 'bias' not in name:
                    # print(f'Layer {name}, Weight Scale {torch.max(param).data}')
                    if 'query_key_value' in name or 'dense_h_to_4h' in name or 'embed_in' in name or 'embed_out' in name:
                        param *= math.sqrt(2 / (5 * 2048)) / 0.02
                    elif 'attention.dense' in name or 'dense_4h_to_h' in name:
                        param *= 2 / 16 / math.sqrt(2048) / 0.02

        for name, param in model.named_parameters():
            if 'norm' not in name and 'bias' not in name:
                print(f'Layer {name}, Weight Scale {torch.max(param).data}')

    column_names = raw_datasets["train"].column_names
    eval_column_names = eval_datasets["train"].column_names

    test_text_column_name = 'highlights'
    text_column_name = eval_text_column_name = "article"

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
    train_dataloader = DataLoader(train_dataset, shuffle=False, collate_fn=default_data_collator, batch_size=args.per_device_train_batch_size)

    eval_dataset = eval_lm_datasets["train"]
    eval_dataloader = DataLoader(eval_dataset, shuffle=False, collate_fn=default_data_collator, batch_size=args.per_device_eval_batch_size)

    test_dataset = test_lm_datasets['train']
    test_dataloader = DataLoader(test_dataset, shuffle=False, collate_fn=default_data_collator, batch_size=args.per_device_eval_batch_size)


    for index in random.sample(range(len(train_dataset)), 1):
        index = 0
        # logger.info(f"Sample {index} of the training set: {train_dataset[index]}.")
        logger.info(f"Sample {index} of the training set (decoded): {tokenizer.decode(train_dataset[index]['input_ids'], skip_special_tokens=True)}.")
    for index in random.sample(range(len(eval_dataset)), 1):
        # logger.info(f"Sample {index} of the validation set: {eval_dataset[index]}.")
        logger.info(f"Sample {index} of the validation set (decoded): {tokenizer.decode(eval_dataset[index]['input_ids'], skip_special_tokens=True)}.")
    
    no_decay = ["bias", "layer_norm.weight"]
    optimizer_grouped_parameters = [
        {"params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], "weight_decay": args.weight_decay},
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
    ]

    optimizer = torch.optim.SGD(optimizer_grouped_parameters, lr=args.learning_rate)

    # Prepare everything with our `accelerator`.
    model, optimizer, train_dataloader, eval_dataloader, test_dataloader = accelerator.prepare(model, optimizer, train_dataloader, eval_dataloader, test_dataloader)

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader)) * args.num_grad_steps
    args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch

    # Train!
    total_batch_size = args.per_device_train_batch_size * accelerator.num_processes

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.per_device_train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    
    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(args.max_train_steps), disable=not accelerator.is_local_main_process)
    completed_steps = 0
    starting_epoch = 0

    # update the progress_bar if load from checkpoint
    progress_bar.update(starting_epoch * num_update_steps_per_epoch)
    completed_steps = starting_epoch * num_update_steps_per_epoch

    total_num_task_steps = args.num_train_epochs * args.num_data_samples + 1

    train_losses = []
    eval_losses_all = torch.zeros(total_num_task_steps, len(eval_dataloader))
    test_losses_all = torch.zeros(total_num_task_steps, len(test_dataloader))

    # Initial Eval
    model.eval()
    with torch.no_grad():
        for eval_step, batch in enumerate(eval_dataloader):
            outputs = model(**batch)
            loss = outputs.loss
            eval_losses_all[0, eval_step] = loss.detach()
        logger.info(f"Mean eval loss: {torch.mean(eval_losses_all[0, :])}")

        for test_step, batch in enumerate(test_dataloader):
            outputs = model(**batch)
            loss = outputs.loss
            test_losses_all[0, test_step] = loss.detach()
        logger.info(f"Mean test loss: {torch.mean(test_losses_all[0, :])}")

    for epoch in range(starting_epoch, args.num_train_epochs):

        for step, batch in enumerate(train_dataloader):

            global_train_step = epoch * args.num_data_samples + step + 1

            model.train()

            if args.num_frozen_blocks > 0:
                for layer_idx, (name, param) in enumerate(model.named_parameters()):
                    if layer_idx <= 12 * args.num_frozen_blocks:
                        param.requires_grad = False

            if args.num_frozen_blocks_rev > 0:
                for layer_idx, (name, param) in enumerate(model.named_parameters()):
                    if layer_idx > 12 * (16 - args.num_frozen_blocks_rev) and layer_idx <= 12 * 16:
                        param.requires_grad = False
            
            
            for grad_step in range(args.num_grad_steps):

                assert model.training
                outputs = model(**batch)
                loss = outputs.loss
                # keep track of the loss at each epoch
                train_losses.append(loss.detach().unsqueeze(0))
                optimizer.zero_grad()
                accelerator.backward(loss)
                optimizer.step()

                # Checks if the accelerator has performed an optimization step behind the scenes
                if accelerator.sync_gradients:
                    progress_bar.update(1)
                    completed_steps += 1

            if args.eval_every_step:
                model.eval()
                with torch.no_grad():
                    for eval_step, eval_batch in enumerate(eval_dataloader):
                        eval_outputs = model(**eval_batch)
                        eval_loss = eval_outputs.loss
                        eval_losses_all[global_train_step, eval_step] = eval_loss.detach()

                    for test_step, test_batch in enumerate(test_dataloader):
                        test_outputs = model(**test_batch)
                        test_loss = test_outputs.loss
                        test_losses_all[global_train_step, test_step] = test_loss.detach()
                    
        # Logging
        output_dir = f"epoch_{epoch}"
        if args.output_dir is not None:
            output_dir = os.path.join(args.output_dir, output_dir)
            
        os.makedirs(output_dir, exist_ok=True)
        # if epoch == 0 or (epoch+1) % args.save_freq == 0:
        #     accelerator.save_state(output_dir)

        # save train_losses
        train_losses_ckpt = torch.cat(train_losses)
        train_losses_ckpt = train_losses_ckpt.cpu().numpy()
        logger.info(f"Mean train loss: {np.mean(train_losses_ckpt)}")

        save_path = os.path.join(output_dir, args.save_prefix + '_results.npz')
        np.savez(save_path, train_losses_ckpt=train_losses_ckpt, completed_steps=completed_steps)


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

        eval_losses_all_ckpt = eval_losses_all.cpu().numpy()
        test_losses_all_ckpt = test_losses_all.cpu().numpy()

        # save results
        save_path = os.path.join(output_dir, args.save_prefix + '_results.npz')
        np.savez(save_path, train_losses_ckpt=train_losses_ckpt, eval_losses_ckpt=eval_losses_all_ckpt, test_losses_ckpt=test_losses_all_ckpt, completed_steps=completed_steps)


if __name__ == "__main__":
    main()
