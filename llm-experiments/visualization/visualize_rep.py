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
    parser.add_argument("--load_dir", type=str, default=None, help="Directory to experiment for loading.")
    parser.add_argument("--max_train_steps", type=int, default=None, help="Total number of training steps to perform. If provided, overrides num_train_epochs.")
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
    parser.add_argument("--per_device_eval_batch_size", type=int, default=8, help="Batch size (per device) for the evaluation dataloader.")
    parser.add_argument("--eval_freq", type=int, default=1, help="Number of epochs before every recall experiment.")
    parser.add_argument("--save_freq", type=int, default=10, help="Number of epochs before every moodel and optimizer save.")
    parser.add_argument("--num-data-samples", type=int, default=1, help="Number of tasks to interleave.")
    parser.add_argument("--num-eval-data-samples", type=int, default=100, help="Number of tasks to interleave.")

    args = parser.parse_args()

    return args


def main():
    args = parse_args()
    print(args)

    eval_sample = [12]
    # eval_sample = range(args.num_data_samples)

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

    raw_datasets = load_dataset(args.dataset_name, args.dataset_config_name)
    
    if 'test' in raw_datasets.keys():
        raw_datasets.pop('test')
    print("Length of Training Set", raw_datasets['train'])
    subset_task_index = range(args.num_data_samples)
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

    def tokenize_function_eval(examples):
        return tokenizer(examples[eval_text_column_name], truncation=True, max_length=block_size)

    with accelerator.main_process_first():
        eval_tokenized_datasets = eval_datasets.map(
            tokenize_function_eval,
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
        eval_lm_datasets = eval_tokenized_datasets.map(
            preprocess_function,
            batched=True,
            num_proc=args.preprocessing_num_workers,
            load_from_cache_file=not args.overwrite_cache,
            desc=f"Not grouping text.",
        )

    eval_dataset = eval_lm_datasets["train"]
    eval_dataloader = DataLoader(eval_dataset, shuffle=False, collate_fn=default_data_collator, batch_size=args.per_device_eval_batch_size)

    for index in random.sample(range(len(eval_dataset)), 1):
        # logger.info(f"Sample {index} of the validation set: {eval_dataset[index]}.")
        logger.info(f"Sample {index} of the validation set (decoded): {tokenizer.decode(eval_dataset[index]['input_ids'], skip_special_tokens=True)}")
    
    # Prepare everything with our `accelerator`.
    model, eval_dataloader = accelerator.prepare(model, eval_dataloader)
    
    num_steps = 125
    reps_all = np.zeros((num_steps, 262144))

    # Initial Eval
    for task_idx in tqdm(range(0, num_steps)):
        model_weights = torch.load(f'{args.load_dir}/task_{task_idx}/pytorch_model.bin')
        model.load_state_dict(model_weights)
        model.eval()

        with torch.no_grad():
            for step, batch in enumerate(eval_dataloader):
                rep = model.gpt_neox(batch['input_ids'])['last_hidden_state'].flatten()
                reps_all[task_idx] = rep.detach().cpu().numpy()
        
        del model_weights

    all_norms = np.zeros((num_steps, 1))
    for i in range(num_steps):
        all_norms[i] = np.linalg.norm(reps_all[i])
    norm_outer = all_norms @ all_norms.T
    all_corr = reps_all @ reps_all.T
    print(all_corr.shape)
    print(norm_outer.shape)
    all_corr = all_corr / norm_outer

if __name__ == "__main__":
    main()
