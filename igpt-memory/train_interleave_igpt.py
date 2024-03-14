import argparse
import logging
import math
import os
import random

import torch
import numpy as np
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import transformers
from logging import getLogger
from transformers import (
    AutoConfig,
    default_data_collator,
    GPTNeoXForCausalLM,
)
from copy import deepcopy
from transformers import AutoImageProcessor, ImageGPTImageProcessor, ImageGPTForCausalImageModeling
import torchvision
from PIL import Image
import time

logger = getLogger(__name__)

class CustomSamplerWithoutReplacement(torch.utils.data.Sampler):
    def __init__(self, length, shuffle_start=1):
        assert length >= shuffle_start
        self.length = length
        self.shuffle_start = shuffle_start
        self.indices = list(range(shuffle_start, length))

    def __iter__(self):
        for idx in range(self.shuffle_start):
            yield idx
        random.shuffle(self.indices)
        for idx in self.indices:
            yield idx

    def __len__(self):
        return self.length

class CIFAR100IGPT(torchvision.datasets.CIFAR100):
    def __init__(self, root, train, download, image_processor):
        super().__init__(root=root, train=train, download=download)
        self.image_processor = image_processor

    def __getitem__(self, index):
        img = self.data[index]
        img = Image.fromarray(img)
        batch = self.image_processor(img, return_tensors='pt')
        batch["labels"] = batch["input_ids"].detach().clone()
        return batch


def parse_args():

    parser = argparse.ArgumentParser(description="Finetune large language models on causal language modeling tasks")
    parser.add_argument("--dataset_name", type=str, default='CIFAR100', help="The name of the dataset to use (via the datasets library).")
    parser.add_argument("--model_size", type=str, default='small', choices=['small', 'medium', 'large'], help="pretrained image GPT model size.")
    parser.add_argument("--learning_rate", type=float, default=0.001, help="Initial learning rate (after the potential warmup period) to use.")
    parser.add_argument("--num_train_epochs", type=int, default=1, help="Total number of training epochs to perform.")
    parser.add_argument("--output_dir", type=str, default=None, help="Where to store the final model.")
    parser.add_argument("--resume_from_checkpoint", type=str, default=None, help="If the training should continue from a checkpoint folder.")
    parser.add_argument("--save_prefix", type=str, default='', help="Informative string prefix for saving purposes.")
    parser.add_argument("--use_pretrained_weights", action=argparse.BooleanOptionalAction, help="Whether to use pretrained weights.")
    parser.set_defaults(use_pretrained_weights=True)
    parser.add_argument("--eval_every_step", action=argparse.BooleanOptionalAction, help="Whether to eval every step.")
    parser.set_defaults(eval_every_step=True)
    parser.add_argument("--use_validation", action=argparse.BooleanOptionalAction, help="Whether to eval on validation set.")
    parser.set_defaults(use_validation=False)
    parser.add_argument("--store_state", action=argparse.BooleanOptionalAction, help="Whether to store model weights.")
    parser.set_defaults(use_validation=False)
    parser.add_argument("--num-grad-steps", type=int, default=1, help="Number of gradient updates for each data point.")
    parser.add_argument("--batch_size", type=int, default=1, help="Number of images in each task (batch).")
    parser.add_argument("--num-data-samples", type=int, default=50, help="Number of tasks to interleave.")

    args = parser.parse_args()

    return args


def main():
    device = 'cuda'
    args = parse_args()
    print(args)

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )

    transformers.utils.logging.set_verbosity_error()

    if args.output_dir is not None:
        os.makedirs(args.output_dir, exist_ok=True)

    # Load pretrained model and Image Processor
    image_processor = AutoImageProcessor.from_pretrained(f"openai/imagegpt-{args.model_size}")
    model = ImageGPTForCausalImageModeling.from_pretrained(f"openai/imagegpt-{args.model_size}").to(device)

    # Load Datasets
    data_transform = torchvision.transforms.ToTensor()

    if args.dataset_name == 'CIFAR100':
        dataset = CIFAR100IGPT(root=str(os.environ.get('DATA')), train=True, download=True, image_processor=image_processor)
    else:
        raise NotImplementedError

    subset_task_index = random.sample(range(len(dataset)), args.num_data_samples * args.batch_size)
    train_dataset = torch.utils.data.Subset(dataset, subset_task_index)
    eval_dataset = torch.utils.data.Subset(dataset, subset_task_index)

    # model.resize_token_embeddings(len(tokenizer))

    train_dataloader = DataLoader(train_dataset, shuffle=False, collate_fn=default_data_collator, batch_size=args.batch_size)
    eval_dataloader = DataLoader(eval_dataset, shuffle=False, collate_fn=default_data_collator, batch_size=args.batch_size)

    # optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    # Scheduler and math around the number of training steps.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader)) * args.num_grad_steps
    args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    
    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(args.max_train_steps))
    completed_steps = 0
    starting_epoch = 0

    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        raise NotImplementedError()

    # update the progress_bar if load from checkpoint
    progress_bar.update(starting_epoch * num_update_steps_per_epoch)
    completed_steps = starting_epoch * num_update_steps_per_epoch

    total_num_task_steps = args.num_train_epochs * args.num_data_samples + 1

    train_losses = []
    eval_losses_all = torch.zeros(total_num_task_steps, len(eval_dataloader))

    # Initial Eval
    model.eval()
    with torch.no_grad():
        for eval_step, (batch) in enumerate(eval_dataloader):
            batch['input_ids'] = batch['input_ids'].to(device)
            batch['labels'] = batch['labels'].to(device)
            outputs = model(**batch)
            loss = outputs.loss
            eval_losses_all[0, eval_step] = loss.detach()
        logger.info(f"Mean eval loss: {torch.mean(eval_losses_all[0, :])}")

    for epoch in range(starting_epoch, args.num_train_epochs):

        if args.resume_from_checkpoint and epoch == starting_epoch:
            if resume_step is not None and step < resume_step:
                progress_bar.update(1)
                completed_steps += 1
                continue

        for step, (batch) in enumerate(train_dataloader):
            global_train_step = epoch * args.num_data_samples + step + 1

            batch['input_ids'] = batch['input_ids'].to(device)
            batch['labels'] = batch['labels'].to(device)
            for _ in range(args.num_grad_steps):
                outputs = model(**batch)
                loss = outputs.loss
                train_losses.append(loss.detach().unsqueeze(0))
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                progress_bar.update(1)
                completed_steps += 1

            if args.eval_every_step:
                with torch.no_grad():
                    for eval_step, (eval_batch) in enumerate(eval_dataloader):
                        eval_batch['input_ids'] = eval_batch['input_ids'].to(device)
                        eval_batch['labels'] = eval_batch['labels'].to(device)
                        eval_outputs = model(**eval_batch)
                        eval_loss = eval_outputs.loss
                        eval_losses_all[global_train_step, eval_step] = eval_loss.detach()

                if args.store_state:
                    save_dir = f"task_{epoch * args.num_data_samples + step}.pth"
                    save_dir = os.path.join(args.output_dir, save_dir)
                    torch.save(model.state_dict(), save_dir)

        output_dir = f"epoch_{epoch}"
        if args.output_dir is not None:
            output_dir = os.path.join(args.output_dir, output_dir)
        os.makedirs(output_dir, exist_ok=True)

        # save train_losses
        train_losses_ckpt = torch.cat(train_losses)
        train_losses_ckpt = train_losses_ckpt.cpu().numpy()
        logger.info(f"Mean train loss: {np.mean(train_losses_ckpt)}")

        save_path = os.path.join(output_dir, args.save_prefix + '_results.npz')
        np.savez(save_path, train_losses_ckpt=train_losses_ckpt, completed_steps=completed_steps)

    if args.output_dir is not None:
        output_dir = os.path.join(args.output_dir, f'final')
        os.makedirs(output_dir, exist_ok=True)

        # save train_losses
        train_losses_ckpt = torch.cat(train_losses)
        train_losses_ckpt = train_losses_ckpt.cpu().numpy()
        logger.info(f"Final mean train loss: {np.mean(train_losses_ckpt)}")

        eval_losses_all_ckpt = eval_losses_all.cpu().numpy()

        # save results
        save_path = os.path.join(output_dir, args.save_prefix + '_results.npz')
        np.savez(save_path, train_losses_ckpt=train_losses_ckpt, eval_losses_ckpt=eval_losses_all_ckpt, completed_steps=completed_steps)


if __name__ == "__main__":
    main()
