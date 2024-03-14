import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchvision.models
from torchvision import transforms
from torchvision.datasets import ImageFolder
import os
from torch.utils.data import DataLoader, Subset
from tqdm.auto import tqdm
import random

device = 'cuda'

### CONFIGS ###
num_grad_steps = 10
num_epochs = 5
lr = 0.0001
batch_size = 32
num_tasks = 25

# model = torchvision.models.vit_b_32(weights=torchvision.models.ViT_B_32_Weights.DEFAULT)
# model = torchvision.models.vit_b_16()
model = torchvision.models.vgg19(weights=torchvision.models.VGG19_Weights.DEFAULT)
transform = torchvision.models.VGG19_Weights.DEFAULT.transforms() # Correspond to the same model above

print("Model Parameter Count", sum([np.prod(p.size()) for p in model.parameters()]))
model = model.to(device)


class_names = os.listdir('/imagenet/train')
dataset = ImageFolder(root='/imagenet/train', transform=transform)
random_idx = random.sample(range(len(dataset)), batch_size * num_tasks)
small_dataset = Subset(dataset, random_idx)
eval_dataset = Subset(dataset, random_idx)

dataloader = DataLoader(small_dataset, shuffle=False, batch_size=batch_size, num_workers=2)
eval_dataloader = DataLoader(eval_dataset, shuffle=False, batch_size=batch_size, num_workers=2)

loss_fn = nn.CrossEntropyLoss()
total_num_task_steps = num_epochs * num_tasks + 1
train_losses = []
eval_losses = torch.zeros(total_num_task_steps, num_tasks)

# Initial Eval
model.eval()
with torch.no_grad():
    for eval_step, (eval_samples, eval_targets) in enumerate(eval_dataloader):
        eval_samples = eval_samples.to(device)
        eval_targets = eval_targets.to(device)
        eval_preds = model(eval_samples)
        eval_loss = loss_fn(eval_preds, eval_targets)
        eval_losses[0, eval_step] = eval_loss.detach()

progress_bar = tqdm(range(num_epochs * num_tasks))

for epoch in range(num_epochs):
    for step, (samples, targets) in enumerate(dataloader):
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        global_train_step = epoch * num_tasks + step + 1

        model.train()
        samples = samples.to(device)
        targets = targets.to(device)
        for _ in range(num_grad_steps):
            preds = model(samples)
            loss = loss_fn(preds, targets)
            train_losses.append(loss.detach().unsqueeze(0))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            for eval_step, (eval_samples, eval_targets) in enumerate(eval_dataloader):
                eval_samples = eval_samples.to(device)
                eval_targets = eval_targets.to(device)
                eval_preds = model(eval_samples)
                eval_loss = loss_fn(eval_preds, eval_targets)
                eval_losses[global_train_step, eval_step] = eval_loss.detach()
    
        progress_bar.update(1)

eval_losses = eval_losses.cpu().numpy()
