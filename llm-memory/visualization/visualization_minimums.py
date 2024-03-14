import numpy as np
import matplotlib.pyplot as plt
import os
import torch
import numpy as np
import sklearn
import sklearn.decomposition
from tqdm import tqdm

load_dir = '.' # Change to the experiment saving directory

num_tasks = 25
num_epochs = 3
num_total_tasks = num_tasks * num_epochs
last_layer_weights = []
all_tasks = range(num_total_tasks)
# all_tasks = range(10)
for task_num in tqdm(all_tasks):
    log_dir = f'{load_dir}/task_{task_num}'
    model_file = os.path.join(log_dir, 'pytorch_model.bin')
    model_weights = torch.load(model_file)
    last_layer_weight = model_weights['embed_out.weight'].cpu().numpy().flatten()
    last_layer_weights.append(last_layer_weight)

last_layer_weights = np.stack(last_layer_weights)
print(last_layer_weights.shape)

pca = sklearn.decomposition.PCA(n_components=3)
X = pca.fit_transform(last_layer_weights)
