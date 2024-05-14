#  Anticipatory Recovery in Sequential Cyclic Fine-tuning

Code for the paper "Reawakening knowledge: Anticipatory recovery from catastrophic interference via structured training" (ArXiv 2024).

The `llm-experiments` folder includes the Language model ([Pythia](https://github.com/EleutherAI/pythia)) experiments; the `igpt-experiments` folder includes the [Image GPT](https://github.com/openai/image-gpt) experiments; the `imagenet-experiments` folder includes the image classification experiments.

## LLM Experiments

Example commands for cyclic fine-tuning experiments can be found in the `llm-experiments/scripts` folder.

Code for visualizing the pairwise recovery matrix (Figures 8b and 18), PCA in the last layer weights (Figure 9), and representations (Figure 8d) can be found in the `llm-experiments/visualization` folder.

## Image GPT Experiments

Example command for cyclic fine-tuning with Image GPT:
```
python train_interleave_igpt.py \
        --learning_rate 0.001 \
        --model_size medium \
        --output_dir ./medium-20steps \
        --save_prefix batch1_gpu1 \
        --num_train_epochs 5 \
        --num-grad-steps 20 \
        --num-data-samples 25
```

where ```num-grad-steps``` is the number of consecutive gradient update steps on each image, and ```num-data-samples``` is the number of images in the sequence.

## Acknowledgements

The code is adapted from [Huggingface Transformers](https://github.com/huggingface/transformers.git) and Emin Orhan's [LLM Memory Experiments](https://github.com/eminorhan/llm-memory.git).
