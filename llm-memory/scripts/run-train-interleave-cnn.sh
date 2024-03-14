python training_scripts/train_interleave.py --model_name_or_path "EleutherAI/pythia-1b" --revision main --dataset_name "cnn_dailymail" --dataset_config_name 3.0.0 --per_device_train_batch_size 1 --per_device_eval_batch_size 1 --learning_rate 0.001 --output_dir checkpoints/0201-1b-5grad-512-exp$SLURM_ARRAY_TASK_ID --save_prefix batch1_gpu1 --block_size 512 --num_train_epochs 5 --overwrite_cache --save_freq 1 --num-grad-steps 5 --num-data-samples 25"

