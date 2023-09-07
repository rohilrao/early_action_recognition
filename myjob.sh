#!/bin/bash
#SBATCH --job-name=video_prediction
#SBATCH --partition=A40devel  
#SBATCH --time=01:00:00              # Set the maximum runtime for your job (e.g., 2 hours)
#SBATCH --mem=16G                    # Amount of memory per node
#SBATCH --gpus=1                    # Number of GPUs required
#SBATCH --output=output.log         # Specify the output log file

# Run your video prediction script
python train.py --exp_name=randomtest --videos_dir=/home/s6roraoo/smth-smth/videos --batch_size=16 --noclip_lr=3e-5 --transformer_dropout=0.3 --huggingface --dataset_name=SMTH --num_workers=1
