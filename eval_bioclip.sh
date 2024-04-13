#!/bin/bash
#SBATCH --job-name=test
#SBATCH --account=project_2002784
#SBATCH --partition=gpu
#SBATCH --time=02:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=3G
#SBATCH --gres=gpu:v100:1

module load pytorch/2.2
# pip install open_clip_torch
srun python3 CLIP_eval_model.py --dataset PLANKTON --start_trial 0 --num_trials 5 --clip bioclip --name BIOCLIP
srun python3 CLIP_eval_model_2.py --dataset PLANKTON --start_trial 0 --num_trials 5 --clip bioclip --name BIOCLIP2_