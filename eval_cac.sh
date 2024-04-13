#!/bin/bash
#SBATCH --job-name=test
#SBATCH --account=project_2002784
#SBATCH --partition=gputest
#SBATCH --time=00:15:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=3G
#SBATCH --gres=gpu:v100:1

module load pytorch/2.2
srun python3 CAC_eval_model.py --dataset PLANKTON --start_trial 0 --num_trials 5 --name CAC