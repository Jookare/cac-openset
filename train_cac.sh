#!/bin/bash
#SBATCH --job-name=Plankton
#SBATCH --account=project_2002784
#SBATCH --partition=gpu
#SBATCH --time=08:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=3G
#SBATCH --gres=gpu:v100:1

module load pytorch/2.2
srun python3 CAC_train_model.py --dataset PLANKTON --trial 0 --name CAC_
srun python3 CAC_train_model.py --dataset PLANKTON --trial 1 --name CAC_
srun python3 CAC_train_model.py --dataset PLANKTON --trial 2 --name CAC_
srun python3 CAC_train_model.py --dataset PLANKTON --trial 3 --name CAC_
srun python3 CAC_train_model.py --dataset PLANKTON --trial 4 --name CAC_
