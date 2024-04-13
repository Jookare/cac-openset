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
# srun python3 CLIP_train_model.py --dataset PLANKTON --trial 0 --clip bioclip --name BIOCLIP
# srun python3 CLIP_train_model.py --dataset PLANKTON --trial 1 --clip bioclip --name BIOCLIP
# srun python3 CLIP_train_model.py --dataset PLANKTON --trial 2 --clip bioclip --name BIOCLIP
# srun python3 CLIP_train_model.py --dataset PLANKTON --trial 3 --clip bioclip --name BIOCLIP
# srun python3 CLIP_train_model.py --dataset PLANKTON --trial 4 --clip bioclip --name BIOCLIP

srun python3 CLIP_train_model_2.py --dataset PLANKTON --trial 0  --clip bioclip --name BIOCLIP2_
srun python3 CLIP_train_model_2.py --dataset PLANKTON --trial 1  --clip bioclip --name BIOCLIP2_
srun python3 CLIP_train_model_2.py --dataset PLANKTON --trial 2  --clip bioclip --name BIOCLIP2_
srun python3 CLIP_train_model_2.py --dataset PLANKTON --trial 3  --clip bioclip --name BIOCLIP2_
srun python3 CLIP_train_model_2.py --dataset PLANKTON --trial 4  --clip bioclip --name BIOCLIP2_