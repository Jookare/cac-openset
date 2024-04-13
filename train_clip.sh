#!/bin/bash
#SBATCH --job-name=Plankton
#SBATCH --account=project_2002784
#SBATCH --partition=gpu
#SBATCH --time=02:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=3G
#SBATCH --gres=gpu:v100:1

module load pytorch/2.2
# pip install git+https://github.com/openai/CLIP.git
srun python3 CLIP_train_model.py --dataset PLANKTON --trial 0  --clip clip --name CLIP_test2_
srun python3 CLIP_train_model.py --dataset PLANKTON --trial 1  --clip clip --name CLIP_test2_
srun python3 CLIP_train_model.py --dataset PLANKTON --trial 2  --clip clip --name CLIP_test2_
srun python3 CLIP_train_model.py --dataset PLANKTON --trial 3  --clip clip --name CLIP_test2_
srun python3 CLIP_train_model.py --dataset PLANKTON --trial 4  --clip clip --name CLIP_test2_

# srun python3 CLIP_train_model_2.py --dataset PLANKTON --trial 0  --clip clip --name CLIP2_
# srun python3 CLIP_train_model_2.py --dataset PLANKTON --trial 1  --clip clip --name CLIP2_
# srun python3 CLIP_train_model_2.py --dataset PLANKTON --trial 2  --clip clip --name CLIP2_
# srun python3 CLIP_train_model_2.py --dataset PLANKTON --trial 3  --clip clip --name CLIP2_
# srun python3 CLIP_train_model_2.py --dataset PLANKTON --trial 4  --clip clip --name CLIP2_