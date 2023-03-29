#!/bin/bash
#SBATCH --nodes=1
#SBATCH --time=5:0:0
#SBATCH --qos=gpu
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --partition=gpu
#SBATCH --mem 32000
#SBATCH --output=train.out

echo STARTING AT `date`

module load gcc cuda cudnn mvapich2
source ../venvs/cr_squad/bin/activate
srun python train.py 'dataset/dev-v1.1.json' 20 64
deactivate

echo FINISHED at `date`
