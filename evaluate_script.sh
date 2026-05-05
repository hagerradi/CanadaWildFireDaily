#!/bin/bash
#SBATCH --job-name=eval_script
#SBATCH --output=logs/train_%A_%a.out
#SBATCH --error=logs/train_%A_%a.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32Gb
#SBATCH --time=02:00:00
#SBATCH --mail-user=mahacine.ettahri@mila.quebec
#SBATCH --mail-type=BEGIN,END,FAIL

# Load environment
module load python/3.10
source PATH/TO/VENV

cd /PATH/TO/PROJECT/FOLDER

python -m src.evaluate --config configs/default.yaml