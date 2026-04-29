#!/bin/bash
#SBATCH --job-name=unet_age_array
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:l40s:1
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --array=0-2                   # Launching 3 parallel runs (0, 1, 2)

# Load environment
module load python/3.10
source /PATH/TO/env_wildfires_spread/bin/activate
cd /PATH/TO/Wildfires_Spread_v2

# Map Array Task ID to Run ID (0,1,2 -> 1,2,3)
RUN_ID=$(( SLURM_ARRAY_TASK_ID + 1 ))

echo "Starting Parallel Run ID: $RUN_ID"

export COMET_API_KEY=""

# Execute with dynamic overrides
python -m parallel_main \
    --config configs/default.yaml \
    --arch unet_age \
    --run_id $RUN_ID