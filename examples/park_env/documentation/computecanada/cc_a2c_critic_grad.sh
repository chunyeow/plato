#!/bin/bash
#SBATCH --time=10:00:00
#SBATCH --cpus-per-task=8
#SBATCH --nodes=1
#SBATCH --mem=128G
#SBATCH --account=def-baochun
#SBATCH --output=results_critic_grad_seed_5.out

module load gcc/9.3.0 arrow cuda/11 python/3.9 scipy-stack
source ~/.federated/bin/activate
python examples/park_env/a2c.py -c examples/park_env/a2c_critic_grad_seed_5.yml

