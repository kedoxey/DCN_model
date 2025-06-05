#!/bin/bash
#SBATCH -N 1
#SBATCH -c 16
#SBATCH --mem=64G
#SBATCH -t 1-00:00:00
#SBATCH -p general
#SBATCH -q public
#SBATCH -e slurm.%j.err
#SBATCH --mail-type=ALL
#SBATCH --export=NONE

module load mamba/latest
source activate python3_10

python ~/DCN_model/notebooks/cnmodel_pyramidal_response.py
