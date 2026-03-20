#!/bin/bash
#SBATCH -N 1
#SBATCH -c 32
#SBATCH --mem=128G
#SBATCH -t 1-00:00:00
#SBATCH -p public
#SBATCH -q public
#SBATCH -e slurm.%j.err
#SBATCH --mail-type=ALL
#SBATCH --export=NONE

module load mamba/latest
source activate python3_10

# python ~/DCN_model/cnmodel_pyramidal_response.py --hearing normal -i 5 -cf 22000 -f
# python ~/DCN_model/cnmodel_pyramidal_response.py --hearing normal -i 10 -cf 22000 -if 22000 -idb 90 -f

python ~/DCN_model/cnmodel_pyramidal_response.py --hearing normal -i 10 -cf 22000 -if 16000 -rl -f
python ~/DCN_model/cnmodel_pyramidal_response.py --hearing normal -i 10 -cf 22000 -if 22000 -rl -f
python ~/DCN_model/cnmodel_pyramidal_response.py --hearing normal -i 10 -cf 22000 -if 28000 -rl -f
