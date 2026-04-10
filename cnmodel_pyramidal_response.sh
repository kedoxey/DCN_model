#!/bin/bash
#SBATCH -N 1
#SBATCH -c 32
#SBATCH --mem=128G
#SBATCH -t 2-00:00:00
#SBATCH -p public
#SBATCH -q public
#SBATCH -e slurm.%j.err
#SBATCH --mail-type=ALL
#SBATCH --export=NONE

module load mamba/latest
source activate python3_10

# python ~/DCN_model/cnmodel_pyramidal_response.py --hearing normal -i 5 -cf 22000 -f
# python ~/DCN_model/cnmodel_pyramidal_response.py --hearing normal -i 10 -cf 22000 -if 22000 -idb 90 -f

python ~/DCN_model/cnmodel_pyramidal_response.py --hearing normal -i 10 -cf 22000 -if 16000 -rl
python ~/DCN_model/cnmodel_pyramidal_response.py --hearing normal -i 10 -cf 22000 -if 22000 -rl
python ~/DCN_model/cnmodel_pyramidal_response.py --hearing normal -i 10 -cf 22000 -if 28000 -rl

# python ~/DCN_model/cnmodel_pyramidal_response.py --hearing loss --loss_limit 13000 -i 10 -cf 22000 -if 16000 -rl --cohc 0.5 --cihc 1.0 -f
# python ~/DCN_model/cnmodel_pyramidal_response.py --hearing loss --loss_limit 13000 -i 10 -cf 22000 -if 16000 -rl --cohc 1.0 --cihc 0.5 -f
# python ~/DCN_model/cnmodel_pyramidal_response.py --hearing loss --loss_limit 13000 -i 10 -cf 22000 -if 16000 -rl --cohc 0.25 --cihc 0.25 -f

# python ~/DCN_model/cnmodel_pyramidal_response.py --hearing loss --loss_limit 13000 -i 10 -cf 22000 -if 22000 -rl --cohc 0.5 --cihc 1.0 -f
# python ~/DCN_model/cnmodel_pyramidal_response.py --hearing loss --loss_limit 13000 -i 10 -cf 22000 -if 22000 -rl --cohc 1.0 --cihc 0.5 -f

# python ~/DCN_model/cnmodel_pyramidal_response.py --hearing loss --loss_limit 13000 -i 10 -cf 22000 -if 22000 -rl --cohc 0.25 --cihc 1.0 -f
# python ~/DCN_model/cnmodel_pyramidal_response.py --hearing loss --loss_limit 13000 -i 10 -cf 22000 -if 22000 -rl --cohc 1.0 --cihc 0.25 -f

# python ~/DCN_model/cnmodel_pyramidal_response.py --hearing loss --loss_limit 13000 -i 10 -cf 22000 -if 22000 -rl --cohc 0.25 --cihc 0.25 -f

# python ~/DCN_model/cnmodel_pyramidal_response.py --hearing loss --loss_limit 13000 -i 10 -cf 22000 -if 28000 -rl --cohc 0.5 --cihc 1.0 -f
# python ~/DCN_model/cnmodel_pyramidal_response.py --hearing loss --loss_limit 13000 -i 10 -cf 22000 -if 28000 -rl --cohc 1.0 --cihc 0.5 -f
# python ~/DCN_model/cnmodel_pyramidal_response.py --hearing loss --loss_limit 13000 -i 10 -cf 22000 -if 28000 -rl --cohc 0.25 --cihc 0.25 -f
