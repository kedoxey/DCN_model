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

# python ~/DCN_model/cnmodel_network.py --hearing normal -c 1 -i 2 -if 22000 
# python ~/DCN_model/cnmodel_network.py --hearing normal -c 1 -i 10 -if 22000 -f --hf 0.3
# python ~/DCN_model/cnmodel_network.py --hearing normal -c 1 -i 10 -if 22000 -f --hf 0.2
# python ~/DCN_model/cnmodel_network.py --hearing normal -c 1 -i 10 -if 22000 -f --hf 0.1
# python ~/DCN_model/cnmodel_network.py --hearing normal -c 1 -i 10 -if 22000 -f --hf 1
# python ~/DCN_model/cnmodel_network.py --hearing normal -c 1 -i 10 -if 22000 -f --hf 1.5
# python ~/DCN_model/cnmodel_network.py --hearing normal -c 1 -i 10 -if 22000 -f --hf 2
# python ~/DCN_model/cnmodel_network.py --hearing normal -c 1 -i 10 -if 22000 -f --hf 2.5
# python ~/DCN_model/cnmodel_network.py --hearing normal -c 1 -i 10 -if 22000 -f --hf 3

python ~/DCN_model/cnmodel_network.py --hearing loss --loss_limit 13000 -c 1 -i 10 -if 22000 -f --hp_loss --hf_loss 1.1
python ~/DCN_model/cnmodel_network.py --hearing loss --loss_limit 13000 -c 1 -i 10 -if 22000 -f --hp_loss --hf_loss 1.2
python ~/DCN_model/cnmodel_network.py --hearing loss --loss_limit 13000 -c 1 -i 10 -if 22000 -f --hp_loss --hf_loss 1.3
python ~/DCN_model/cnmodel_network.py --hearing loss --loss_limit 13000 -c 1 -i 10 -if 22000 -f --hp_loss --hf_loss 1.4
python ~/DCN_model/cnmodel_network.py --hearing loss --loss_limit 13000 -c 1 -i 10 -if 22000 -f --hp_loss --hf_loss 1.5

python ~/DCN_model/cnmodel_network.py --hearing loss --loss_limit 13000 -c 1 -i 10 -if 22000 -f --cohc 0.25 --cihc 1.0 --hp_loss --hf_loss 1.0
python ~/DCN_model/cnmodel_network.py --hearing loss --loss_limit 13000 -c 1 -i 10 -if 22000 -f --cohc 0.25 --cihc 1.0 --hp_loss --hf_loss 1.1
python ~/DCN_model/cnmodel_network.py --hearing loss --loss_limit 13000 -c 1 -i 10 -if 22000 -f --cohc 0.25 --cihc 1.0 --hp_loss --hf_loss 1.2
python ~/DCN_model/cnmodel_network.py --hearing loss --loss_limit 13000 -c 1 -i 10 -if 22000 -f --cohc 0.25 --cihc 1.0 --hp_loss --hf_loss 1.3
python ~/DCN_model/cnmodel_network.py --hearing loss --loss_limit 13000 -c 1 -i 10 -if 22000 -f --cohc 0.25 --cihc 1.0 --hp_loss --hf_loss 1.4
python ~/DCN_model/cnmodel_network.py --hearing loss --loss_limit 13000 -c 1 -i 10 -if 22000 -f --cohc 0.25 --cihc 1.0 --hp_loss --hf_loss 1.5

python ~/DCN_model/cnmodel_network.py --hearing loss --loss_limit 13000 -c 1 -i 10 -if 22000 -f --cohc 1.0 --cihc 0.25 --hp_loss --hf_loss 1.0
python ~/DCN_model/cnmodel_network.py --hearing loss --loss_limit 13000 -c 1 -i 10 -if 22000 -f --cohc 1.0 --cihc 0.25 --hp_loss --hf_loss 1.1
python ~/DCN_model/cnmodel_network.py --hearing loss --loss_limit 13000 -c 1 -i 10 -if 22000 -f --cohc 1.0 --cihc 0.25 --hp_loss --hf_loss 1.2
python ~/DCN_model/cnmodel_network.py --hearing loss --loss_limit 13000 -c 1 -i 10 -if 22000 -f --cohc 1.0 --cihc 0.25 --hp_loss --hf_loss 1.3
python ~/DCN_model/cnmodel_network.py --hearing loss --loss_limit 13000 -c 1 -i 10 -if 22000 -f --cohc 1.0 --cihc 0.25 --hp_loss --hf_loss 1.4
python ~/DCN_model/cnmodel_network.py --hearing loss --loss_limit 13000 -c 1 -i 10 -if 22000 -f --cohc 1.0 --cihc 0.25 --hp_loss --hf_loss 1.5


# python ~/DCN_model/cnmodel_network.py --hearing loss --loss_limit 13000 -c 1 -i 1 -if 22000 -f --cohcs
# python ~/DCN_model/cnmodel_network.py --hearing loss --loss_limit 13000 -c 1 -i 1 -if 22000 -f --cihcs
# python ~/DCN_model/cnmodel_network.py --hearing loss --loss_limit 13000 -c 1 -i 1 -if 22000 -f --cohcs --cihcs

# python ~/DCN_model/cnmodel_network.py --hearing loss --loss_limit 13000 -c 1 -i 5 -if 22000 -f --cohc 0.25 --cihc 1


# python ~/DCN_model/cnmodel_network.py --hearing loss --loss_limit 13000 -c 1 -i 1 -if 0 -f --cohc 1 --cihc 0.04 -rm
# python ~/DCN_model/cnmodel_network.py --hearing loss --loss_limit 13000 -c 1 -i 1 -if 0 -f --cohc 0.1 --cihc 1 -rm 
# python ~/DCN_model/cnmodel_network.py --hearing loss --loss_limit 13000 -c 1 -i 1 -if 0 -f --cohc 0.1 --cihc 0.04 -rm