#!/bin/bash
#SBATCH -N 1
#SBATCH -c 16
#SBATCH --mem=32G
#SBATCH -t 1-00:00:00
#SBATCH -p public
#SBATCH -q public
#SBATCH -e slurm.%j.err
#SBATCH --mail-type=ALL
#SBATCH --export=NONE

module load mamba/latest
source activate python3_10_cochlea

# python ~/DCN_model/cnmodel_network.py --hearing normal -c 1 -i 10 -if 22000 -rt III 


# python ~/DCN_model/cnmodel_network.py --hearing loss --loss_limit 13000 -c 1 -i 10 -if 22000 -rt III -f --hp_loss --hf_loss 1.01
# python ~/DCN_model/cnmodel_network.py --hearing loss --loss_limit 13000 -c 1 -i 10 -if 22000 -rt III -f --hp_loss --hf_loss 1.1
# python ~/DCN_model/cnmodel_network.py --hearing loss --loss_limit 13000 -c 1 -i 10 -if 22000 -rt III -f --hp_loss --hf_loss 1.3
# python ~/DCN_model/cnmodel_network.py --hearing loss --loss_limit 13000 -c 1 -i 10 -if 22000 -rt III -f --hp_loss --hf_loss 1.5

# python ~/DCN_model/cnmodel_network.py --hearing loss --loss_limit 13000 -c 1 -i 10 -if 22000 -rt III -f --hp_loss --hf_loss 1.1 --cohc 0.25 --cihc 0.6
# python ~/DCN_model/cnmodel_network.py --hearing loss --loss_limit 13000 -c 1 -i 10 -if 22000 -rt III -f --hp_loss --hf_loss 1.3 --cohc 0.25 --cihc 0.6
# python ~/DCN_model/cnmodel_network.py --hearing loss --loss_limit 13000 -c 1 -i 10 -if 22000 -rt III -f --hp_loss --hf_loss 2.0 --cohc 0.25 --cihc 0.6

# python ~/DCN_model/cnmodel_network.py --hearing loss --loss_limit 13000 -c 1 -i 10 -if 22000 -rt III -f --hp_loss --hf_loss 1.7 --cohc 0.25 --cihc 0.6 --sgn_loss 60
# python ~/DCN_model/cnmodel_network.py --hearing loss --loss_limit 13000 -c 1 -i 10 -if 22000 -rt III -f --hp_loss --hf_loss 1.1 --cohc 0.25 --cihc 0.6 --sgn_loss 60
# python ~/DCN_model/cnmodel_network.py --hearing loss --loss_limit 13000 -c 1 -i 10 -if 22000 -rt III -f --hp_loss --hf_loss 2.0 --cohc 0.25 --cihc 0.6 --sgn_loss 60


python ~/DCN_model/cnmodel_network.py --hearing loss --loss_limit 13000 -c 1 -i 10 -if 22000 -rt III -f --sgn_loss 40 --cohc 0.25 --cihc 0.6 --hp_loss --hf_loss 1.5
python ~/DCN_model/cnmodel_network.py --hearing loss --loss_limit 13000 -c 1 -i 10 -if 22000 -rt III -f --sgn_loss 40 --cohc 0.25 --cihc 0.6 --hp_loss --hf_loss 2.0

python ~/DCN_model/cnmodel_network.py --hearing loss --loss_limit 13000 -c 1 -i 10 -if 22000 -rt III -f --sgn_loss 50 --cohc 0.25 --cihc 0.6 --hp_loss --hf_loss 1.5
python ~/DCN_model/cnmodel_network.py --hearing loss --loss_limit 13000 -c 1 -i 10 -if 22000 -rt III -f --sgn_loss 50 --cohc 0.25 --cihc 0.6 --hp_loss --hf_loss 2.0

# python ~/DCN_model/cnmodel_network.py --hearing loss --loss_limit 13000 -c 1 -i 10 -if 22000 -rt IV --cohc 0.25 --cihc 0.5 -f --sgn_loss 60
# python ~/DCN_model/cnmodel_network.py --hearing loss --loss_limit 13000 -c 1 -i 10 -if 22000 -rt IV --cohc 0.25 --cihc 0.5 -f --sgn_loss 70

# python ~/DCN_model/cnmodel_network.py --hearing loss --loss_limit 13000 -c 1 -i 10 -if 22000 -rt IV --cohc 0.35 --cihc 0.6 -f --hp_loss --hf_loss 1.01
# python ~/DCN_model/cnmodel_network.py --hearing loss --loss_limit 13000 -c 1 -i 10 -if 22000 -rt IV --cohc 0.35 --cihc 0.6 -f --hp_loss --hf_loss 1.001

# python ~/DCN_model/cnmodel_network.py --hearing loss --loss_limit 13000 -c 1 -i 10 -if 22000 -rt IV --cohc 0.35 --cihc 0.5 -f --hp_loss --hf_loss 1.01
# python ~/DCN_model/cnmodel_network.py --hearing loss --loss_limit 13000 -c 1 -i 10 -if 22000 -rt IV --cohc 0.35 --cihc 0.5 -f --hp_loss --hf_loss 1.001

# python ~/DCN_model/cnmodel_network.py --hearing loss --loss_limit 13000 -c 1 -i 10 -if 22000 -f --cohc 1.0 --cihc 0.25 --hp_loss --hf_loss 1.0
# python ~/DCN_model/cnmodel_network.py --hearing loss --loss_limit 13000 -c 1 -i 10 -if 22000 -f --cohc 1.0 --cihc 0.25 --hp_loss --hf_loss 1.01
# python ~/DCN_model/cnmodel_network.py --hearing loss --loss_limit 13000 -c 1 -i 10 -if 22000 -f --cohc 1.0 --cihc 0.25 --hp_loss --hf_loss 2
# python ~/DCN_model/cnmodel_network.py --hearing loss --loss_limit 13000 -c 1 -i 10 -if 22000 -f --cohc 1.0 --cihc 0.25 --hp_loss --hf_loss 1.3
# python ~/DCN_model/cnmodel_network.py --hearing loss --loss_limit 13000 -c 1 -i 10 -if 22000 -f --cohc 1.0 --cihc 0.25 --hp_loss --hf_loss 1.4
# python ~/DCN_model/cnmodel_network.py --hearing loss --loss_limit 13000 -c 1 -i 10 -if 22000 -f --cohc 1.0 --cihc 0.25 --hp_loss --hf_loss 1.5


# python ~/DCN_model/cnmodel_network.py --hearing loss --loss_limit 13000 -c 1 -i 1 -if 22000 -f --cohcs
# python ~/DCN_model/cnmodel_network.py --hearing loss --loss_limit 13000 -c 1 -i 1 -if 22000 -f --cihcs
# python ~/DCN_model/cnmodel_network.py --hearing loss --loss_limit 13000 -c 1 -i 1 -if 22000 -f --cohcs --cihcs

# python ~/DCN_model/cnmodel_network.py --hearing loss --loss_limit 13000 -c 1 -i 5 -if 22000 -f --cohc 0.25 --cihc 1


# python ~/DCN_model/cnmodel_network.py --hearing loss --loss_limit 13000 -c 1 -i 1 -if 0 -f --cohc 1 --cihc 0.04 -rm
# python ~/DCN_model/cnmodel_network.py --hearing loss --loss_limit 13000 -c 1 -i 1 -if 0 -f --cohc 0.1 --cihc 1 -rm 
# python ~/DCN_model/cnmodel_network.py --hearing loss --loss_limit 13000 -c 1 -i 1 -if 0 -f --cohc 0.1 --cihc 0.04 -rm