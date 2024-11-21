import simulate_network
import numpy as np
from sklearn.model_selection import ParameterGrid

config_name = 'default_config'

paramGrid = {'sim_name': ['fully connected'],
             'in_amp': [0.55],
             'sf_exc_gmax': [0.19],
             'fin_exc_gmax': [0.016],
             'fic_exc_gmax': [0.05],
             'icf_exc_gmax': [0.05],
             'icin_exc_gmax': [0.05],
             'inf_inh_gmax': [0.0045],
             'enable_loss': [True, False],
             'enable_IC': [True, False]}
batchParamsList = list(ParameterGrid(paramGrid))

for batchParams in batchParamsList:
    
    simulate_network.run_sim(config_name, batchParams)
