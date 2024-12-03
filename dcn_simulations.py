import simulate_network
import numpy as np
from sklearn.model_selection import ParameterGrid

config_name = 'dcn_config'

paramGrid = {'sim_name': ['no_conns-Poisson'],
             'num_cells': [800],
             'in_amp': [0],
             'bkg_rate': [26],
             'bkg_weight': [0.3],
             'nsa_freq': [23],
             'enable_loss': [False],
             'enable_IC': [False],
             'enable_conns': [False]}

batchParamsList = list(ParameterGrid(paramGrid))

for batchParams in batchParamsList:
    
    simulate_network.run_sim(config_name, batchParams)
