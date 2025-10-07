from . import simulate_network
import numpy as np
from sklearn.model_selection import ParameterGrid

config_name = 'dcn_config'

# TODO: add one connection at a time

conn = None

paramGrid = {'sim_name': [f'all_conns-bkg-pref_stim'],
             'num_cells': [800],
             'in_amp': [0],
             'bkg_rate_P': [30],
             'bkg_rate_AN': [13],
             'bkg_weight': [0.05],
             'nsa_freq': [0],
             'enable_loss': [False],
             'enable_IC': [False],
             'enable_conns': [True],
             'single_conn': [conn]}

batchParamsList = list(ParameterGrid(paramGrid))

for batchParams in batchParamsList:
    
    simulate_network.run_sim(config_name, batchParams)
