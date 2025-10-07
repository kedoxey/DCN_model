from neuron import h, load_mechanisms
from netpyne import specs, sim
import matplotlib.pyplot as plt
import os
import numpy as np
import random
import pandas as pd
from . import model_helpers as mh
from numpy.random import default_rng

seed = 74
random.seed(seed)
np.random.seed(seed)

h.load_file("stdrun.hoc")

save_fig = True

n_cells = 1
center_cell = n_cells//2

stim_delay = 10
stim_dur = 200
sim_dur = stim_dur + stim_delay*2

dt = 0.025

i_weights = [0.5]  #, 1, 1.5, 2]  # [0.5, 1, 1.5, 2], [1, 2.25, 3.5, 4.75]
weights_label = f'{i_weights[0]}_{i_weights[-1]}' if len(i_weights) > 1 else f'{i_weights[0]}'

nsa_freq = 80

bf_scale = 0.2
bfs = mh.define_input_freqs(9)
# bfs = [0, 50, 100, 150, 200, 250, 300]

sim_name = 'no_W-np_params'
sim_flag = f'{dt}_{sim_dur}ms-NSA_{nsa_freq}Hz-BFx{bf_scale}'
sim_label = f'Ix{weights_label}-{n_cells}cells-{sim_flag}'

output_dir, data_dir, sim_dir = mh.get_output_dir(sim_name, sim_label)

cwd = os.getcwd()
mod_dir = f'{cwd}/mod'
load_mechanisms(mod_dir)

cfg = specs.SimConfig()	
cfg.duration = sim_dur				                 
cfg.dt = dt								                # Internal integration timestep to use
cfg.verbose = True							                # Show detailed messages
cfg.recordCells = ['all']  
cfg.recordTraces = {'V_soma': {'sec': 'soma', 'loc': 0.5, 'var': 'v'}}
cfg.recordStep = dt
# cfg.recordStim = True
# cfg.filename = os.path.join(sim_dir, f'{sim_label}-DCN') 	# Set file output name
cfg.savePickle = False
# cfg.analysis['plotTraces'] = {'include': ['all'], 'saveFig': False, 'showFig': False}  # Plot recorded traces for this list of cells
# cfg.analysis['recordTraces'] = 
# cfg.analysis['plotSpikeFreq'] = {'include': ['all'], 'saveFig': True, 'showFig': True}
cfg.hParams['celsius'] = 34.0 
cfg.hParams['v_init'] = -60

netParams = specs.NetParams()

IzhCell = {'secs': {}}
IzhCell['secs']['soma'] = {'geom': {}, 'pointps': {}}                        # soma params dict
IzhCell['secs']['soma']['geom'] = {'diam': 10.0, 'L': 10.0, 'cm': 31.831}    # soma geometry, cm = 31.831
IzhCell['secs']['soma']['pointps']['Izhi'] = {                               # soma Izhikevich properties
    'mod':'Izhi2007b',
    'C':1,
    'k': 0.7,
    'vr':-60,
    'vt':-40,
    'vpeak':35,
    'a':0.03,
    'b':-2,
    'c':-50,
    'd':100,
    'celltype':1}
# IzhCell['secs']['soma']['threshold'] = -20
netParams.cellParams['IzhCell'] = IzhCell 

cell_types = ['P', 'I2']  #['P', 'I2', 'W', 'AN']
# record_cells = [(f'{cell_type}_pop', n_cells) for cell_type in cell_types]

pop_labels_nums = {cell_type: n_cells for cell_type in cell_types}

for pop_label, pop_num in list(pop_labels_nums.items()):
    netParams.popParams[f'{pop_label}_pop'] = {'cellType': 'IzhCell',
                                                'numCells': pop_num}


e_syn_weight = 0.2

netParams.popParams[f'vecstim_NSA'] = {
            'cellModel': 'VecStim',
            'numCells': n_cells,  # int(len(syn_secs)/4),
            'spikePattern': {'type': 'poisson',
                            'start': stim_delay,
                            'stop': stim_dur+stim_delay,
                            'frequency': nsa_freq}  #  np.random.randint(params.spk_freq_lb, params.spk_freq_ub, 1)[0]}
        }

netParams.connParams[f'vecstim_NSA->P'] = {
    'preConds': {'pop': f'vecstim_NSA'},
    'postConds': {'pop': 'P_pop'},
    'sec': 'soma',
    'synsPerConn': 1,
    'synMech': 'exc',
    'weight': e_syn_weight,  # 
    # 'synMechWeightFactor': [0.5,0.5],
    'delay': 'defaultDelay + dist_2D/propVelocity',
    'connList': [[i,i] for i in range(n_cells)]
    }


cell_ids = {'ANF': [i for i in range(n_cells)],
            'I2': [i for i in range(n_cells)],
            'P': [i for i in range(n_cells)]}

num_octaves = 4
bw_octs = {'ANF': 0.4, 'I2': 0.6}
n_scales = {'ANF': 0.6, 'I2': 0.175}

if n_cells == 10:

    bw_nums = {'ANF': int(bw_octs['ANF']*n_cells), 'I2': int(bw_octs['I2']*n_cells)}
    ns = {'ANF': int(bw_nums['ANF']*n_scales['ANF']), 'I2': int(bw_nums['I2']*n_scales['I2'])}

else:

    bw_nums = {'ANF': int(bw_octs['ANF'] / (num_octaves/n_cells)), 'I2': int(bw_octs['I2'] / (num_octaves/n_cells))}
    ns = {'ANF': int(np.floor(n_scales['ANF']*bw_nums['ANF'])), 'I2': int(np.floor(n_scales['I2']*bw_nums['I2']))}

conns_list = {'ANF_I2': [],
              'ANF_P': [],
              'I2_P': []}


for conn in list(conns_list.keys()):

    source = conn.split('_')[0]
    source_ids = cell_ids[source]

    target = conn.split('_')[1]
    target_ids = cell_ids[target]

    bw = bw_nums[source]
    if bw == 1: bw += 1
    bw_split = bw//2

    conn_list = []

    if n_cells == 1:

        conn_list.extend([i, j] for i, j in zip(source_ids, target_ids))

    else:
    
        for target_id in target_ids:

            n_source = ns[source]
            if n_source == 0: n_source += 1

            lb = target_id - bw_split
            if lb < 0: lb = 0
            ub = target_id + bw_split+1

            source_pool = source_ids[lb:ub]

            # if n_source > bw/len(source_pool):
            n_source *= (len(source_pool)/bw)

            source_rand = random.sample(source_pool, int(np.floor(n_source)))

            conn_list.extend([[i, target_id] for i in source_rand])
        
    conns_list[conn] = conn_list


rng = default_rng(seed)

rand_is = rng.choice(n_cells, size=n_cells, replace=False)
if n_cells == 1:
    anf_freqs_orig = [40]
else:
    anf_freqs_orig = [0 for i in range(n_cells)]

for j, rand_i in enumerate(rand_is):

    if j < n_cells//2:
        anf_freqs_orig[rand_i] = 0
    else:
        anf_freqs_orig[rand_i] = np.random.uniform(20, 40)

# e_tau = 0.2
# i_tau = 3
# netParams.synMechParams['exc'] = {'mod': 'ExpSyn', 'tau': e_tau, 'e': 10}
# netParams.synMechParams['inh'] = {'mod': 'ExpSyn', 'tau': i_tau, 'e': -70}

netParams.synMechParams['exc'] = {'mod': 'Exp2Syn', 'tau1': 0.1, 'tau2': 1.0, 'e': 0}  
netParams.synMechParams['inh'] = {'mod': 'Exp2Syn', 'tau1': 0.1, 'tau2': 1.0, 'e':-80}  


# num_spikes = {}


# bfs = [0, 10, 20, 30, 40, 50, 60]  #, 70, 80, 90]
# bf_scale = 0.2
middle_is = [i for i in range(n_cells)]

num_spikes_path = os.path.join(data_dir, f'{sim_label}-num_spikes.npy')
# if os.path.exists(num_spikes_path):
#     num_spikes = np.load(num_spikes_path, allow_pickle=True).item()
# else:
num_spikes = {}

num_sims = 0

for i_weight in i_weights:

    if i_weight not in list(num_spikes.keys()):
        num_spikes[i_weight] = {}

    for bf in bfs:

        # if (i_weight == 0.5) and (bf == 50):
        #     save_fig = True
        # else:
        #     save_fig = False

        for middle_i in middle_is:
        
            anf_freqs = anf_freqs_orig[:]

            if bf not in list(num_spikes[i_weight].keys()):
                num_spikes[i_weight][bf] = {}

            lb = middle_i - 2
            if lb < 0: lb = 0
            ub = middle_i + 3
            input_cells = cell_ids['P'][lb:ub]

            input_scales = [1 - bf_scale*np.abs(cell_i-middle_i) for cell_i in range(n_cells) if np.abs(cell_i-middle_i) <= 2]

            inputs = {input_cell: bf*input_scale for input_cell, input_scale in zip(input_cells, input_scales)} 

            for i, anf_freq in enumerate(anf_freqs):
                
                if i in list(inputs.keys()):
                    anf_freqs[i] += inputs[i]

            print(anf_freqs)

            for i, anf_freq in enumerate(anf_freqs):

                netParams.popParams[f'vecstim_ANF{i}'] = {
                    'cellModel': 'VecStim',
                    'numCells': 1,  # int(len(syn_secs)/4),
                    'spikePattern': {'type': 'poisson',
                                    'start': stim_delay,
                                    'stop': stim_dur+stim_delay,
                                    'frequency': anf_freq}  #  np.random.randint(params.spk_freq_lb, params.spk_freq_ub, 1)[0]}
                }

                for cell_type in cell_types:

                    cell_conns = [conn for conn in conns_list[f'ANF_{cell_type}'] if conn[0] == i]

                    netParams.connParams[f'vecstim_ANF{i}->{cell_type}'] = {
                        'preConds': {'pop': f'vecstim_ANF{i}'},
                        'postConds': {'pop': f'{cell_type}_pop'},
                        'sec': 'soma',
                        'synsPerConn': 1,
                        'synMech': 'exc',
                        'weight': e_syn_weight,  # 
                        # 'synMechWeightFactor': [0.5,0.5],
                        'delay': 'defaultDelay + dist_2D/propVelocity',
                        'connList': [[0, cell_conn[1]] for cell_conn in cell_conns]  # [[0,i] for i in range(n_cells)]
                    }

       
            netParams.connParams['I2->P'] = {
                'preConds': {'pop': 'I2_pop'},
                'postConds': {'pop': 'P_pop'},
                'synsPerConn': 1,
                'synMech': 'inh',
                'weight': i_weight,
                # 'probability': 1.0
                'connList': conns_list['I2_P']
            }

            (pops, cells, conns, stims, simData) = sim.createSimulateAnalyze(netParams=netParams, simConfig=cfg, output=True)
            num_sims += 1

            t = np.array(simData['t'])
            spkid = np.array(simData['spkid'])
            spkt = np.array(simData['spkt'])

            spikes = spkt[np.where(spkid == center_cell)]
            n_spikes = len(spikes)
            msf = (n_spikes - 1) / (spikes[-1] - spikes[0]) * 1000 if n_spikes > 0 else 0
            num_spikes[i_weight][bf][middle_i] = msf

            # for pop_i, (pop_label, pop_cells) in enumerate(pops.items()):

            #     cell_type = pop_label.split('_')[0]

            #     if 'vecstim' in cell_type:
            #         continue
                
            #     for cell_i, cell_id in enumerate(pop_cells.cellGids):

            #         spikes = spkt[np.where(spkid == center_cell)]
            #         n_spikes = len(spikes)
            #         msf = (n_spikes - 1) / (spikes[-1] - spikes[0]) * 1000 if n_spikes > 0 else 0
            #         if ('P' in cell_type) and (cell_id == middle_i):  # only record msf for middle P cell
            #             num_spikes[i_weight][bf][cell_id] = msf

            if save_fig:
                fig = mh.plot_cells(simData, cell_types, n_cells, pops, conns)
                fig.suptitle(f'{i_weight} x inh - {bf} Hz BF')
                fig.tight_layout()

                fig.savefig(os.path.join(sim_dir, f'{i_weight}xinh-{bf}Hz_{middle_i}BF.png'), dpi=300)


np.save(num_spikes_path, num_spikes)

print(f'!!! Ran {num_sims} simulations !!!')

# col_list = pd.MultiIndex.from_tuples([(i_w, bf) for i_w in i_weights for bf in bfs])  # [(bf, i_w) for i_w in num_spikes.keys() for bf in num_spikes[i_w].keys()])

# num_spikes_arr = np.ones((len(i_weights), n_cells, len(bfs)), dtype=int)

# for i, weight_vals in enumerate(num_spikes.values()):

#     for j, bf_vals in enumerate(weight_vals.values()):

#         for cell_id, n_spikes in bf_vals.items():

#             num_spikes_arr[i, cell_id, j] = n_spikes

# num_spikes_arr = np.concatenate((num_spikes_arr[0,:,:], num_spikes_arr[1,:,:]), axis=1)
# np.save('num_spikes_arr', num_spikes_arr)

# num_spikes_df = pd.DataFrame(num_spikes_arr, columns=col_list)
# num_spikes_df.to_pickle(f'num_spikes_BF.pkl')

# fig, axs = plt.subplots(1, len(i_weights), figsize=(10,8))

# p_spon_freq = 36
# scale = 6

# for i, i_w in enumerate(i_weights):

#     yticks = []

#     for bf, col in num_spikes_df[i_w].items():
#         # print(bf)
#         # print(col.values)

#         yticks.append(bf*scale)

#         diff = col.values - p_spon_freq
#         pos = diff.copy()
#         pos[pos < 0] = 0
#         neg = diff.copy()
#         neg[neg > 0] = 0
#         temp = 5

#         octaves = [i for i in range(n_cells)]

#         # axs[i].plot(col.values + bf*scale)
#         # axs[i].plot([p_spon_freq + bf*scale for i in col.values])
#         axs[i].fill_between(octaves, pos + bf*scale, [bf*scale for i in col.values], color='tab:blue', alpha=0.8)
#         axs[i].fill_between(octaves, neg + bf*scale, [bf*scale for i in col.values], color='tab:blue', alpha=0.4)

#     axs[i].set_title(f'{i_w} I2->P weight')
#     axs[i].set_xticks([i for i in range(n_cells)])
#     axs[i].set_xticklabels([i for i in range(1,n_cells+1)])
#     axs[i].set_xlabel('Octaves')
#     axs[i].set_yticks(yticks)
#     axs[i].set_yticklabels(bfs)

# axs[0].set_ylabel('Sound Level')

# fig.tight_layout()
# fig.savefig('BF_response_map.png', dpi=300)
