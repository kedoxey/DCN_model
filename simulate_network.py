import matplotlib.pyplot as plt
import os
import yaml
import numpy as np
from . import model_helpers as mh
import argparse as ap
from neuron import h, load_mechanisms
from netpyne import specs, sim


def run_sim(config_name, *batch_params):
    h.load_file("stdrun.hoc")

    cwd = os.getcwd()
    mod_dir = os.path.join(cwd, 'mod')
    load_mechanisms(mod_dir)
    ### Simulation configuration ###

    ### Import simulation config ###
    params = ap.Namespace(**mh.load_config(config_name))

    for batch_param, batch_value in list(batch_params[0].items()):
        setattr(params, batch_param, batch_value)

    sim_label = f'{params.in_amp}nA'
    if params.nsa_freq > 0:
        sim_label += f'-{params.nsa_freq}Hz'
    if params.bkg_rate_P > 0:
        sim_label += f'-{params.bkg_rate_P}P_{params.bkg_rate_AN}ANx{params.bkg_weight}bkg'
    else:
        sim_label += f'-{params.bkg_rate_AN}ANx{params.bkg_weight}bkg'
    sim_label += '-loss' if params.enable_loss else '-normal'

    output_dir, sim_dir = mh.get_output_dir(params.sim_name, sim_label)
    mh.write_config(params,sim_dir,sim_label,config_name)

    # center cell gid
    center_in = params.num_cells // 2

    # names of cell types and which cells to record voltage traces from
    cell_types = ['P', 'I2', 'W', 'AN']
    record_cells = [(f'{cell_type}_pop', [center_in-1, center_in, center_in+1]) for cell_type in cell_types]

    cfg = specs.SimConfig()	
    cfg.duration = params.sim_dur				                 
    cfg.dt = 0.05								                # Internal integration timestep to use
    cfg.verbose = True							                # Show detailed messages
    cfg.recordTraces = {'V_soma': {'sec': 'soma', 'loc': 0.5, 'var': 'v'}}
    cfg.recordStep = 0.1
    # cfg.recordStim = True
    cfg.filename = os.path.join(sim_dir, f'{sim_label}-DCN') 	# Set file output name
    cfg.savePickle = False
    cfg.analysis['plotTraces'] = {'include': record_cells, 'saveFig': False, 'showFig': False}  # Plot recorded traces for this list of cells
    # cfg.analysis['plotSpikeFreq'] = {'include': ['all'], 'saveFig': True, 'showFig': True}
    cfg.hParams['celsius'] = 34.0 
    cfg.hParams['v_init'] = -60

    ### Define cells and network ###
    netParams = specs.NetParams()

    IzhCell = {'secs': {}}
    IzhCell['secs']['soma'] = {'geom': {}, 'pointps': {}}                        # soma params dict
    IzhCell['secs']['soma']['geom'] = {'diam': params.diam, 'L': params.diam, 'cm': 1}    # soma geometry, cm = 31.831
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
    IzhCell['secs']['soma']['threshold'] = -10
    netParams.cellParams['IzhCell'] = IzhCell             # add dict to list of cell parameters                                  # add dict to list of cell parameters
    
    pop_labels_nums = {cell_type: params.num_cells for cell_type in cell_types}

    for pop_label, pop_num in list(pop_labels_nums.items()):
        netParams.popParams[f'{pop_label}_pop'] = {'cellType': 'IzhCell',
                                                   'numCells': pop_num}


    ### Synapses ###
    netParams.synMechParams['exc'] = {'mod': 'ExpSyn', 'tau': 3, 'e': -10}
    netParams.synMechParams['inh'] = {'mod': 'ExpSyn', 'tau': 10, 'e': -70}

    ### Connections ###
    freqs = mh.define_freqs(params.num_cells)

    conn_params = {'BW': {'AN_W': 2,
                      'AN_I2': 0.4,
                      'AN_P': 0.4,
                      'W_I2': 0.1,
                      'W_P': 0.1,
                      'I2_P': 0.6},
                    'N': {'AN_W': 140,
                          'AN_I2': 48,
                          'AN_P': 48,
                          'W_I2': 15,
                          'W_P': 15,
                          'I2_P': 21}}

    

    ### Excitatory ###
    if params.enable_conns:

        conns_list = mh.define_conns(params.num_cells, freqs, conn_params)

        if params.single_conn:
            if [center_in, center_in] not in conns_list[params.single_conn]:
                conns_list[params.single_conn].append([center_in, center_in])
        
        netParams.connParams['AN->W'] = {
            'preConds': {'pop': 'AN_pop'},
            'postConds': {'pop': 'W_pop'},
            'synsPerConn': 1,
            'synMech': 'exc',
            'weight': 0.06,
            'connList': conns_list['AN_W']
        }

        netParams.connParams['AN->I2'] = {
            'preConds': {'pop': 'AN_pop'},
            'postConds': {'pop': 'I2_pop'},
            'synsPerConn': 1,
            'synMech': 'exc',
            'weight': 0.55,
            'connList': conns_list['AN_I2']
        }

        netParams.connParams['AN->P'] = {
            'preConds': {'pop': 'AN_pop'},
            'postConds': {'pop': 'P_pop'},
            'synsPerConn': 1,
            'synMech': 'exc',
            'weight': 0.25,
            'connList': conns_list['AN_P']
        }

        ### Inhibitory ###
        netParams.connParams['W->I2'] = {
            'preConds': {'pop': 'W_pop'},
            'postConds': {'pop': 'I2_pop'},
            'synsPerConn': 1,
            'synMech': 'inh',
            'weight': 1.4,
            'connList': conns_list['W_I2']
        }

        netParams.connParams['W->P'] = {
            'preConds': {'pop': 'W_pop'},
            'postConds': {'pop': 'P_pop'},
            'synsPerConn': 1,
            'synMech': 'inh',
            'weight': 0.1,
            'connList': conns_list['W_P']
        }

        netParams.connParams['I2->P'] = {
            'preConds': {'pop': 'I2_pop'},
            'postConds': {'pop': 'P_pop'},
            'synsPerConn': 1,
            'synMech': 'inh',
            'weight': 2.25,
            'connList': conns_list['I2_P']
        }


    ### Input ###
    recip_conns = [[i,i] for i in range(params.num_cells)]

    # Principal cell spontaneous activity
    if params.nsa_freq > 0:

        netParams.popParams[f'vecstim_NSA'] = {
                'cellModel': 'VecStim',
                'numCells': 1,  # int(len(syn_secs)/4),
                'spikePattern': {'type': 'poisson',
                                'start': 0,
                                'stop': -1,
                                'frequency': params.nsa_freq}  #  np.random.randint(params.spk_freq_lb, params.spk_freq_ub, 1)[0]}
            }

        netParams.connParams[f'vecstim_NSA->P'] = {
            'preConds': {'pop': f'vecstim_NSA'},
            'postConds': {'pop': 'P_pop'},
            'sec': 'soma',
            'synsPerConn': 1,
            'synMech': 'exc',
            'weight': 1,  # 
            # 'synMechWeightFactor': [0.5,0.5],
            'delay': 'defaultDelay + dist_2D/propVelocity',
            'probability': 1.0,
            'connList': recip_conns
        }

    else:
        if params.bkg_rate_P > 0:
            netParams.stimSourceParams['bkg_P'] = {'type': 'NetStim', 'rate': params.bkg_rate_P, 'noise': 1}
            netParams.stimTargetParams['bkg_P->ALL'] = {'source': 'bkg_P', 'conds': {'pop': ['P_pop']}, 'weight': params.bkg_weight, 'delay': 0, 'synMech': 'exc'}

    
    # Auditory nerve fiber spontaneous activity
    if params.bkg_rate_AN > 0:
        netParams.stimSourceParams['bkg_AN'] = {'type': 'NetStim', 'rate': params.bkg_rate_AN, 'noise': 1}
        netParams.stimTargetParams['bkg_AN->ALL'] = {'source': 'bkg_AN', 'conds': {'pop': ['AN_pop']}, 'weight': params.bkg_weight, 'delay': 0, 'synMech': 'exc'}
    
    # Prefered stimulus 
    netParams.stimSourceParams['IClamp_high'] = {'type': 'IClamp', 'del': params.stim_delay, 'dur': params.stim_dur, 'amp': params.in_amp}
    netParams.stimTargetParams['IClamp_high->mid'] = {'source': 'IClamp_high', 'sec': 'soma', 'loc': 0.5, 'conds': {'pop': 'AN_pop', 'cellList': [center_in]}}

    netParams.stimSourceParams['IClamp_low'] = {'type': 'IClamp', 'del': params.stim_delay, 'dur': params.stim_dur, 'amp': params.in_amp/2}
    netParams.stimTargetParams['IClamp_low->side'] = {'source': 'IClamp_low', 'sec': 'soma', 'loc': 0.5, 'conds': {'pop': 'AN_pop', 'cellList': [center_in-1, center_in+1]}}


    # num_sgn_high = 40
    # sgn_high_ids = [i for i in range(params.num_cells-num_sgn_high-1,params.num_cells)]
    # if params.enable_loss:
    #     netParams.stimSourceParams['IClamp3'] = {'type': 'IClamp', 'del': 0, 'dur': params.sim_dur, 'amp': -(params.in_amp/2)}  # -0.3)}
    #     netParams.stimTargetParams['IClamp->SGNhigh'] = {'source': 'IClamp3', 'sec': 'soma', 'loc': 0.5, 'conds': {'pop': 'SGN_pop', 'cellList': sgn_high_ids}}

    ### Run simulation ###
    (pops, cells, conns, stims, simData) = sim.createSimulateAnalyze(netParams=netParams, simConfig=cfg, output=True)

    times = np.array(simData['spkt'])
    spikes = np.array(simData['spkid'])

    base_colors = ['tab:red', 'tab:green', 'tab:purple', 'tab:orange', 'tab:blue']
    colors = {f'{pop_label}_pop': base_colors[i] for i, pop_label in enumerate(pop_labels_nums.keys())}
    colors['vecstim_NSA'] = base_colors[-1]

    ### Plot spike frequencies ###
    pop_msfs = {}
    for pop_label, pop in list(pops.items()):

        driven_rates, spont_rates = mh.get_firing_rates(times, spikes, pop, params.stim_dur, params.stim_delay, params.sim_dur)

        
        if 'P_pop' in pop_label:
            mh.plot_firing_rates(driven_rates, spont_rates, pop, pop_label, sim_dir, sim_label, colors)

        pop_msf = mh.plot_spike_frequency(times, spikes, pop, pop_label, sim_dir, sim_label, colors)
        pop_msfs[pop_label] = {'driven': float(np.nanmean(driven_rates)),
                               'spontaneous': float(np.nanmean(spont_rates)),
                               'overall': float(pop_msf)}
        
        temp = 5
        

    with open(os.path.join(sim_dir, f'{sim_label}-pop_msfs.yml'), 'w') as outfile:
        yaml.dump(pop_msfs, outfile)

    ### Plot spike times ###
    mh.plot_spike_times(params.num_cells, times, spikes, pops, params.stim_delay, params.stim_dur, sim_dir, sim_label, colors)

    ### Plot voltage traces ###
    mh.plot_traces(simData, pops, record_cells, params.stim_delay, params.stim_dur, sim_dir, sim_label, colors)
    # return pop_msfs

