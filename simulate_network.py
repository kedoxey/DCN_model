import matplotlib.pyplot as plt
import os
import yaml
import numpy as np
import model_helpers as mh
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

    for batch_param, batch_value in batch_params[0].items():
        setattr(params, batch_param, batch_value)

    # TODO: update sim_label
    sim_label = f'SF{params.sf_exc_gmax}-FInt{params.fin_exc_gmax}-FIC{params.fic_exc_gmax}-ICF{params.icf_exc_gmax}-ICInt{params.icin_exc_gmax}-IntF{params.inf_inh_gmax}-{params.in_amp}nA-full_network'
    sim_label += '_loss' if params.enable_loss else '_normal'
    if not params.enable_IC: sim_label += '_no-IC'

    output_dir, sim_dir = mh.get_output_dir(params.sim_name, sim_label)
    mh.write_config(params,sim_dir,sim_label,config_name)

    cfg = specs.SimConfig()	
    cfg.duration = params.sim_dur				                 
    cfg.dt = 0.05								                # Internal integration timestep to use
    cfg.verbose = True							                # Show detailed messages
    cfg.recordTraces = {'V_soma': {'sec': 'soma', 'loc': 0.5, 'var': 'v'}}
    cfg.recordStep = 0.1
    # cfg.recordStim = True
    cfg.filename = os.path.join(sim_dir, f'{sim_label}-DCN') 	# Set file output name
    cfg.savePickle = False
    # cfg.analysis['plotTraces'] = {'include': ['all'], 'saveFig': False, 'showFig': False}  # Plot recorded traces for this list of cells
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
    IzhCell['secs']['soma']['threshold'] = -20
    netParams.cellParams['IzhCell'] = IzhCell                                   # add dict to list of cell parameters                                  # add dict to list of cell parameters

    pop_labels_nums = {'P': params.num_cells,
                       'I2': params.num_cells,
                       'W': params.num_cells,
                       'AN': params.num_cells}

    for pop_label, pop_num in pop_labels_nums.items():
        netParams.popParams[f'{pop_label}_pop'] = {'cellType': 'IzhCell',
                                                   'numCells': pop_num}


    ### Synapses ###
    netParams.synMechParams['exc'] = {'mod': 'ExpSyn', 'tau': 3, 'e': -10}
    netParams.synMechParams['inh'] = {'mod': 'ExpSyn', 'tau': 10, 'e': -70}

    ### Connections ###

    ### Excitatory ###
    netParams.connParams['AN->W'] = {
        'preConds': {'pop': 'AN_pop'},
        'postConds': {'pop': 'W_pop'},
        'synsPerConn': 1,
        'synMech': 'exc',
        'weight': 0.06,
        'connList': None
    }

    netParams.connParams['AN->I2'] = {
        'preConds': {'pop': 'AN_pop'},
        'postConds': {'pop': 'I2_pop'},
        'synsPerConn': 1,
        'synMech': 'exc',
        'weight': 0.55,
        'connList': None
    }

    netParams.connParams['AN->P'] = {
        'preConds': {'pop': 'AN_pop'},
        'postConds': {'pop': 'P_pop'},
        'synsPerConn': 1,
        'synMech': 'exc',
        'weight': 0.25,
        'connList': None
    }

    ### Inhibitory ###
    netParams.connParams['W->I2'] = {
        'preConds': {'pop': 'W_pop'},
        'postConds': {'pop': 'I2_pop'},
        'synsPerConn': 1,
        'synMech': 'inh',
        'weight': 1.4,
        'connList': None
    }

    netParams.connParams['W->P'] = {
        'preConds': {'pop': 'W_pop'},
        'postConds': {'pop': 'P_pop'},
        'synsPerConn': 1,
        'synMech': 'inh',
        'weight': 0.1,
        'connList': None
    }

    netParams.connParams['I2->P'] = {
        'preConds': {'pop': 'I2_pop'},
        'postConds': {'pop': 'P_pop'},
        'synsPerConn': 1,
        'synMech': 'inh',
        'weight': 2.25,
        'connList': None
    }


    ### Input ###
    # TODO: tonic simulation
    netParams.stimSourceParams['bkg'] = {'type': 'NetStim', 'rate': params.bkg_rate, 'noise': 1}
    netParams.stimTargetParams['bkg->ALL'] = {'source': 'bkg', 'conds': {'cellType': ['IzhCell']}, 'weight': params.bkg_weight, 'delay': 0, 'synMech': 'exc'}

    # TODO: preferred stimulus
    netParams.stimSourceParams['IClamp0'] = {'type': 'IClamp', 'del': 0, 'dur': params.sim_dur, 'amp': 0.1625}
    netParams.stimTargetParams['IClamp->allSGN'] = {'source': 'IClamp0', 'sec': 'soma', 'loc': 0.5, 'conds': {'pop': 'SGN_pop'}}

    center_in = (params.num_cells // 2) - 20
    netParams.stimSourceParams['IClamp1'] = {'type': 'IClamp', 'del': 0, 'dur': params.sim_dur, 'amp': params.in_amp}
    netParams.stimTargetParams['IClamp->SGNmid'] = {'source': 'IClamp1', 'sec': 'soma', 'loc': 0.5, 'conds': {'pop': 'SGN_pop', 'cellList': [center_in]}}

    netParams.stimSourceParams['IClamp2'] = {'type': 'IClamp', 'del': 0, 'dur': params.sim_dur, 'amp': params.in_amp/2}  # -0.3}
    netParams.stimTargetParams['IClamp->SGNside'] = {'source': 'IClamp2', 'sec': 'soma', 'loc': 0.5, 'conds': {'pop': 'SGN_pop', 'cellList': [center_in-1, center_in+1]}}

    # num_sgn_high = 40
    # sgn_high_ids = [i for i in range(params.num_cells-num_sgn_high-1,params.num_cells)]
    # if params.enable_loss:
    #     netParams.stimSourceParams['IClamp3'] = {'type': 'IClamp', 'del': 0, 'dur': params.sim_dur, 'amp': -(params.in_amp/2)}  # -0.3)}
    #     netParams.stimTargetParams['IClamp->SGNhigh'] = {'source': 'IClamp3', 'sec': 'soma', 'loc': 0.5, 'conds': {'pop': 'SGN_pop', 'cellList': sgn_high_ids}}

    ### Run simulation ###
    (pops, cells, conns, stims, simData) = sim.createSimulateAnalyze(netParams=netParams, simConfig=cfg, output=True)

    times = np.array(simData['spkt'])
    spikes = np.array(simData['spkid'])

    colors = {'SGN_pop': 'tab:red', 'Int_pop': 'tab:green', 'Fusi_pop': 'tab:purple','IC_pop': 'tab:orange'}

    ### Plot spike frequencies ###
    pop_msfs = {}
    for pop_label, pop in pops.items():
        pop_msf = mh.plot_spike_frequency(times, spikes, pop, pop_label, sim_dir, sim_label, colors)
        pop_msfs[pop_label] = float(pop_msf)

    with open(os.path.join(sim_dir, f'{sim_label}-pop_msfs.yml'), 'w') as outfile:
        yaml.dump(pop_msfs, outfile)

    ### Plot spike times ###
    mh.plot_spike_times(params.num_cells, times, spikes, pops, sim_dir, sim_label, colors)

    return pop_msfs

