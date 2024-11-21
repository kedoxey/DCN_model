import os
import yaml
import shutil
import matplotlib.pyplot as plt
import numpy as np

def get_output_dir(sim_name, sim_label):

    cwd = os.getcwd()

    output_dir = os.path.join(cwd,'output')
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    sim_top_dir = os.path.join(output_dir, sim_name)
    if not os.path.exists(sim_top_dir):
        os.mkdir(sim_top_dir)

    sim_dir = os.path.join(sim_top_dir, sim_label)
    if not os.path.exists(sim_dir):
        os.mkdir(sim_dir)

    return output_dir, sim_dir

### For config file ###
def join(loader,node):
    seq = loader.construct_sequence(node)
    return ''.join([str(i) for i in seq])

def load_config(config_name='default_config'):
    cwd = os.getcwd()
    config_dir = os.path.join(cwd, 'config')
    config_file = os.path.join(config_dir, config_name+'.yml')

    yaml.add_constructor('!join', join)
    with open(config_file) as f:
        config_params = yaml.full_load(f)

    # config_params['input_amp'] = config_params['input_amp']
    
    return config_params

def write_config(config, sim_dir, sim_label, config_name='default_config'):
    cwd = os.getcwd()

    with open(os.path.join(sim_dir,'config-'+sim_label+'.yml'), 'w') as outfile:
        yaml.dump(config, outfile)

def save_config(sim_dir, sim_label, config_name='default_config'):
    cwd = os.getcwd()
    config_dir = os.path.join(cwd, 'config')
    config_file = os.path.join(config_dir, config_name+'.yml')

    shutil.copy2(config_file, os.path.join(sim_dir,'config-'+sim_label+'.yml'))
    
    print('Config saved!')

def define_int_conns(num_cells):
    int_conn_ids = {}
    int_ids = [i for i in range(num_cells)]
    weight_pattern = [1, 2, 3, 3, 3, 2, 1]
    int_weight_patterns = {}
    for int_id in int_ids:
        l_lb, l_ub = (int_id-7, int_id-1) if int_id>0 else (0.5, 0.5)
        if l_lb < 0: l_lb = 0
        r_lb, r_ub = int_id+1, int_id+7

        left_ids = int_ids[l_lb:l_ub+1] if l_lb != 0.5 else []
        right_ids = int_ids[r_lb:r_ub+1]

        int_conn_ids[int_id] = {'left': left_ids,
                                'right': right_ids}

        left_pattern = [weight_pattern[-i-1] for i, _ in enumerate(left_ids)] if left_ids else []
        right_pattern = [weight_pattern[i] for i, _ in enumerate(right_ids)]

        int_weight_patterns[int_id] = {'left': left_pattern,
                                    'right': right_pattern}

    i2f_connList = {}

    for int_conn_id, lr_ids in int_conn_ids.items():
        temp = 5
        for side, side_ids in lr_ids.items():
            for side_i, side_id in enumerate(side_ids):
                side_weight = int_weight_patterns[int_conn_id][side][side_i]
                if side_weight in i2f_connList.keys():
                    i2f_connList[side_weight].append([int_conn_id, side_id])
                else:
                    i2f_connList[side_weight] = [[int_conn_id, side_id]]

    return i2f_connList

def plot_spike_frequency(times, spikes, pop, pop_label, sim_dir, sim_label, colors):

    fig, axs = plt.subplots(1, 1, figsize=(30,8))

    pop_msf = []

    for gid in pop.cellGids:
        spike_times = times[np.where(spikes == gid)]
        num_spikes = len(spike_times)
        num_isi = num_spikes - 1 if num_spikes > 0 else 0

        msf = num_isi / times[-1] * 1000
        axs.plot(gid, msf, 'o', color=colors[pop_label])

        pop_msf.append(msf)

    pop_msf = np.average(pop_msf)

    axs.set_title(f'{pop_label} spike frequency')
    axs.set_ylabel('Freuency (Hz)')
    axs.set_ylim([-3,200])
    axs.set_xlim([pop.cellGids[0]-2, pop.cellGids[-1]+2])
    axs.set_xticks(pop.cellGids)
    axs.set_xticklabels(range(len(pop.cellGids)), rotation=90)
    axs.set_xlabel('Cells (id)')
    fig.tight_layout()
    fig.savefig(os.path.join(sim_dir,f'{sim_label}-{pop_label}-spike_frequency.png'), dpi=300)

    return pop_msf

def plot_spike_times(num_cells, times, spikes, pops, sim_dir, sim_label, colors):
    fig, axs = plt.subplots(1, 1, figsize=(8,12))

    tot_cells = 0
    for pop_label, pop in pops.items():
        for gid in pop.cellGids:
            spike_times = times[np.where(spikes == gid)]

            if gid%num_cells == 0:
                add_label = True
            else:
                add_label = False

            # loc = -1 if gid == 5 else gid
            if add_label:
                axs.vlines(spike_times, gid-0.25, gid+0.25, color=colors[pop_label], label=pop_label)
            else:
                axs.vlines(spike_times, gid-0.25, gid+0.25, color=colors[pop_label])  #, label=pop_label)

            tot_cells += 1
            # print(f'{pop_label}: {spike_times.shape[0]} spikes')

    # axs.set_yticks(range(39))
    # axs.set_yticklabels([0,0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18])

    axs.legend(loc='upper left', bbox_to_anchor=(1, 1))
    yticks = [(100*i)-1 for i in range(7)]
    yticks[0] = 0
    axs.set_yticks(yticks)
    axs.set_ylim([-2, tot_cells+2])
    # axs.set_yticks([i for i in range(tot_cells)])
    # axs.set_yticklabels([i for _ in range(len(pops)) for i in range(num_cells)])
    axs.set_xlim(0,1000)
    axs.set_ylabel('Cells (gid)')
    axs.set_xlabel('Time (ms)')

    fig.tight_layout()

    fig.savefig(os.path.join(sim_dir,f'{sim_label}-spike_times.png'), dpi=300)
