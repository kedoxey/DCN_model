import os
import yaml
import shutil
import random
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import pandas as pd

def get_output_dir(sim_name, sim_label):

    cwd = os.getcwd()

    output_dir = os.path.join(cwd,'output')
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    sim_top_dir = os.path.join(output_dir, sim_name)
    if not os.path.exists(sim_top_dir):
        os.mkdir(sim_top_dir)

    data_dir = os.path.join(sim_top_dir, 'data')
    if not os.path.exists(data_dir):
        os.mkdir(data_dir)

    sim_dir = os.path.join(sim_top_dir, sim_label)
    if not os.path.exists(sim_dir):
        os.mkdir(sim_dir)

    return output_dir, data_dir, sim_dir

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

def sigmoid(x, A, K, C, B, M):
    return A+ K/(C+np.exp(-B*(x-M)))

def define_input_freqs(num_ins):
    A = 1
    K = 1
    C = 0.01
    B = 0.02  #0.02
    M = 100  #-4.4

    xs = np.linspace(10, 600, num_ins)
    ys = np.array([sigmoid(x, A, K, C, B, M) for x in xs])

    y0_lin = ys[0]
    yn_lin = ys[-1]
    y0_log = 1
    yn_log = 100

    b = (yn_log - y0_log) / np.log(yn_lin / y0_lin)
    a = y0_log - b * np.log(y0_lin)

    ys_log = a + b * np.log(ys)

    ys_log = np.append(0, ys_log)

    return np.append(0, ys)

def define_freqs(num_cells):

    freqs = [0 for i in range(num_cells)]

    initial = 1250
    final = 20000

    oct_step = 0.005
    step_factor = 2**oct_step

    for i in range(num_cells):
        if i == 0:
            freqs[i] = initial
            continue

        freqs[i] = freqs[i-1]*step_factor
    
    return freqs

def define_conns(num_cells, freqs, conn_params):

    df = pd.DataFrame.from_dict(conn_params)

    cell_ids = {'AN': [i for i in range(num_cells)],
                'W': [i for i in range(num_cells)],
                'I2': [i for i in range(num_cells)],
                'P': [i for i in range(num_cells)]}

    conns = {'AN_W': [],
            'AN_I2': [],
            'AN_P': [],
            'W_I2': [],
            'W_P': [],
            'I2_P': []}
    
    freqs_arr = np.array(freqs)
    freqs_round = np.round(freqs_arr,0)

    for conn, row in df.iterrows():
        temp = 5

        source = conn.split('_')[0]
        source_ids = cell_ids[source]

        target = conn.split('_')[1]
        target_ids = cell_ids[target]

        conn_list = []

        for target_id in target_ids:

            target_freq = freqs_round[target_id]

            bw = row.BW
            bw_split = bw / 2
            num_source = int(row.N)

            lb_freq = int(target_freq * 2**-bw_split)
            if lb_freq < 1250: lb_freq = 1250
            ub_freq = int(target_freq * 2**bw_split)

            lb = (np.abs(freqs_round - lb_freq)).argmin()
            ub = (np.abs(freqs_round - ub_freq)).argmin()

            source_pool = source_ids[lb:ub]

            if target_id == 0:
                bw_whole = len(source_pool)*2
            if num_source > len(source_pool):
                num_source *= (len(source_pool)/bw_whole)

            try:
                source_rand = random.sample(source_pool,int(num_source))
            except ValueError:
                print(conn,target_id)

            conn_list.extend([[i,target_id] for i in source_rand])

        conns[conn] = conn_list

    return conns


def get_firing_rates(times, spikes, pop, stim_dur, stim_delay, sim_dur):

    window_dur = 160
    stim_end = stim_delay + stim_dur

    driven_rates = []
    driven_times = times[np.where((times <= stim_end) & (times >= stim_end-window_dur))]

    spont_rates = []
    spont_times = times[np.where((times <= sim_dur) & (times >= sim_dur-window_dur))]

    for gid in pop.cellGids:
        spike_times = times[np.where(spikes == gid)]

        try:
            driven_spikes = spike_times[np.where((driven_times[0] <= spike_times) & (spike_times <= driven_times[-1]))]
            num_driven = len(driven_spikes)
            driven_isi = num_driven -1 if num_driven > 0 else 0
            driven_msf = driven_isi / (driven_spikes[-1] - driven_spikes[0]) * 1000 if num_driven > 0 else 0
        except:
            driven_msf = 0
        driven_rates.append(driven_msf)

        try:
            spont_spikes = spike_times[np.where((spont_times[0] <= spike_times) & (spike_times <= spont_times[-1]))]
            num_spont = len(spont_spikes)
            spont_isi = num_spont -1 if num_spont > 0 else 0
            spont_msf = spont_isi / (spont_spikes[-1] - spont_spikes[0]) * 1000 if num_spont > 0 else 0
        except:
            spont_msf = 0
        spont_rates.append(spont_msf)

    driven_avg = np.average(driven_rates)
    spont_avg = np.average(spont_rates)

    return driven_rates, spont_rates

def plot_firing_rates(driven_rates, spont_rates, pop, pop_label, sim_dir, sim_label, colors):

    # plot together
    fig, axs = plt.subplots(1, 1, figsize=(30,8))

    color_rgb = list(mpl.colors.to_rgb(colors[pop_label]))
    driven_color = [color_rgb[0]/2, color_rgb[1], color_rgb[2]]
    spont_color = [color_rgb[0]/0.9, color_rgb[1], color_rgb[2]]

    axs.plot(pop.cellGids, driven_rates, 'o', color=driven_color, label='driven')
    axs.plot(pop.cellGids, spont_rates, 'o', color=spont_color, label='spontaneous')

    axs.set_title(f'{pop_label} spike frequency')
    axs.set_ylabel('Frequency (Hz)')
    axs.set_xlim([pop.cellGids[0]-2, pop.cellGids[-1]+2])
    axs.set_xticks(pop.cellGids)
    axs.set_xticklabels(range(len(pop.cellGids)), rotation=90)
    axs.set_xlabel('Cells (id)')
    axs.legend(loc='upper right')
    fig.tight_layout()
    fig.savefig(os.path.join(sim_dir,f'{sim_label}-{pop_label}-firing_rates_comb.png'), dpi=300)

    # plot separate
    fig, axs = plt.subplots(1, 2, figsize=(30,8))
    axs = axs.ravel()

    axs[0].plot(pop.cellGids, driven_rates, 'o', color=driven_color)
    axs[0].set_title('Driven')
    axs[1].plot(pop.cellGids, spont_rates, 'o', color=spont_color)
    axs[1].set_title('Spontaneous')

    for ax in axs:
        ax.set_xlim([pop.cellGids[0]-2, pop.cellGids[-1]+2])
        ax.set_xticks([i*100 for i in range((pop.cellGids[-1]//100)+2)])
        ax.set_xticklabels([i*100 for i in range((pop.cellGids[-1]//100)+2)], rotation=90)
        ax.set_xlabel('Cells (id)')

    axs[0].set_ylabel('Frequency (Hz)')
    
    fig.suptitle(f'{pop_label} spike frequency')
    fig.tight_layout()
    fig.savefig(os.path.join(sim_dir,f'{sim_label}-{pop_label}-firing_rates_sep.png'), dpi=300)


def plot_spike_frequency(times, spikes, pop, pop_label, sim_dir, sim_label, colors):

    fig, axs = plt.subplots(1, 1, figsize=(30,8))

    pop_msf = []

    for gid in pop.cellGids:
        spike_times = times[np.where(spikes == gid)]
        num_spikes = len(spike_times)
        num_isi = num_spikes - 1 if num_spikes > 0 else 0

        msf = num_isi / (spike_times[-1] - spike_times[0]) * 1000 if num_spikes > 0 else 0
        axs.plot(gid, msf, 'o', color=colors[pop_label])

        pop_msf.append(msf)

    pop_msf = np.nanmean(pop_msf)

    axs.set_title(f'{pop_label} spike frequency')
    axs.set_ylabel('Frequency (Hz)')
    # axs.set_ylim([-3,200])
    axs.set_xlim([pop.cellGids[0]-2, pop.cellGids[-1]+2])
    axs.set_xticks([i*100 for i in range((pop.cellGids[-1]//100)+2)])
    # axs.set_xticks(pop.cellGids)
    axs.set_xticklabels([i*100 for i in range((pop.cellGids[-1]//100)+2)], rotation=90)
    axs.set_xlabel('Cells (id)')
    fig.tight_layout()
    fig.savefig(os.path.join(sim_dir,f'{sim_label}-{pop_label}-spike_frequency.png'), dpi=300)

    return pop_msf

def plot_spike_times(num_cells, times, spikes, pops, stim_dur, stim_delay, sim_dir, sim_label, colors):


    fig, axs = plt.subplots(1, 1, figsize=(8,12))
    fig2, axs2 = plt.subplots(1,1, figsize=(8,8))

    tot_cells = 0
    for pop_label, pop in pops.items():
        # if 'vecstim' in pop_label:
        #     continue
        for gid in pop.cellGids:
            spike_times = times[np.where(spikes == gid)]

            if gid%num_cells == 0:
                add_label = True
            else:
                add_label = False

            # loc = -1 if gid == 5 else gid
            if add_label:
                axs.vlines(spike_times, gid-0.25, gid+0.25, color=colors[pop_label], label=pop_label, zorder=12)
                if 'P_pop' in pop_label:
                    axs2.vlines(spike_times, gid-0.25, gid+0.25, color=colors[pop_label], label=pop_label, zorder=12)
                if 'vecstim' in pop_label:
                    axs2.vlines(spike_times, 801-0.25, 801+3, color=colors[pop_label], label=pop_label, zorder=12)
            else:
                axs.vlines(spike_times, gid-0.25, gid+0.25, color=colors[pop_label], zorder=12)  #, label=pop_label)
                if 'P_pop' in pop_label:
                    axs2.vlines(spike_times, gid-0.25, gid+0.25, color=colors[pop_label], zorder=12)

            tot_cells += 1
            # print(f'{pop_label}: {spike_times.shape[0]} spikes')

    # axs.set_yticks(range(39))
    # axs.set_yticklabels([0,0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18])

    axs.axvspan(stim_delay, stim_delay+stim_dur, color='k', alpha=0.08, zorder=1)
    axs.legend(loc='upper left', bbox_to_anchor=(1, 1))
    # yticks = [(100*i)-1 for i in range(7)]
    # yticks[0] = 0
    # axs.set_yticks(yticks)
    axs.set_ylim([-2, tot_cells+2])
    # axs.set_yticks([i for i in range(tot_cells)])
    # axs.set_yticklabels([i for _ in range(len(pops)) for i in range(num_cells)])
    axs.set_xlim(-10,1010)
    axs.set_ylabel('Cells (gid)')
    axs.set_xlabel('Time (ms)')

    fig.tight_layout()
    fig.savefig(os.path.join(sim_dir,f'{sim_label}-spike_times.png'), dpi=300)


    axs2.axvspan(stim_delay, stim_delay+stim_dur, color='k', alpha=0.08, zorder=1)
    axs2.legend(loc='upper left', bbox_to_anchor=(1, 1))
    axs2.set_ylim([-2, 810])
    axs2.set_xlim(-10,1010)
    axs2.set_ylabel('Cells (gid)')
    axs2.set_xlabel('Time (ms)')
    fig2.tight_layout()
    fig2.savefig(os.path.join(sim_dir,f'{sim_label}-spike_times-P_V.png'), dpi=300)


def plot_traces(simData, pops, pop_cells, stim_delay, stim_dur, sim_dir, sim_label, colors):

    fig, axs = plt.subplots(len(pop_cells), len(pop_cells[0][1]), figsize=(15,8))

    t = simData['t']

    for i, (pop_label, cells) in enumerate(pop_cells):

        cell_type = pop_label.split('_')[0]

        for j, cell_id in enumerate(cells):
            
            gid = pops[pop_label].cellGids[cell_id]

            v_trace = simData['V_soma'][f'cell_{gid}']

            axs[i,j].plot(t,v_trace,color=colors[pop_label],zorder=12)
            axs[i,j].set_title(f'{cell_type}_{cell_id}')
            axs[i,j].set_xlabel('Time (ms)')
            axs[i,j].axvspan(stim_delay, stim_delay+stim_dur, color='k', alpha=0.1, zorder=1)

            temp = 5

    axs[0,0].set_ylabel('Voltage (mV)')
    axs[1,0].set_ylabel('Voltage (mV)')

    fig.tight_layout()
    fig.savefig(os.path.join(sim_dir, f'{sim_label}-cell_traces.png'), dpi=300)


def plot_cells(simData, cell_types, n_cells, pops, conns):

    t = np.array(simData['t'])
    spkid = np.array(simData['spkid'])
    spkt = np.array(simData['spkt'])

    fig, axs = plt.subplots(len(cell_types), n_cells, figsize=(9*n_cells,5*len(cell_types)))
    
    for pop_i, (pop_label, pop_cells) in enumerate(pops.items()):

        cell_type = pop_label.split('_')[0]

        if 'vecstim' in cell_type:
            continue
        
        for cell_i, cell_id in enumerate(pop_cells.cellGids):

            v_soma = list(simData['V_soma'][f'cell_{cell_id}'])
    
            if n_cells == 1:
                ax = axs[pop_i]
            else:
                ax = axs[pop_i, cell_i]
            ax.plot(t, v_soma, color='dimgrey', linewidth=1)

            for conn in conns[cell_id]:
                pregid = conn['preGid']

                conn_label = conn['label'].split('-')[0].split('_')[1] if '_' in conn['label'] else conn['label']


                add_train = False
                if 'NSA' in conn_label:
                    color = 'forestgreen'
                    middle = 15
                    add_train = True
                elif 'ANF' in conn_label:
                    color = 'firebrick'
                    middle = -10*(int(conn_label.split('F')[1]))
                    add_train = True
                else:
                    add_train = False
                # elif 'inh' in conn['label']:
                #     color= 'tab:blue'
                #     middle = -15

                if add_train:
                    spike_train = spkt[np.where(spkid == pregid)]

                    if len(spike_train) > 0:
                        ax.vlines(spike_train, ymin=middle-5, ymax=middle+5, color=color, label=conn_label)


                spikes = spkt[np.where(spkid == cell_id)]
                n_spikes = len(spikes)
                msf = (n_spikes - 1) / (spikes[-1] - spikes[0]) * 1000 if n_spikes > 0 else 0

                ax.set_title(f'{pop_label} {cell_id} - {msf} spikes')
                ax.legend()

    return fig