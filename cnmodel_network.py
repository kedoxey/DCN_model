from cnmodel import populations
from cnmodel.util import sound, random_seed
from cnmodel.protocols import Protocol
from collections import OrderedDict
import os, sys, time
import pickle
import random
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from neuron import h
import multiprocessing as mp
import yaml
from scipy.signal import savgol_filter
import itertools

start_t = time.time()

class CharacterizeNetwork(Protocol):
    def __init__(self, seed, frequencies, cells_per_band=1, loss_limit=99e3, loss_frac=0, temp=34.0, dt=0.025, 
                 hearing='normal', synapsetype="simple", enable_ic=False, cohc=1, cihc=1, syn_opts=None):
        Protocol.__init__(self)

        self.base_data = None

        self.seed = seed
        self.temp = temp
        self.dt = dt
        self.syn_opts = syn_opts
        self.hearing = hearing  # normal or loss
        self.enable_ic = enable_ic
        self.loss_limit = loss_limit
        self.loss_frac = loss_frac

        # Seed now to ensure network generation is stable
        random_seed.set_seed(seed)
        # Create cell populations.
        # This creates a complete set of _virtual_ cells for each population. No
        # cells are instantiated at this point.
        self.sgc = populations.SGC(model="dummy", hearing=self.hearing, loss_limit=self.loss_limit, loss_frac=loss_frac, cohc=cohc, cihc=cihc)
        self.dstellate = populations.DStellate(synapsetype='simple')
        # self.tstellate = populations.TStellate()
        # if self.enable_ic:
        #     self.pyramidal = populations.Pyramidal(synapsetype='simple')
        #     self.tuberculoventral = populations.Tuberculoventral(synapsetype='simple')
        #     self.ic = populations.IC()
        # else:
        self.pyramidal = populations.Pyramidal(synapsetype='simple', hearing=self.hearing, loss_limit=self.loss_limit, syn_opts=syn_opts)
        self.tuberculoventral = populations.Tuberculoventral(synapsetype='simple')

        pops = [
            self.sgc,
            self.dstellate,
            self.tuberculoventral,
            # self.tstellate,
            self.pyramidal
        ]
        if self.enable_ic:
            pops.append(self.ic)
        self.populations = OrderedDict([(pop.type, pop) for pop in pops])

        # set synapse type to use in the sgc population - simple is fast, multisite is slower
        # (eventually, we could do this for all synapse types..)
        self.sgc._synapsetype = synapsetype

        # Connect populations.
        # This only defines the connections between populations; no synapses are
        # created at this stage.
        self.sgc.connect(
            self.pyramidal, self.dstellate, self.tuberculoventral  # , self.tstellate
        )
        self.dstellate.connect(self.pyramidal, self.tuberculoventral)
        self.tuberculoventral.connect(self.pyramidal)  #, self.tstellate) 
        # self.tstellate.connect(self.pyramidal)

        if self.enable_ic:
            self.pyramidal.connect(self.ic)
            self.ic.connect(self.pyramidal, self.tuberculoventral)

        # Select cells to record from.
        # At this time, we actually instantiate the selected cells.
        # frequencies = [16e3]
        # cells_per_band = 1
        self.cells_per_band = cells_per_band
        self.pyramidal_ids_per_band = {}
        for f in frequencies:
            pyramidal_cell_ids = self.pyramidal.select(self.cells_per_band, cf=f, create=True)
            self.pyramidal_ids_per_band[f] = pyramidal_cell_ids

        # Now create the supporting circuitry needed to drive the cells we selected.
        # At this time, cells are created in all populations and automatically
        # connected with synapses.
        self.pyramidal.resolve_inputs(depth=2)
        # Note that using depth=2 indicates the level of recursion to use when
        # resolving inputs. For example, resolving inputs for the bushy cell population
        # (level 1) creates presynaptic cells in the dstellate population, and resolving
        # inputs for the dstellate population (level 2) creates presynaptic cells in the
        # sgc population.

    def run(self, stim, seed, result):
        """Run the network simulation with *stim* as the sound source and a unique
        *seed* used to configure the random number generators.
        """

        self.reset()

        # Generate 2 new seeds for the SGC spike generator and for the NEURON simulation
        rs = np.random.RandomState()
        rs.seed(self.seed+seed)
        seed1, seed2 = rs.randint(0, (2 ** 32) - 5, 2)
        random_seed.set_seed(seed1)
        self.sgc.set_seed(seed2)

        self.sgc.set_sound_stim(stim, parallel=False)

        # set up recording vectors
        for pop in self.pyramidal, self.dstellate, self.tuberculoventral:  # self.bushy, self.dstellate, self.tstellate, self.tuberculoventral:
            for ind in pop.real_cells():
                cell = pop.get_cell(ind)
                self[cell] = cell.soma(0.5)._ref_v
        self["t"] = h._ref_t

        h.tstop = stim.duration * 1000
        h.celsius = self.temp
        h.dt = self.dt

        self.custom_init()
        last_update = time.time()
        while h.t < h.tstop:
            h.fadvance()
            now = time.time()
            if now - last_update > 1.0:
                print(("%0.2f / %0.2f" % (h.t, h.tstop)))
                last_update = now

        # record vsoma and spike times for all cells
        # vec = {}
        for k in self._vectors:
            v = self[k].copy()
            if k == "t":
                result[k] = v
                continue
            spike_inds = np.argwhere((v[1:] > -20) & (v[:-1] <= -20))[:, 0]
            spikes = self["t"][spike_inds]
            pop = k.celltype
            # print('pop: ', pop)
            assert isinstance(pop, str)
            cell_ind = getattr(self, pop).get_cell_index(k)
            result[(pop, cell_ind)] = [v, spikes]

        # record SGC spike trains
        for ind in self.sgc.real_cells():
            cell = self.sgc.get_cell(ind)
            result[("sgc", ind)] = [None, cell._spiketrain]

        # result = vec
        temp = 5

        # return vec

    def plot_results(self, nreps, results, baseline, response, output_dir):

        results_od = OrderedDict()
        max_iter = 0
        stim_order = []
        freqs = set()
        levels = set()

        for k, v in list(results.items()):

            f0, dbspl, iteration = k
            max_iter = max(max_iter, iteration)
            stim, result = v
            key = "f0: %0.0f  dBspl: %0.0f" % (f0, dbspl)
            results_od.setdefault(key, [stim,{}])
            results_od[key][1][iteration] = result
            stim_order.append((f0, dbspl))
            freqs.add(f0)
            levels.add(dbspl)
        
        # results = results_od

        freqs = sorted(list(freqs))
        levels = sorted(list(levels))

        spont_spikes = 0
        spont_time = 0

        pop_type = 'pyramidal'
        cell_ind = self.pyramidal.real_cells()[0]

        for stim, iterations in list(results_od.values()):
            for vec in list(iterations.values()):
                print(f"plotting - f = {round(stim.opts['f0'])}, dbspl = {stim.opts['dbspl']}")
                spikes = vec[(pop_type, cell_ind)][1]
                spont_spikes += ((spikes >= baseline[0]) * (spikes < baseline[1])).sum()
                spont_time += baseline[1] - baseline[0]

        spont_rate = spont_spikes / spont_time

        matrix = np.zeros((len(levels), len(freqs)))

        for stim, iteration in list(results_od.values()):

            for i in range(nreps):

                vec = iteration[i]
                spikes = vec[(pop_type, cell_ind)][1]
                n_spikes = ((spikes >= response[0]) & (spikes < response[1])).sum()

                freq = stim.key()['f0']
                level = stim.key()['dbspl']

                j = freqs.index(freq)
                i = levels.index(level)

                # response_map_df.loc[level, [freq]] += n_spikes - spont_rate * (response[1] - response[0])
                matrix[i,j] += n_spikes - spont_rate * (response[1] - response[0])

        matrix /= nreps

        b = (freqs[-1] - freqs[0]) / np.log(freqs[-1] / freqs[0])
        a = freqs[0] - b * np.log(freqs[0])
        freqs_log = a + b * np.log(freqs)

        fig, axs = plt.subplots(1,1,figsize=(5,4))

        im = axs.pcolormesh(freqs_log, levels, matrix, cmap=cm.viridis)  # TODO: change colormap
        # axs.invert_yaxis()
        axs.set_xlabel('Frequency (kHz)')
        axs.set_ylabel('Sound Level (dBSPL)')
        # axs.set_xscale('log')
        axs.set_xticks([freqs_log[i] for i in [0, 6, 12, 18, 24]])
        axs.set_xticklabels([round(np.ceil(freqs[i]/1000)) for i in [0, 6, 12, 18, 24]])

        title = 'Hearing Loss' if 'loss' in self.hearing else 'Normal Hearing'
        if self.enable_ic:
            title += ' with IC'
        axs.set_title(title)

        fig.colorbar(im)
        fig.tight_layout()
        filename = f'{len(freqs)}fs_{len(levels)}dbs-response_map'
        if 'loss' in self.hearing:
            filename += '-loss'
        if self.enable_ic:
            filename += '-ic'
        fig.savefig(os.path.join(output_dir, f'{filename}.png'), dpi=300)

        rm_data = {'freqs': freqs,
                   'freqs_log': freqs_log,
                   'levels': levels,
                   'matrix': matrix}
        pickle.dump(rm_data, open(os.path.join(output_dir, f'DATA-{filename}.pkl'), 'wb'))


    def save_pyramidal_firing_rates(self, results, response, baseline, output_dir):

        results_od = OrderedDict()
        max_iter = 0
        stim_order = []
        freqs = set()
        levels = set()

        for k, v in list(results.items()):

            f0, dbspl, iteration = k
            max_iter = max(max_iter, iteration)
            stim, result = v
            key = "f0: %0.0f  dBspl: %0.0f" % (f0, dbspl)
            results_od.setdefault(key, [stim,{}])
            results_od[key][1][iteration] = result
            stim_order.append((f0, dbspl))
            freqs.add(f0)
            levels.add(dbspl)

        freqs = sorted(list(freqs))
        levels = sorted(list(levels))

        avg_evoked_msfs = {}
        avg_spont_msfs = {}
        for stim, iterations in list(results_od.values()):

            stim_avg_evoked_msfs  = []
            stim_avg_spont_msfs  = []

            for vec in list(iterations.values()):
                temp = 6
                rep_avg_evoked_msfs = []
                rep_avg_spont_msfs = []

                for cf, ids in list(self.pyramidal_ids_per_band.items()):

                    evoked_msfs = []
                    spont_msfs = []
                    for pyr_id in ids:
                        spikes = vec[('pyramidal', pyr_id)][1]
                        # resp_spkt = spkt[((spkt >= response[0]) & (spkt <= response[1]))]

                        # num_spikes = len(resp_spkt)
                        # num_isi = num_spikes - 1 if num_spikes > 0 else 0
                        # msf = num_isi / (resp_spkt[-1] - resp_spkt[0]) * 1000 if num_spikes > 1 else 1

                        n_spont_spikes = ((spikes >= baseline[0]) & (spikes <= baseline[1])).sum()
                        spont_msf = (n_spont_spikes / (baseline[1] - baseline[0])) * 1000
                        spont_msfs.append(spont_msf)

                        n_evoked_spikes = ((spikes >= response[0]) & (spikes <= response[1])).sum()
                        evoked_msf = (n_evoked_spikes / (response[1] - response[0])) * 1000
                        evoked_msfs.append(evoked_msf)

                    rep_avg_evoked_msfs.append(np.mean(evoked_msfs))
                    rep_avg_spont_msfs.append(np.mean(spont_msfs))

                stim_avg_evoked_msfs.append(rep_avg_evoked_msfs)
                stim_avg_spont_msfs.append(rep_avg_spont_msfs)
            
            avg_evoked_msfs[stim] = np.average(np.array(stim_avg_evoked_msfs), axis=0)
            avg_spont_msfs[stim] = np.average(np.array(stim_avg_spont_msfs), axis=0)

        pickle.dump(avg_evoked_msfs, open(os.path.join(output_dir, 'stim_avg_evoked_rates.pkl'), 'wb'))
        pickle.dump(avg_evoked_msfs, open(os.path.join(output_dir, 'stim_avg_spont_rates.pkl'), 'wb'))

        metadata = {
            'populations': {'sgc': len(self.sgc.real_cells()),
                            'dstellate': len(self.dstellate.real_cells()),
                            'vertical': len(self.tuberculoventral.real_cells()),
                            'pyramidal': len(self.pyramidal.real_cells())}
        }

        return avg_spont_msfs, avg_evoked_msfs, metadata


    def plot_pyramidal_evoked_firing_rates(self, avg_msfs, freqs, output_dir):

        b = (freqs[-1] - freqs[0]) / np.log(freqs[-1] / freqs[0])
        a = freqs[0] - b * np.log(freqs[0])
        freqs_log = a + b * np.log(freqs)

        fig, axs = plt.subplots(1,1,figsize=(10,5))

        for stim, avg_rates in list(avg_msfs.items()):
            level = stim.opts['dbspl']

            # axs[i].plot(freqs_log, avg_rates, color='tab:blue')  #, 'o-')
            axs.plot(freqs, avg_rates, label=f'{level} dB')  #, color='tab:red')

        # xspan = len(freqs)//4
        # axs.set_xticks([freqs[i] for i in [0, xspan, xspan*2, xspan*3, xspan*4]])
        # axs.set_xticklabels([round(freqs[i]/1000) for i in [0,xspan, xspan*2, xspan*3, xspan*4]])
        axs.set_xlabel('Characteristic Frequency (kHz)')
        axs.set_ylabel('Firing Rate (Hz)')
        axs.legend(loc='upper right')

        title = 'Hearing Loss' if 'loss' in self.hearing else 'Normal Hearing'
        axs.set_title(title)
        fig.tight_layout()

        filename = 'pyramidal_response_curve'
        if 'loss' in self.hearing:
            filename += '-loss'
        fig.savefig(os.path.join(output_dir, f'{filename}.png'), dpi=300)


    def reformat_results(self, results_df):

        results_od = OrderedDict()
        max_iter = 0
        stim_order = []
        freqs = set()
        levels = set()

        for k, v in list(results_df.items()):

            f0, dbspl, iteration = k
            max_iter = max(max_iter, iteration)
            stim, result = v
            key = "f0: %0.0f  dBspl: %0.0f" % (f0, dbspl)
            results_od.setdefault(key, [stim,{}])
            results_od[key][1][iteration] = result
            stim_order.append((f0, dbspl))
            freqs.add(f0)
            levels.add(dbspl)

        freqs = sorted(list(freqs))
        levels = sorted(list(levels))

        return freqs, levels, results_od


    def get_evoked_firing_rates(self, cell_type, results, response, pyr_ids_per_band=None):

        avg_rates = {}
        cell_ids = {}

        for stim, iterations in list(results.values()):

            stim_avg_rates  = []
            stim_cell_ids = []

            n_reps = len(iterations)

            for i in range(n_reps):

                rep_rates = []
                rep_cell_ids = []

                for cell, vec in iterations[i].items():
                # rep_avg_msfs = []

                    if isinstance(cell, str):
                        continue

                    if cell_type in cell[0]:

                        cell_id = cell[1]

                        spikes = np.array(vec[1])
                        n_spikes = ((spikes >= response[0]) & (spikes <= response[1])).sum()
                        spike_rate = (n_spikes / (response[1] - response[0])) * 1000  # ms to s

                        rep_rates.append(spike_rate)
                        rep_cell_ids.append(cell_id)
            
                stim_avg_rates.append(rep_rates)
                stim_cell_ids.append(rep_cell_ids)

            stim_avg_rates = np.average(stim_avg_rates, axis=0)
            
            avg_rates[(stim.opts['f0'], stim.opts['dbspl'])] = stim_avg_rates
            cell_ids[(stim.opts['f0'], stim.opts['dbspl'])] = stim_cell_ids[0]

        return cell_ids, avg_rates
    

    def get_spontaneous_firing_rates(self, cell_type, results, baseline, pyr_ids_per_band=None):

        avg_rates = {}
        cell_ids = {}

        for stim, iterations in list(results.values()):

            stim_avg_rates  = []
            stim_cell_ids = []

            n_reps = len(iterations)

            for i in range(n_reps):

                rep_rates = []
                rep_cell_ids = []

                for cell, vec in iterations[i].items():
                # rep_avg_msfs = []

                    if isinstance(cell, str):
                        continue

                    if cell_type in cell[0]:

                        cell_id = cell[1]

                        spikes = np.array(vec[1])
                        n_spikes = ((spikes >= baseline[0]) & (spikes <= baseline[1])).sum()
                        spike_rate = (n_spikes / (baseline[1] - baseline[0])) * 1000  # ms to s

                        rep_rates.append(spike_rate)
                        rep_cell_ids.append(cell_id)
            
                stim_avg_rates.append(rep_rates)
                stim_cell_ids.append(rep_cell_ids)

            stim_avg_rates = np.average(stim_avg_rates, axis=0)
            
            avg_rates[(stim.opts['f0'], stim.opts['dbspl'])] = stim_avg_rates
            cell_ids[(stim.opts['f0'], stim.opts['dbspl'])] = stim_cell_ids[0]

        return cell_ids, avg_rates


    def plot_network_firing_rates(self, results, stim, baseline, response, output_dir):

        colors = {'pyramidal': 'tab:blue',
                  'vertical': 'tab:green',
                  'dstellate': 'tab:orange',
                  'sgc': 'tab:purple'}
        
        freq = stim['freq']
        level = stim['level']

        pyr_cfs = self.pyramidal._get_cf_array(species='mouse')
        vert_cfs = self.tuberculoventral._get_cf_array(species='mouse')
        ds_cfs = self.dstellate._get_cf_array(species='mouse')
        sgc_cfs = self.sgc._get_cf_array(species='mouse')

        ### Evoked Activity ###
        pyr_ids, pyr_rates = self.get_evoked_firing_rates('pyramidal', results, response)
        ds_ids, ds_rates = self.get_evoked_firing_rates('dstellate', results, response)
        vert_ids, vert_rates = self.get_evoked_firing_rates('tuberculoventral', results, response)
        sgc_ids, sgc_rates = self.get_evoked_firing_rates('sgc', results, response)

        fig, axs = plt.subplots(4,1,figsize=(15,8), sharex=True)  #, sharey=True)
        axs = axs.ravel()

        axs[0].plot(pyr_cfs[pyr_ids[(freq, level)]], pyr_rates[(freq, level)], color=colors['pyramidal'], zorder=13, alpha=0.8, label='pyramidal')
        axs[0].legend(loc='upper right')

        axs[1].plot(vert_cfs[vert_ids[(freq, level)]], vert_rates[(freq, level)], color=colors['vertical'], zorder=13, alpha=0.8, label='vertical')
        axs[1].legend(loc='upper right')

        axs[2].plot(ds_cfs[ds_ids[(freq, level)]], ds_rates[(freq, level)], color=colors['dstellate'], zorder=13, alpha=0.8, label='dstellate')
        axs[2].legend(loc='upper right')

        axs[3].plot(sgc_cfs[sgc_ids[(freq, level)]], sgc_rates[(freq, level)], color=colors['sgc'], zorder=13, alpha=0.8, label='sgc')
        axs[3].legend(loc='upper right')

        axs[-1].set_xlabel('Characteristic Frequency (Hz)')
        axs[-1].set_ylabel('Firing Rate (Hz)')

        fig.suptitle('Evoked Activity')
        fig.tight_layout()

        filename = 'population_evoked_activity'
        if 'loss' in self.hearing:
            filename += '-loss'
        fig.savefig(os.path.join(output_dir, f'{filename}.png'), dpi=300)

        ### Spontaneous Activity ###
        pyr_ids, pyr_rates = self.get_spontaneous_firing_rates('pyramidal', results, baseline)
        ds_ids, ds_rates = self.get_spontaneous_firing_rates('dstellate', results, baseline)
        vert_ids, vert_rates = self.get_spontaneous_firing_rates('tuberculoventral', results, baseline)
        sgc_ids, sgc_rates = self.get_spontaneous_firing_rates('sgc', results, baseline)

        fig, axs = plt.subplots(4,1,figsize=(15,8), sharex=True)  #, sharey=True)
        axs = axs.ravel()

        axs[0].plot(pyr_cfs[pyr_ids[(freq, level)]], pyr_rates[(freq, level)], color=colors['pyramidal'], zorder=13, alpha=0.8, label='pyramidal')
        axs[0].legend(loc='upper right')

        axs[1].plot(vert_cfs[vert_ids[(freq, level)]], vert_rates[(freq, level)], color=colors['vertical'], zorder=13, alpha=0.8, label='vertical')
        axs[1].legend(loc='upper right')

        axs[2].plot(ds_cfs[ds_ids[(freq, level)]], ds_rates[(freq, level)], color=colors['dstellate'], zorder=13, alpha=0.8, label='dstellate')
        axs[2].legend(loc='upper right')

        axs[3].plot(sgc_cfs[sgc_ids[(freq, level)]], sgc_rates[(freq, level)], color=colors['sgc'], zorder=13, alpha=0.8, label='sgc')
        axs[3].legend(loc='upper right')

        axs[-1].set_xlabel('Characteristic Frequency (Hz)')
        axs[-1].set_ylabel('Firing Rate (Hz)')

        fig.suptitle('Spontaneous Activity')
        fig.tight_layout()

        filename = 'population_spontaneous_activity'
        if 'loss' in self.hearing:
            filename += '-loss'
        fig.savefig(os.path.join(output_dir, f'{filename}.png'), dpi=300)


def plot_hair_cell_firing_rates(hc_msfs, freqs, output_dir):

    b = (freqs[-1] - freqs[0]) / np.log(freqs[-1] / freqs[0])
    a = freqs[0] - b * np.log(freqs[0])
    freqs_log = a + b * np.log(freqs)

    fig, axs = plt.subplots(1,1,figsize=(15,5))

    for cohc, cihc in list(hc_msfs.keys()):
        avg_msfs = hc_msfs[(cohc, cihc)]
        for stim, avg_rates in list(avg_msfs.items()):
            level = stim.opts['dbspl']

            # axs[i].plot(freqs_log, avg_rates, color='tab:blue')  #, 'o-')
            axs.plot(freqs, avg_rates, label=f'Cohc={cohc}, Cihc={cihc}')  #, color='tab:red')

    axs.set_xlabel('Characteristic Frequency (kHz)')
    axs.set_ylabel('Firing Rate (Hz)')
    axs.legend(loc='upper left')

    title = 'Hearing Loss'
    axs.set_title(title)
    fig.tight_layout()

    filename = 'pyramidal_response_curve'
    filename += '-loss'
    fig.savefig(os.path.join(output_dir, f'{filename}.png'), dpi=300)


def main():

    parser = argparse.ArgumentParser(description='run network with array of pyramidal cells spanning frequency range')
    parser.add_argument('--hearing', type=str, choices=['normal', 'loss'], help='type of hearing')
    parser.add_argument('--loss_limit', type=int, default=99e3, help='lower limit for hearing loss (Hz) - default=None')
    parser.add_argument('-if', '--input_frequency', type=int, help='input sound frequency (Hz)')
    parser.add_argument('-rt', '--response_type', type=str, choices=['III', 'IV'], help='pyramdial cell reponse type - only III and IV supported')
    parser.add_argument('-rm', '--response_map', action='store_true', help='create response map')
    parser.add_argument('-c', '--cells_per_band', type=int, help='number of pyramidal cells per characteristic frequency')
    parser.add_argument('-i', '--iterations', type=int, help='number of simulation iterations')
    parser.add_argument('-f', '--force_run', action='store_true', help='force run cell simulation')
    parser.add_argument('-ic', '--enable_ic', action='store_true', help='enable inferior colliculus')
    parser.add_argument('--cohcs', action='store_true', help='run batch iterations of level of outer hair cell impairment')
    parser.add_argument('--cihcs', action='store_true', help='run batch iterations of level of inner hair cell impairment')
    parser.add_argument('--cohc', type=float, default=1.0, help='level of impairment of outer hair cells in cochlea model [0, 1]')
    parser.add_argument('--cihc', type=float, default=1.0, help='level of impairment of inner hair cells in cochlea model [0, 1]')
    parser.add_argument('--hf', type=float, default=1.0, help='homeostasis factor')
    parser.add_argument('--hp_loss', action='store_true', help='add homeostatic plasticiting to hearing loss range')
    parser.add_argument('--hf_loss', type=float, default=1.0, help='homeostasis factor for hearing loss range')
    parser.add_argument('--sgn_loss', type=float, default=0, help='percent loss of SGN inputs in hearing loss range')
    parser.add_argument('--octave_spacing', type=float, default=200.0, help='octave spacing for pyramidal cells')
    
    args = parser.parse_args()

    stims = []
    parallel = True

    # nreps = args.iterations
    cf_fmin = 4e3   # 12e3 for response map otherwise 4e3
    cf_fmax = 40e3
    cf_octavespacing = 1 / args.octave_spacing # 0.5 -> 2, 1 -> 4, 16 -> 66, 32 -> 107, 64 -> 213, 200 -> 665
    cf_n_frequencies = int(np.log2(cf_fmax / cf_fmin) / cf_octavespacing) + 1
    cf_fvals = (
        np.logspace(
            np.log2(cf_fmin / 1000.0),
            np.log2(cf_fmax / 1000.0),
            num=cf_n_frequencies,
            endpoint=True,
            base=2,
        )
        * 1000.0
    )
    # cf_fvals = np.array([10e3])

    if args.response_map: 
        fmin = 4e3
        fmax = 32e3
        octavespacing = 1 / 2.0  # 2 -> 7, 4 -> 13, 8 -> 25, 16 -> 66, 32 -> , 64 -> 262
        n_frequencies = int(np.log2(fmax / fmin) / octavespacing) + 1
        fvals = (
            np.logspace(
                np.log2(fmin / 1000.0),
                np.log2(fmax / 1000.0),
                num=n_frequencies,
                endpoint=True,
                base=2,
            )
            * 1000.0
        )
        input_fvals = fvals

        n_levels = 6  # 2 for debugging, 11 for full, 6 for small response map
        levels = np.linspace(20, 100, n_levels)  # 20, 100, n_levels

        input_type = 'response_map'

    else:
        input_fvals = [args.input_frequency]

        levels = [60]
        n_levels = len(levels)

        input_type = 'single_sound'

        # n_levels = 11  # 2 for debugging, 11 for full
        # levels = np.linspace(20, 100, n_levels)
        # if args.input_frequency > 0:
        #     n_levels = 11  # 2 for debugging, 11 for full
        #     levels = np.linspace(20, 100, n_levels)  # 20, 100, n_levels
        # else:
        #     levels = [60]        

    print(("Frequencies:", cf_fvals / 1000.0))
    print(("Levels:", levels))

    seed = 34657845
    temp = 34.0
    dt = 0.025
    # hearing = args.hearing
    # enable_ic = False
    # force_run = False
    # cells_per_band = args.cells_per_band
    syntype = "simple"
    loss_method = 'hc'
    if args.hp_loss:
        loss_method += '_hf'
    loss_frac = args.sgn_loss
    if loss_frac > 0:
        loss_method += f'_{loss_frac}'

    cohc = args.cohc
    cihc = args.cihc

    resp_type = args.response_type
    response_weights = {'III': {'sgc': 0.00044,
                                'tv': 0.0006,
                                'ds': 0.002228},
                        'IV': {'sgc': 0.00038,
                                'tv': 0.0039,
                                'ds': 0.002228}}

    hf = args.hf

    sgc_weight = np.round(response_weights[resp_type]['sgc']*hf, 7)  # type III - 0.00044, type IV - 0.00038, default - 0.000327
    tv_weight = np.round(response_weights[resp_type]['tv']/hf, 7)    # type III - 0.0006, type IV - 0.0039, default - 0.0027
    ds_weight = np.round(response_weights[resp_type]['ds']/hf, 7)

    syn_opts = {'sgc': {'weight': sgc_weight},
                'tuberculoventral': {'weight': tv_weight},
                'dstellate': {'weight': ds_weight}}

    # if args.hp_loss:
    syn_opts['hf_loss'] = {'sgc': sgc_weight*args.hf_loss,
                            'tuberculoventral': tv_weight/args.hf_loss,
                            'dstellate': ds_weight/args.hf_loss}

    stimpar = {
            "dur": 0.26,
            "pip": 0.1,
            "start": [0.15],
            "baseline": [50, 150],
            "response": [150, 250],
        }

    sim_flag = f'{input_type}-network-{resp_type}'
    if args.cohcs or args.cihcs:
        sim_flag += f'-batch'
        if args.cohcs:
            sim_flag += f'_ohcs'
        if args.cihcs:
            sim_flag += f'_ihcs'
    else:
        sim_flag += f'-ohc_{cohc}-ihc_{cihc}'
    # sim_flag += f'-SPx{sgc_weight}_VPx{tv_weight}'

    if args.enable_ic:
        sim_flag += '_ic'
    if 'loss' in args.hearing:
        sim_flag += f'-{args.loss_limit}loss-{loss_method}'

    cwd = os.getcwd() # os.path.dirname(__file__)
    cachepath = os.path.join('/scratch/kedoxey', "cache_network")
    cachepath += sim_flag

    print(cachepath)
    if not os.path.isdir(cachepath):
        os.mkdir(cachepath)

    sim_dir = os.path.join('/data/scrook/dcnmodel_scratch', 'output', f'{sim_flag}')
    if not os.path.exists(sim_dir):
        os.mkdir(sim_dir)
    
    fig_dir = os.path.join(sim_dir, f'{len(cf_fvals)}cfs_{len(levels)}dbs_{levels[0]}dB_{args.input_frequency}if_{args.cells_per_band}cpb_{args.iterations}nreps')
    if not os.path.exists(fig_dir):
        os.mkdir(fig_dir)
    
    sub_fig_dir = os.path.join(fig_dir, f'SPx{sgc_weight}_VPx{tv_weight}_DPx{ds_weight}')
    if args.hp_loss:
        sub_fig_dir += f'-HFx{args.hf_loss}'
    if not os.path.exists(sub_fig_dir):
        os.mkdir(sub_fig_dir)

    batch_hair_cells = True if args.cohcs or args.cihcs else False

    if batch_hair_cells:

        if args.cohcs and args.cihcs:
            cohcs = np.round(np.arange(0.2,1.1,0.2), 1)
            # cihcs = np.round(np.arange(0.2,1.1,0.2), 1)
            cihcs = np.round(np.arange(0.01,0.2,0.04), 2)
        elif args.cohcs:
            cohcs = np.round(np.arange(0,1.1,0.2), 1)
            cihcs = [1.0 for _ in cohcs]
        else:
            cihcs = np.round(np.arange(0,1.1,0.2), 1)
            cohcs = [1.0 for _ in cihcs]

        hair_cell_msfs = {}

        chc_combos = list(itertools.product(*[cohcs, cihcs]))

        for cohc, cihc in chc_combos:

            hc_dir = os.path.join(fig_dir, f'ohc_{cohc}-ihc_{cihc}')
            if not os.path.exists(hc_dir):
                os.mkdir(hc_dir)

            prot = CharacterizeNetwork(seed=seed, frequencies=cf_fvals, cells_per_band=args.cells_per_band, 
                                    hearing=args.hearing, loss_limit=args.loss_limit, loss_frac=loss_frac,
                                    synapsetype=syntype, enable_ic=args.enable_ic,
                                    cohc=cohc, cihc=cihc, syn_opts=syn_opts)
            pickle.dump(prot.pyramidal_ids_per_band, open(os.path.join(hc_dir, 'pyramidal_ids_per_band.pkl'), 'wb'))

            tasks = []
            for f in input_fvals:
                for db in levels:
                    for i in range(args.iterations):
                        tasks.append((f, db, i))

            results = {}

            print(sub_fig_dir)

            for i, task in enumerate(tasks):

                f, db, iteration = task
                stim = sound.TonePip(
                    rate=100e3,
                    duration=stimpar["dur"],
                    f0=f,
                    dbspl=db,  # dura 0.2, pip_start 0.1 pipdur 0.04
                    ramp_duration=2.5e-3,
                    pip_duration=stimpar["pip"],
                    pip_start=stimpar["start"],
                )

                cachefile = os.path.join(cachepath, f'seed={seed}_nfs={len(cf_fvals)}_f0={f}_dbspl={db}_cpb={args.cells_per_band}_syntype={syntype}_iter={iteration}')
                print(f'cache file exists - {bool(os.path.isfile(cachefile))}')
                
                if (not os.path.isfile(cachefile)) or args.force_run:
                    print(f'running - f = {f}, dbspl = {db}')
                    # result = prot.run(f, db, iteration, seed=i)  # parallelize
                    result = mp.Manager().dict()
                    p1 = mp.Process(target=prot.run, args=(stim, i, result))
                    p1.start()
                    p1.join()
                    result = dict(result)
                    pickle.dump(result, open(cachefile, 'wb'))
                else:
                    print(f'loading - f = {f}, dbspl = {db}')
                    result = pickle.load(open(cachefile, 'rb'))
                
                results[(f, db, iteration)] = (stim, result)

            results_od = prot.reformat_results(results)
            pickle.dump(results_od, open(os.path.join(sub_fig_dir, 'results_od.pkl'), 'wb'))

            metadata = {
                'populations': {'sgc': len(prot.sgc.real_cells()),
                                'dstellate': len(prot.dstellate.real_cells()),
                                'vertical': len(prot.tuberculoventral.real_cells()),
                                'pyramidal': len(prot.pyramidal.real_cells())}
            }

            prot.plot_network_firing_rates(results_od, {'freq': args.input_frequency, 'level': levels[0]}, 
                                           stimpar['baseline'], stimpar['response'], output_dir=sub_fig_dir)

            # avg_msfs, metadata = prot.save_stimulus_firing_rates(results, response=stimpar['response'], output_dir=hc_dir)
        #     prot.plot_stimulus_firing_rates(avg_msfs, cf_fvals, hc_dir)

        #     hair_cell_msfs[(cohc, cihc)] = avg_msfs

        # plot_hair_cell_firing_rates(hair_cell_msfs, cf_fvals, fig_dir)

    else:

        prot = CharacterizeNetwork(seed=seed, frequencies=cf_fvals, cells_per_band=args.cells_per_band, 
                                   hearing=args.hearing, loss_limit=args.loss_limit, loss_frac=loss_frac,
                                   synapsetype=syntype, enable_ic=args.enable_ic,
                                   cohc=cohc, cihc=cihc, syn_opts=syn_opts)
        pickle.dump(prot.pyramidal_ids_per_band, open(os.path.join(sub_fig_dir, 'pyramidal_ids_per_band.pkl'), 'wb'))

        tasks = []
        for f in input_fvals:
            for db in levels:
                for i in range(args.iterations):
                    tasks.append((f, db, i))

        results = {}

        for i, task in enumerate(tasks):

            f, db, iteration = task
            stim = sound.TonePip(
                rate=100e3,
                duration=stimpar["dur"],
                f0=f,
                dbspl=db,  # dura 0.2, pip_start 0.1 pipdur 0.04
                ramp_duration=2.5e-3,
                pip_duration=stimpar["pip"],
                pip_start=stimpar["start"],
            )

            cachefile = os.path.join(cachepath, f'seed={seed}_nfs={len(cf_fvals)}_f0={f}_dbspl={db}_cpb={args.cells_per_band}_syntype={syntype}_iter={iteration}')
            print(f'cache file exists - {bool(os.path.isfile(cachefile))}')
            
            if (not os.path.isfile(cachefile)) or args.force_run:
                print(f'running - f = {f}, dbspl = {db}')
                # result = prot.run(f, db, iteration, seed=i)  # parallelize
                result = mp.Manager().dict()
                p1 = mp.Process(target=prot.run, args=(stim, i, result))
                p1.start()
                p1.join()
                result = dict(result)
                pickle.dump(result, open(cachefile, 'wb'))
            else:
                print(f'loading - f = {f}, dbspl = {db}')
                result = pickle.load(open(cachefile, 'rb'))
            
            results[(f, db, iteration)] = (stim, result)

        _, _, results_od = prot.reformat_results(results)
        pickle.dump(results_od, open(os.path.join(sub_fig_dir, 'results_od.pkl'), 'wb'))

        metadata = {
            'populations': {'sgc': len(prot.sgc.real_cells()),
                            'dstellate': len(prot.dstellate.real_cells()),
                            'vertical': len(prot.tuberculoventral.real_cells()),
                            'pyramidal': len(prot.pyramidal.real_cells())}
        }

        prot.plot_network_firing_rates(results_od, {'freq': args.input_frequency, 'level': levels[0]}, 
                                       stimpar['baseline'], stimpar['response'], output_dir=sub_fig_dir)

        # avg_msfs, metadata = prot.save_stimulus_firing_rates(results, response=stimpar['response'], output_dir=sub_fig_dir)
        # prot.plot_stimulus_firing_rates(avg_msfs, cf_fvals, sub_fig_dir)
        # prot.plot_results(nreps, results, baseline=stimpar['baseline'], response=stimpar['response'], output_dir=fig_dir)

    metadata['stimpar'] = stimpar
    metadata['hearing'] = {'type': args.hearing,
                        'loss_limit': args.loss_limit}
    metadata['cfs'] = {'low': cf_fmin,
                    'hight': cf_fmax}
    metadata['synapse type'] = syntype
    metadata['synapse weights'] = {'sgc': {'weight': float(sgc_weight)},
                                   'tuberculoventral': {'weight': float(tv_weight)},
                                   'dstellate': {'weight': float(ds_weight)},
                                   'hf_loss': {'sgc': float(sgc_weight*args.hf_loss),
                                               'tuberculoventral': float(tv_weight/args.hf_loss),
                                               'dstellate': float(ds_weight/args.hf_loss)}}

    end_t = time.time()
    elapsed_t = end_t-start_t
    hours, remainder = divmod(elapsed_t,3600)
    minutes, seconds = divmod(remainder,60)
    metadata['runtime'] = f'{int(hours)}h:{int(minutes)}m:{int(seconds)}s'
    
    # temp = 5
    filename = os.path.join(sub_fig_dir, f'metadata')
    if 'loss' in args.hearing:
        filename += '-loss'
    if args.enable_ic:
        filename += '-ic'
    with open(os.path.join(sub_fig_dir, f'{filename}.yml'), 'w') as outfile:
        yaml.dump(metadata, outfile,)

if __name__ == '__main__':
    main()
