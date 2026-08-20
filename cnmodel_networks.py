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
import matplotlib as mpl
import matplotlib.cm as cm
from neuron import h
import multiprocessing as mp
import yaml
from scipy.signal import savgol_filter
from matplotlib.ticker import ScalarFormatter
import itertools
import compress_pickle
import seaborn as sns

COMPRESSION = 'gzip'

parula_colors = [(0.2081, 0.1663, 0.5292), (0.2116, 0.1898, 0.5777), (0.2123, 0.2138, 0.6270), (0.2081, 0.2386, 0.6771), 
                (0.1959, 0.2645, 0.7279), (0.1707, 0.2919, 0.7792), (0.1253, 0.3242, 0.8303), (0.0591, 0.3598, 0.8683), 
                (0.0117, 0.3875, 0.8819), (0.0060, 0.4086, 0.8828), (0.0165, 0.4266, 0.8786), (0.0329, 0.4430, 0.8720), 
                (0.0498, 0.4586, 0.8641), (0.0629, 0.4737, 0.8554), (0.0723, 0.4887, 0.8467), (0.0779, 0.5040, 0.8384), 
                (0.0793, 0.5200, 0.8312), (0.0749, 0.5375, 0.8263), (0.0641, 0.5570, 0.8240), (0.0488, 0.5772, 0.8228), 
                (0.0343, 0.5966, 0.8199), (0.0265, 0.6137, 0.8135), (0.0239, 0.6287, 0.8038), (0.0231, 0.6418, 0.7913), 
                (0.0228, 0.6535, 0.7768), (0.0267, 0.6642, 0.7607), (0.0384, 0.6743, 0.7436), (0.0590, 0.6838, 0.7254), 
                (0.0843, 0.6928, 0.7062), (0.1133, 0.7015, 0.6859), (0.1453, 0.7098, 0.6646), (0.1801, 0.7177, 0.6424), 
                (0.2178, 0.7250, 0.6193), (0.2586, 0.7317, 0.5954), (0.3022, 0.7376, 0.5712), (0.3482, 0.7424, 0.5473), 
                (0.3953, 0.7459, 0.5244), (0.4420, 0.7481, 0.5033), (0.4871, 0.7491, 0.4840), (0.5300, 0.7491, 0.4661), 
                (0.5709, 0.7485, 0.4494), (0.6099, 0.7473, 0.4337), (0.6473, 0.7456, 0.4188), (0.6834, 0.7435, 0.4044), 
                (0.7184, 0.7411, 0.3905), (0.7525, 0.7384, 0.3768), (0.7858, 0.7356, 0.3633), (0.8185, 0.7327, 0.3498), 
                (0.8507, 0.7299, 0.3360), (0.8824, 0.7274, 0.3217), (0.9139, 0.7258, 0.3063), (0.9450, 0.7261, 0.2886), 
                (0.9739, 0.7314, 0.2666), (0.9938, 0.7455, 0.2403), (0.9990, 0.7653, 0.2164), (0.9955, 0.7861, 0.1967), 
                (0.9880, 0.8066, 0.1794), (0.9789, 0.8271, 0.1633), (0.9697, 0.8481, 0.1475), (0.9626, 0.8705, 0.1309), 
                (0.9589, 0.8949, 0.1132), (0.9598, 0.9218, 0.0948), (0.9661, 0.9514, 0.0755), (0.9763, 0.9831, 0.0538)]

# Create the colormap
parula_cmap = mpl.colors.LinearSegmentedColormap.from_list('parula', parula_colors, N=256)

start_t = time.time()

class CharacterizeNetwork(Protocol):
    def __init__(self, seed, frequencies, cells_per_band=1, synapsetype="simple", syn_opts=None, pyr_erev=-62, pyr_erev_loss=-62,
                 hearing='normal', loss_limit=99e3, loss_frac=0, cohc=1, cihc=1, 
                 temp=34.0, dt=0.025, enable_ic=False):
        Protocol.__init__(self)

        self.seed = seed
        self.temp = temp
        self.dt = dt
        self.syn_opts = syn_opts
        self.hearing = hearing  # normal or loss
        self.loss_limit = loss_limit
        self.loss_frac = loss_frac
        self.enable_ic = enable_ic

        # Seed now to ensure network generation is stable
        random_seed.set_seed(seed)
        # Create cell populations.
        # This creates a complete set of _virtual_ cells for each population. No
        # cells are instantiated at this point.
        self.sgc = populations.SGC(model="dummy", synapsetype=synapsetype, hearing=self.hearing, loss_frac=loss_frac, loss_limit=self.loss_limit, cohc=cohc, cihc=cihc)
        self.dstellate = populations.DStellate(synapsetype=synapsetype)

        self.pyramidal = populations.Pyramidal(synapsetype=synapsetype, syn_opts=syn_opts, 
                                               hearing=self.hearing, loss_limit=self.loss_limit, 
                                               erev=pyr_erev, erev_loss=pyr_erev_loss)
        self.tuberculoventral = populations.Tuberculoventral(synapsetype=synapsetype)

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
        self.sgc._synapsetype = synapsetype

        # Connect populations.
        # This only defines the connections between populations; no synapses are
        # created at this stage.
        self.sgc.connect(self.pyramidal, self.dstellate, self.tuberculoventral)
        self.dstellate.connect(self.pyramidal, self.tuberculoventral)
        self.tuberculoventral.connect(self.pyramidal)

        # Select pyramidal cells to record from.
        self.cells_per_band = cells_per_band
        self.pyramidal_ids_per_band = {}
        for f in frequencies:
            pyramidal_cell_ids = self.pyramidal.select(self.cells_per_band, cf=f, create=True)
            self.pyramidal_ids_per_band[f] = pyramidal_cell_ids

        # Now create the supporting circuitry needed to drive the cells we selected.
        # At this time, cells are created in all populations and automatically
        # connected with synapses.
        self.pyramidal.resolve_inputs(depth=2)


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
        for pop in self.pyramidal, self.dstellate, self.tuberculoventral: 
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
            assert isinstance(pop, str)
            cell_ind = getattr(self, pop).get_cell_index(k)
            result[(pop, cell_ind)] = [v, spikes]

        # record SGC spike trains
        for ind in self.sgc.real_cells():
            cell = self.sgc.get_cell(ind)
            result[("sgc", ind)] = [None, cell._spiketrain]


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
            key = 'f0: %0.0f' % f0 if not isinstance(f0, str) else f'f0: {f0}'
            key += '  dBspl: %0.0f' % dbspl
            # key = "f0: %0.0f  dBspl: %0.0f" % (f0, dbspl) 
            results_od.setdefault(key, [stim.opts,{}])
            results_od[key][1][iteration] = result
            stim_order.append((f0, dbspl))
            freqs.add(f0)
            levels.add(dbspl)

        freqs = sorted(list(freqs))
        levels = sorted(list(levels))

        return freqs, levels, results_od


    def plot_response_map(self, cf, nreps, freqs, levels, results, baseline, response, output_dir):

        spont_spikes = 0
        spont_time = 0

        pop_type = 'pyramidal'
        cell_ind = self.pyramidal_ids_per_band[cf][0]

        for stim, iterations in list(results.values()):
            print(stim['f0'], stim['dbspl'])
            for vec in list(iterations.values()):
                spikes = vec[(pop_type, cell_ind)][1]
                spont_spikes += ((spikes >= baseline[0]) * (spikes < baseline[1])).sum()
                spont_time += baseline[1] - baseline[0]

        spont_rate = spont_spikes / spont_time

        matrix_spont = np.zeros((len(levels), len(freqs)))
        matrix_evoked = np.zeros((len(levels), len(freqs)))
        matrix_diff = np.zeros((len(levels), len(freqs)))

        for stim, iteration in list(results.values()):

            for i in range(nreps):

                vec = iteration[i]
                spikes = vec[(pop_type, cell_ind)][1]
                n_spikes_s = ((spikes >= baseline[0]) & (spikes < baseline[1])).sum()
                n_spikes_e = ((spikes >= response[0]) & (spikes < response[1])).sum()

                spont_rate = (n_spikes_s / (baseline[1] - baseline[0])) * 1000 
                evoked_rate = (n_spikes_e / (response[1] - response[0])) * 1000 

                freq = stim['f0']
                level = stim['dbspl']

                j = freqs.index(freq)
                i = levels.index(level)

                # response_map_df.loc[level, [freq]] += n_spikes - spont_rate * (response[1] - response[0])
                matrix_spont[i,j] += spont_rate
                matrix_evoked[i,j] += evoked_rate
                matrix_diff[i,j] += evoked_rate - spont_rate

        matrix_spont /= nreps
        matrix_evoked /= nreps
        matrix_diff /= nreps
        matrix_diff_norm = matrix_diff / np.max(np.abs(matrix_diff))

        b = (freqs[-1] - freqs[0]) / np.log(freqs[-1] / freqs[0])
        a = freqs[0] - b * np.log(freqs[0])
        freqs_log = a + b * np.log(freqs)

        # cmap = sns.color_palette("icefire", as_cmap=True)
        cmap = parula_cmap

        for m_type, m in zip(['spont', 'evoked', 'diff', 'diff_norm'], [matrix_spont, matrix_evoked, matrix_diff, matrix_diff_norm]):
            fig, axs = plt.subplots(1,1,figsize=(5,4))

            if 'norm' in m_type:
                im = axs.pcolormesh(freqs_log, levels, m, cmap=cmap, vmin=-1, vmax=1)
            else:
                im = axs.pcolormesh(freqs_log, levels, m, cmap=cmap)
            axs.set_xlabel('Frequency (kHz)')
            axs.set_ylabel('Sound Level (dBSPL)')
            axs.set_xticks([freqs_log[int(i)] for i in np.linspace(0,len(freqs)-1,5)])
            axs.set_xticklabels([round(np.ceil(freqs[int(i)]/1000)) for i in np.linspace(0,len(freqs)-1,5)])

            title = 'Hearing Loss' if 'loss' in self.hearing else 'Normal Hearing'
            axs.set_title(f'{title} - {m_type}')

            if 'norm' in m_type:
                fig.colorbar(im, ticks=[-1,0,1])
            else:
                fig.colorbar(im)
            fig.tight_layout()
            filename = f'{len(freqs)}fs_{len(levels)}dbs_{cf}cf-response_map'
            if 'loss' in self.hearing:
                filename += '-loss'
            filename += f'-{m_type}'
            fig.savefig(os.path.join(output_dir, f'{filename}.png'), dpi=300)

        rm_data = {'freqs': freqs,
                   'freqs_log': freqs_log,
                   'levels': levels,
                   'matrix_spont': matrix_spont,
                   'matrix_evoked': matrix_evoked,
                   'matrix_diff': matrix_diff,
                   'matrix_diff_norm': matrix_diff_norm}
        pickle.dump(rm_data, open(os.path.join(output_dir, f'DATA-{len(freqs)}fs_{len(levels)}dbs_{cf}cf-response_maps.pkl'), 'wb'))
        # compress_pickle.dump(rm_data, open(os.path.join(output_dir, f'DATA-{filename}.pkl'), 'wb'), compression='gzip')


    def plot_rate_level_curve(self, cf, nreps, freqs, levels, results, baseline, response, output_dir):

        spont_spikes = 0
        spont_time = 0

        pop_type = 'pyramidal'
        pyr_cfs = np.array(list(self.pyramidal_ids_per_band.keys()))
        cell_ind = self.pyramidal_ids_per_band[pyr_cfs[np.argmin(np.abs(cf - pyr_cfs))]][0]

        for stim, iterations in list(results.values()):
            for vec in list(iterations.values()):
                spikes = vec[(pop_type, cell_ind)][1]
                spont_spikes += ((spikes >= baseline[0]) * (spikes < baseline[1])).sum()
                spont_time += baseline[1] - baseline[0]

        spont_rate = spont_spikes / spont_time

        matrix_exact = np.zeros((len(levels), len(freqs)))
        matrix = np.zeros((len(levels), len(freqs)))

        for stim, iteration in list(results.values()):

            for i in range(nreps):

                vec = iteration[i]
                spikes = vec[(pop_type, cell_ind)][1]
                n_spikes = ((spikes >= response[0]) & (spikes < response[1])).sum()

                # freq = stim['f0']
                freq = 'bbn' if 'f0' not in stim.keys() else stim['f0']
                level = stim['dbspl']

                j = freqs.index(freq)
                i = levels.index(level)

                # response_map_df.loc[level, [freq]] += n_spikes - spont_rate * (response[1] - response[0])
                matrix_exact[i,j] += (n_spikes / (response[1] - response[0])) * 1000
                matrix[i,j] += n_spikes - spont_rate * (response[1] - response[0])

        matrix_exact /= nreps
        matrix /= nreps

        # cf_freq_idx = np.argmin(np.abs(np.array(freqs) - cf))

        fig, axs = plt.subplots(1,1,figsize=(5,4))

        axs.plot(levels, matrix_exact)

        axs.set_ylim(bottom=0)
        axs.set_xlabel('Sound Level (dB SPL)')
        axs.set_ylabel('Evoked Firing Rate (spikes/s)')

        axs.set_title(f'{int(cf // 1e3)}kHz CF Rate Level Curve')

        fig.tight_layout()

        filename = f'{len(freqs)}fs_{len(levels)}dbs_{cf}cf-rate_level_curveE'
        if 'loss' in self.hearing:
            filename += '-loss'
        fig.savefig(os.path.join(output_dir, f'{filename}.png'), dpi=300)

        rm_data = {'freqs': freqs,
                   'levels': levels,
                   'matrix': matrix_exact}
        # pickle.dump(rm_data, open(os.path.join(output_dir, f'DATA-{filename}.pkl'), 'wb'))
        compress_pickle.dump(rm_data, open(os.path.join(output_dir, f'DATA-{filename}.pkl'), 'wb'), compression=COMPRESSION)


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
            
            rates_key = ('bbn', stim['dbspl']) if 'f0' not in stim.keys() else (stim['f0'], stim['dbspl'])
            avg_rates[rates_key] = stim_avg_rates
            cell_ids[rates_key] = stim_cell_ids[0]

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
            
            rates_key = ('bbn', stim['dbspl']) if 'f0' not in stim.keys() else (stim['f0'], stim['dbspl'])
            avg_rates[rates_key] = stim_avg_rates
            cell_ids[rates_key] = stim_cell_ids[0]

        return cell_ids, avg_rates
    

    def convert_to_log(self, freqs):

        b = (freqs[-1] - freqs[0]) / np.log(freqs[-1] / freqs[0])
        a = freqs[0] - b * np.log(freqs[0])
        freqs_log = a + b * np.log(freqs)

        return freqs_log


    def plot_network_firing_rates(self, nreps, freq, level, results, baseline, response, output_dir):

        colors = {'pyramidal': 'tab:blue',
                  'vertical': 'tab:green',
                  'dstellate': 'tab:orange',
                  'sgc': 'tab:purple'}

        pyr_cfs = self.pyramidal._get_cf_array(species='mouse')
        vert_cfs = self.tuberculoventral._get_cf_array(species='mouse')
        ds_cfs = self.dstellate._get_cf_array(species='mouse')
        sgc_cfs = self.sgc._get_cf_array(species='mouse')

        ### Evoked Activity ###
        # pyr_ids, pyr_rates = self.get_evoked_firing_rates('pyramidal', results, response)
        # ds_ids, ds_rates = self.get_evoked_firing_rates('dstellate', results, response)
        # vert_ids, vert_rates = self.get_evoked_firing_rates('tuberculoventral', results, response)
        # sgc_ids, sgc_rates = self.get_evoked_firing_rates('sgc', results, response)

        evoked_rates = {'pyramidal': self.get_evoked_firing_rates('pyramidal', results, response),
                        'tuberculoventral': self.get_evoked_firing_rates('tuberculoventral', results, response),
                        'dstellate': self.get_evoked_firing_rates('dstellate', results, response),
                        'sgc': self.get_evoked_firing_rates('sgc', results, response)}

        spont_rates = {'pyramidal': self.get_spontaneous_firing_rates('pyramidal', results, baseline),
                       'tuberculoventral': self.get_spontaneous_firing_rates('tuberculoventral', results, baseline),
                       'dstellate': self.get_spontaneous_firing_rates('dstellate', results, baseline),
                       'sgc': self.get_spontaneous_firing_rates('sgc', results, baseline)}
                
        for firing_type, firing_rates in zip(['evoked', 'spontaneous'], [evoked_rates, spont_rates]):

            pyr_ids, pyr_rates = firing_rates['pyramidal']
            vert_ids, vert_rates = firing_rates['tuberculoventral']
            ds_ids, ds_rates = firing_rates['dstellate']
            sgc_ids, sgc_rates = firing_rates['sgc']

            if len(pyr_ids[(freq, level)]) > 1:
                fig, axs = plt.subplots(4,1,figsize=(15,8), sharex=True)
            else:
                fig, axs = plt.subplots(4,1,figsize=(15,8))
            axs = axs.ravel()

            if len(pyr_ids[(freq, level)]) > 1:
                axs[0].plot(pyr_cfs[pyr_ids[(freq, level)]], pyr_rates[(freq, level)], color=colors['pyramidal'], zorder=13, alpha=0.8, label='pyramidal')
                # axs[0].legend(loc='upper right')
                axs[0].set_title('Pyramidal')
                # axs[0].set_xlim([0,sgc_cfs[-1]])
                axs[0].set_xscale('log')
                axs[0].set_xticks([5000, 10000, 15000, 20000, 25000, 30000, 35000, 40000])
                axs[0].xaxis.set_major_formatter(ScalarFormatter())
            else:
                key = 'f0: %0.0f' % freq if not isinstance(freq, str) else f'f0: {freq}'
                key += '  dBspl: %0.0f' % level
                t = results[key][1][0]['t']
                pyr_id = pyr_ids[(freq, level)][0]
                # pyr_v = np.array([results[f'f0: {freq}  dBspl: {level}'][1][i][('pyramidal', pyr_id)][0] for i in range(nreps)])
                # pyr_v_avg = np.average(pyr_v, axis=0)
                pyr_v = results[key][1][int(np.floor(nreps/2))][('pyramidal', pyr_id)][0]
                axs[0].plot(t, pyr_v, color=colors['pyramidal'], zorder=13, label='pyramidal')
                axs[0].axvspan(response[0], response[1], color='grey', alpha=0.3, zorder=1)
                # axs[0].legend(loc='upper right')
                axs[0].set_title(f'Pyramidal - {pyr_rates[(freq, level)]}')
                axs[0].set_xlabel('Time (ms)')
                axs[0].set_ylabel('Voltage (mV)')
                axs[0].set_xlim([t[0], t[-1]])

            axs[1].plot(vert_cfs[vert_ids[(freq, level)]], vert_rates[(freq, level)], color=colors['vertical'], zorder=13, alpha=0.8, label='vertical')
            # axs[1].legend(loc='upper right')
            axs[1].set_title('Vertical')
            # axs[1].set_xlim([0,sgc_cfs[-1]])

            axs[2].plot(ds_cfs[ds_ids[(freq, level)]], ds_rates[(freq, level)], color=colors['dstellate'], zorder=13, alpha=0.8, label='dstellate')
            # axs[2].legend(loc='upper right')
            axs[2].set_title('DStellate')
            # axs[2].set_xlim([0,sgc_cfs[-1]])

            axs[3].plot(sgc_cfs[sgc_ids[(freq, level)]], sgc_rates[(freq, level)], color=colors['sgc'], zorder=13, alpha=0.8, label='sgc')
            # axs[3].legend(loc='upper right')
            axs[3].set_title('SGC')
            # axs[3].set_xlim([0,sgc_cfs[-1]])

            axs[-1].set_xlabel('Characteristic Frequency (Hz)')
            axs[-1].set_ylabel('Firing Rate (Hz)')

            if 'loss' in self.hearing:
                for ax in axs:
                    ax.axvspan(self.loss_limit, pyr_cfs[-1], color='grey', alpha=0.15, zorder=1)
                    
            for i in range(1,4):
                axs[i].set_xscale('log')
                axs[i].set_xticks([5000, 10000, 15000, 20000, 25000, 30000, 35000, 40000])
                axs[i].xaxis.set_major_formatter(ScalarFormatter())

            fig.suptitle(f'{firing_type.title()} Activity')
            fig.tight_layout()

            filename = f'population_{firing_type}_activity'
            if 'loss' in self.hearing:
                filename += '-loss'
            fig.savefig(os.path.join(output_dir, f'{filename}.png'), dpi=300)
        
        fr_data = {'spont_rates': spont_rates,
                   'evoked_rates': evoked_rates}
        pickle.dump(fr_data, open(os.path.join(output_dir, f'DATA-firing_rates.pkl'), 'wb'))
        print('!! PYRAMIDAL FIRING RATE DATA !!')
        print(f"SPONTANEOUS - {spont_rates['pyramidal']}")
        print(f"EVOKED - {evoked_rates['pyramidal']}")


    def plot_results(self, cf, nreps, freqs, levels, results, baseline, response, input_type, output_dir):

        if 'map' in input_type:
            self.plot_response_map(cf, nreps, freqs, levels, results, baseline, response, output_dir)
        elif 'rate' in input_type:
            self.plot_rate_level_curve(cf, nreps, freqs, levels, results, baseline, response, output_dir)
        else:
            self.plot_network_firing_rates(nreps, freqs[0], levels[0], results, baseline, response, output_dir)


def main():

    parser = argparse.ArgumentParser(description='run network with either a single pyramidal cell or a population of pyramidal cells and their presynaptic inputs')
    parser.add_argument('--network_type', type=str, choices=['SPC', 'PCA'], help='type of network to constrct - either single pyramidal (SPC) or population of pyramidal (PCA)')
    parser.add_argument('-cf', '--characteristic_frequency', type=int, default=22000, help='characteristic frequency (Hz) of single pyramidal cell')
    parser.add_argument('-rt', '--response_type', type=str, choices=['III', 'IV'], help='pyramdial cell reponse type - only III and IV supported')
    parser.add_argument('--erev', type=float, default=-62, help='pyramidal cell leak reversal potential')
    parser.add_argument('--octave_spacing', type=float, default=200.0, help='octave spacing for pyramidal cells')
    parser.add_argument('-c', '--cells_per_band', type=int, default=1, help='number of pyramidal cells per characteristic frequency')
    parser.add_argument('-i', '--iterations', type=int, help='number of simulation iterations')
    parser.add_argument('-f', '--force_run', action='store_true', help='force run cell simulation')
    parser.add_argument('-vs', '--verbose_save', action='store_true', default=False, help='save all outputs of simulation')
    parser.add_argument('--full_spont', action='store_true', default=False, help='10 second stimulus conditions for simulation to calcualte spontaneous firing rate')

    parser.add_argument('-if', '--input_frequency', type=int, help='input sound frequency (Hz)')
    parser.add_argument('-bbn', action='store_true', help='broadband noise input')
    parser.add_argument('-idb', '--input_level', default=60, type=int, help='input level (dB SPL) of single input sound')
    parser.add_argument('-rm', '--response_map', action='store_true', help='create response map')
    parser.add_argument('--rm_fs', type=int, default=25, help='number of frequencies in response map')
    parser.add_argument('--rm_ls', type=int, default=11, help='number of sound levels in response map')
    parser.add_argument('-rl', '--rate_level', action='store_true', help='generate rate level curve')

    parser.add_argument('-sp', type=float, default=-1, help='sgc -> pyramidal synaptic weight')
    parser.add_argument('-vp', type=float, default=-1, help='vertical -> pyramidal synaptic weight')
    parser.add_argument('-dp', type=float, default=-1, help='dstellate -> pyramidal synaptic weight')

    parser.add_argument('--hearing', type=str, choices=['normal', 'loss'], help='type of hearing')
    parser.add_argument('--loss_limit', type=int, default=99e3, help='lower limit for hearing loss (Hz) - default=None')
    parser.add_argument('--cohcs', action='store_true', help='run batch iterations of level of outer hair cell impairment')
    parser.add_argument('--cihcs', action='store_true', help='run batch iterations of level of inner hair cell impairment')
    parser.add_argument('--cohc', type=float, default=1.0, help='level of impairment of outer hair cells in cochlea model [0, 1]')
    parser.add_argument('--cihc', type=float, default=1.0, help='level of impairment of inner hair cells in cochlea model [0, 1]')
    parser.add_argument('--hf', type=float, default=1.0, help='homeostasis factor')
    parser.add_argument('--hp_loss', action='store_true', help='add homeostatic plasticity to hearing loss range')
    parser.add_argument('--hf_loss', type=float, default=1.0, help='homeostasis factor for hearing loss range')
    parser.add_argument('--sgn_loss', type=float, default=0, help='percent loss of SGN inputs in hearing loss range')
    parser.add_argument('--erev_loss', type=float, default=-62, help='pyramidal cell leak reversal potential in hearing loss range')
    
    args = parser.parse_args()

    ### Input frequencies and levels
    if args.response_map:

        fmin = 4e3
        fmax = 32e3
        # octavespacing = 1 / 8.0  # 2 -> 7, 4 -> 13, !8 -> 25!, 16 -> 66, 32 -> , 64 -> 262
        # n_frequencies = int(np.log2(fmax / fmin) / octavespacing) + 1
        n_frequencies = args.rm_fs
        octavespacing = np.log2(fmax / fmin) / (n_frequencies - 1)
        input_fvals = (
            np.logspace(
                np.log2(fmin / 1000.0),
                np.log2(fmax / 1000.0),
                num=n_frequencies,
                endpoint=True,
                base=2,
            )
            * 1000.0
        )

        n_levels = args.rm_ls  # 2 for debugging, 11 for full, 6 for small response map
        levels = np.linspace(20, 100, n_levels)  # 20, 100, n_levels

        input_type = 'response_map'
        input_info = f'{len(input_fvals)}fs_{n_levels}ls'
    
    elif args.rate_level:

        input_fvals = np.array([args.input_frequency]) if not args.bbn else np.array(['bbn'])
        n_frequencies = len(input_fvals)

        n_levels = 13   # 13
        levels = np.linspace(4, 100, n_levels)
        levels = np.insert(levels,0,0)
        n_levels += 1

        input_type = 'rate_level'
        input_info = f'{args.input_frequency // 1000}kHz' if not args.bbn else 'BBN'
        input_info += f'_{n_levels}ls'

    else:

        n_frequencies = 1
        input_fvals = np.array([args.input_frequency]) if not args.bbn else np.array(['bbn'])
        
        n_levels = 1
        levels = np.array([args.input_level])

        input_type = 'single_sound'
        
        input_info = f'{args.input_frequency // 1000}kHz' if not args.bbn else 'BBN'
        input_info += f'_{args.input_level}dB'

    if not args.bbn:
        print(("Input Frequencies:", input_fvals / 1000.0))
    else:
        print(('Input: Broadband Noise'))
    print(("Input Levels:", levels))

    ### Pyramidal characerisitc frequencies
    if 'PCA' in args.network_type:

        cf_fmin = 4e3   # 12e3 for response map otherwise 4e3
        cf_fmax = 40e3
        cf_octavespacing = 1 / 200 # 0.5 -> 2, 1 -> 4, 16 -> 66, 32 -> 107, 64 -> 213, 200 -> 665
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
    
    else:

        cf_fvals = np.array([args.characteristic_frequency])

    seed = 34657845
    temp = 34.0
    dt = 0.025

    # stimpar = {
    #     "dur": 0.26,
    #     "pip": 0.1,
    #     "start": [0.15],
    #     "baseline": [50, 150],
    #     "response": [150, 250]
    #     }

    if args.full_spont:

        stimpar = {"dur": 1.25,
                   "pip": 1,
                   "start": [0.25],
                   "baseline": [0, 250],
                   "response": [250, 1250]}
    else:

        stimpar = {"dur": 0.26,
                   "pip": 0.1,
                   "start": [0.125],
                   "baseline": [25, 125],
                   "response": [125, 225]}

    syntype = "simple"
    loss_method = 'hc'
    if args.hp_loss:
        loss_method += '_hf_P'
    loss_frac = args.sgn_loss
    if loss_frac > 0:
        loss_method += f'_{loss_frac}'

    response_weights = {'III': {'sgc': 0.00036,
                                 'tv': 0.0014,
                                 'ds': 0.002228},
                        'IV': {'sgc': 0.00028,
                                'tv': 0.0038,
                                'ds': 0.002228}}

    if args.sp != -1:
        response_weights[args.response_type]['sgc'] = args.sp
    if args.vp != -1:
        response_weights[args.response_type]['tv'] = args.vp
    if args.dp != -1:
        response_weights[args.response_type]['ds'] = args.dp
    
    sgc_weight = np.round(response_weights[args.response_type]['sgc']*args.hf, 7)  # type III - 0.00036, type IV - 0.00028, default - 0.000327
    tv_weight = np.round(response_weights[args.response_type]['tv']/args.hf, 7)    # type III - 0.0014, type IV - 0.0038, default - 0.0027
    ds_weight = np.round(response_weights[args.response_type]['ds']/args.hf, 7)

    syn_opts = {'sgc': {'weight': sgc_weight},
                'tuberculoventral': {'weight': tv_weight},
                'dstellate': {'weight': ds_weight}}
    
    syn_opts['hf_loss'] = {'sgc': sgc_weight*args.hf_loss,
                           'tuberculoventral': tv_weight/args.hf_loss,
                           'dstellate': ds_weight/args.hf_loss}

    network_flag = f'{args.network_type}'
    if 'SPC' in args.network_type:
        network_flag += f'-{args.characteristic_frequency // 1000}kHz'
    network_flag += f'-{args.response_type}-{input_type}'
    network_dir = os.path.join('/data/scrook/dcnmodel_scratch/output/', network_flag)
    if not os.path.exists(network_dir):
        os.mkdir(network_dir)
    
    input_flag = f'{len(cf_fvals)}cfs-{input_info}-{args.iterations}nreps'
    input_dir = os.path.join(network_dir, input_flag)
    if not os.path.exists(input_dir):
        os.mkdir(input_dir)

    if 'normal' in args.hearing:
        hearing_flag = 'NORMAL'
    else:
        hearing_flag = f'LOSS@{args.loss_limit // 1000}kHz-cohc_{args.cohc}-cihc_{args.cihc}-{100-args.sgn_loss}xSGN'
        if args.erev_loss != -62:
            hearing_flag += f'-Erev{int(args.erev_loss)}'
    hearing_dir = os.path.join(input_dir, hearing_flag)
    if not os.path.exists(hearing_dir):
        os.mkdir(hearing_dir)
    
    weight_flag = f'SPx{sgc_weight}_VPx{tv_weight}_DPx{ds_weight}-HFx{args.hf_loss}'
    if args.erev != -62:
        weight_flag += f'-Erev{int(args.erev)}'
    weight_dir = os.path.join(hearing_dir, weight_flag)
    if not os.path.exists(weight_dir):
        os.mkdir(weight_dir)

    sim_flag = f'{network_flag}-{input_flag}-{hearing_flag}-{weight_flag}'
    print(sim_flag)

    cachepath = os.path.join('/scratch/kedoxey/cache', sim_flag)
    if not os.path.exists(cachepath):
        os.mkdir(cachepath)

    temp = 5

    prot = CharacterizeNetwork(seed=seed, frequencies=cf_fvals, cells_per_band=args.cells_per_band, pyr_erev=args.erev, pyr_erev_loss=args.erev_loss,
                                   hearing=args.hearing, loss_limit=args.loss_limit, loss_frac=loss_frac,
                                   synapsetype=syntype,
                                   cohc=args.cohc, cihc=args.cihc, syn_opts=syn_opts)
    # pickle.dump(prot.pyramidal_ids_per_band, open(os.path.join(weight_dir, 'pyramidal_ids_per_band.pkl'), 'wb'))
    compress_pickle.dump(prot.pyramidal_ids_per_band, open(os.path.join(weight_dir, 'pyramidal_ids_per_band.pkl'), 'wb'), compression=COMPRESSION)

    tasks = []
    for f in input_fvals:
        for db in levels:
            for i in range(args.iterations):
                tasks.append((f, db, i))

    results = {}

    for i, task in enumerate(tasks):

        f, db, iteration = task
        if isinstance(f, str):
            stim = sound.NoisePip(
                rate=100e3,
                duration=stimpar["dur"],
                dbspl=db,
                ramp_duration=2.5e-3,
                pip_duration=stimpar["pip"],
                pip_start=stimpar["start"],
                seed=seed+i
            )
        else:
            stim = sound.TonePip(
                rate=100e3,
                duration=stimpar["dur"],
                f0=f,
                dbspl=db,  # dura 0.2, pip_start 0.1 pipdur 0.04
                ramp_duration=2.5e-3,
                pip_duration=stimpar["pip"],
                pip_start=stimpar["start"],
            )

        ## TODO: notch noise

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
            # pickle.dump(result, open(cachefile, 'wb'))
            compress_pickle.dump(result, open(cachefile, 'wb'), compression=COMPRESSION)
        else:
            print(f'loading - f = {f}, dbspl = {db}')
            # result = pickle.load(open(cachefile, 'rb'))
            result = compress_pickle.load(open(cachefile, 'rb'), compression=COMPRESSION)
        
        results[(f, db, iteration)] = (stim, result)

    freqs, levels, results_od = prot.reformat_results(results)
    # pickle.dump(results_od, open(os.path.join(weight_dir, 'results_od.pkl'), 'wb'))
    if args.verbose_save:
        compress_pickle.dump(results_od, open(os.path.join(weight_dir, 'results_od.pkl'), 'wb'), compression=COMPRESSION)

    metadata = {
        'populations': {'sgc': len(prot.sgc.real_cells()),
                        'dstellate': len(prot.dstellate.real_cells()),
                        'vertical': len(prot.tuberculoventral.real_cells()),
                        'pyramidal': len(prot.pyramidal.real_cells())}
    }

    prot.plot_results(args.characteristic_frequency, args.iterations, freqs, levels, results_od, stimpar['baseline'], stimpar['response'], input_type, weight_dir)

    metadata['stimpar'] = stimpar
    metadata['hearing'] = {'type': args.hearing,
                           'loss_limit': args.loss_limit}
    if 'SPC' in args.network_type:
        metadata['cf'] = args.characteristic_frequency
    else:
        metadata['cfs'] = {'low': cf_fmin, 'max': cf_fmax}
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
    
    with open(os.path.join(weight_dir, f'metadata.yml'), 'w') as outfile:
        yaml.dump(metadata, outfile,)


if __name__ == '__main__':
    main()


