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


class CharacterizeNetwork(Protocol):
    def __init__(self, seed, frequencies, cells_per_band=1, loss_limit=99e3, temp=34.0, dt=0.025, hearing='normal', synapsetype="simple", enable_ic=False):
        Protocol.__init__(self)

        self.base_data = None

        self.seed = seed
        self.temp = temp
        self.dt = dt

        self.hearing = hearing  # normal or loss
        self.enable_ic = enable_ic
        self.loss_limit = loss_limit

        # Seed now to ensure network generation is stable
        random_seed.set_seed(seed)
        # Create cell populations.
        # This creates a complete set of _virtual_ cells for each population. No
        # cells are instantiated at this point.
        self.sgc = populations.SGC(model="dummy", hearing=self.hearing, loss_limit=self.loss_limit)
        self.dstellate = populations.DStellate()
        # self.tstellate = populations.TStellate()
        if self.enable_ic:
            self.pyramidal = populations.Pyramidal(synapsetype='simple')
            self.tuberculoventral = populations.Tuberculoventral(synapsetype='simple')
            self.ic = populations.IC()
        else:
            self.pyramidal = populations.Pyramidal()
            self.tuberculoventral = populations.Tuberculoventral()

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
        rs.seed(self.seed)
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

    def save_stimulus_firing_rates(self, results, response, output_dir):

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

        avg_msfs = {}
        for stim, iterations in list(results_od.values()):

            stim_avg_msfs  = []

            for vec in list(iterations.values()):
                temp = 6
                rep_avg_msfs = []

                for cf, ids in list(self.pyramidal_ids_per_band.items()):

                    msfs = []
                    for pyr_id in ids:
                        spkt = vec[('pyramidal', pyr_id)][1]
                        resp_spkt = spkt[((spkt >= response[0]) & (spkt <= response[1]))]

                        num_spikes = len(resp_spkt)
                        num_isi = num_spikes - 1 if num_spikes > 0 else 0
                        msf = num_isi / (resp_spkt[-1] - resp_spkt[0]) * 1000 if num_spikes > 1 else 1
                        msfs.append(msf)
                    rep_avg_msfs.append(np.mean(msfs))

                stim_avg_msfs.append(rep_avg_msfs)
            
            avg_msfs[stim] = np.average(np.array(stim_avg_msfs), axis=0)

        filepath = os.path.join(output_dir, 'stim_avg_rates.pkl')
        pickle.dump(avg_msfs, open(filepath, 'wb'))

        metadata = {
            'populations': {'sgc': len(self.sgc.real_cells()),
                            'dstellate': len(self.dstellate.real_cells()),
                            'vertical': len(self.tuberculoventral.real_cells()),
                            'pyramidal': len(self.pyramidal.real_cells())}
        }
        if self.enable_ic:
            metadata['populations']['ic'] = len(self.ic.real_cells())
        # temp = 5
        filename = os.path.join(output_dir, f'metadata')
        if 'loss' in self.hearing:
            filename += '-loss'
        if self.enable_ic:
            filename += '-ic'
        with open(os.path.join(output_dir, f'{filename}.yml'), 'w') as outfile:
            yaml.dump(metadata, outfile,)

        return avg_msfs


    def plot_stimulus_firing_rates(self, avg_msfs, freqs, output_dir):

        b = (freqs[-1] - freqs[0]) / np.log(freqs[-1] / freqs[0])
        a = freqs[0] - b * np.log(freqs[0])
        freqs_log = a + b * np.log(freqs)

        fig, axs = plt.subplots(1,1,figsize=(10,5))

        for stim, avg_rates in list(avg_msfs.items()):
            level = stim.opts['dbspl']

            # axs[i].plot(freqs_log, avg_rates, color='tab:blue')  #, 'o-')
            axs.plot(freqs, avg_rates, label=f'{level} dB')  #, color='tab:red')

        xspan = len(freqs)//4
        axs.set_xticks([freqs[i] for i in [0, xspan, xspan*2, xspan*3, xspan*4]])
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



def main():

    parser = argparse.ArgumentParser(description='run network with array of pyramidal cells spanning frequency range')
    parser.add_argument('--hearing', type=str, choices=['normal', 'loss'], help='type of hearing')
    parser.add_argument('--loss_limit', type=int, default=99e3, help='lower limit for hearing loss (Hz) - default=None')
    parser.add_argument('-if', '--input_frequency', type=int, help='input sound frequency (Hz)')
    parser.add_argument('-c', '--cells_per_band', type=int, help='number of pyramidal cells per characteristic frequency')
    parser.add_argument('-i', '--iterations', type=int, help='number of simulation iterations')
    parser.add_argument('-f', '--force_run', action='store_true', help='force run cell simulation')
    parser.add_argument('-ic', '--enable_ic', action='store_true', help='enable inferior colliculus')
    
    args = parser.parse_args()

    stims = []
    parallel = True

    # nreps = args.iterations
    fmin = 4e3
    fmax = 40e3
    octavespacing = 1 / 200.0  # 16 -> 66, 32 -> 107, 64 -> 213, 200 -> 665
    n_frequencies = int(np.log2(fmax / fmin) / octavespacing) + 1
    cf_fvals = (
        np.logspace(
            np.log2(fmin / 1000.0),
            np.log2(fmax / 1000.0),
            num=n_frequencies,
            endpoint=True,
            base=2,
        )
        * 1000.0
    )
    # cf_fvals = np.array([10e3])

    n_levels = 6
    levels = [60]  # [20, 40, 60, 80, 100]  # np.linspace(20, 100, n_levels)  # 20, 100, n_levels

    print(("Frequencies:", cf_fvals / 1000.0))
    print(("Levels:", levels))

    seed = 34657845
    temp = 34.0
    dt = 0.025
    # hearing = args.hearing
    # enable_ic = False
    # force_run = False
    # cells_per_band = args.cells_per_band
    syntype = "multisite"

    sim_flag = 'network_ds2v'

    cwd = os.getcwd() # os.path.dirname(__file__)
    cachepath = os.path.join('/scratch/kedoxey', "cache_network")
    if 'loss' in args.hearing:
        cachepath += f'_{args.loss_limit}loss-m3_60'
        sim_flag += f'_{args.loss_limit}loss-m3_60'
    if args.enable_ic:
        cachepath += '_ic'
        sim_flag += '_ic'
    print(cachepath)
    if not os.path.isdir(cachepath):
        os.mkdir(cachepath)

    sim_dir = os.path.join(cwd, 'output', f'response_maps-{sim_flag}')
    if not os.path.exists(sim_dir):
        os.mkdir(sim_dir)
    fig_dir = os.path.join(sim_dir, f'{len(cf_fvals)}cfs_{len(levels)}dbs_{args.input_frequency}if_{args.cells_per_band}cpb_{args.iterations}nreps')
    if not os.path.exists(fig_dir):
        os.mkdir(fig_dir)

    # loss_lim=args.loss_limit, 
    prot = CharacterizeNetwork(seed=seed, frequencies=cf_fvals, cells_per_band=args.cells_per_band, 
                               hearing=args.hearing, loss_limit=args.loss_limit, 
                               synapsetype=syntype, enable_ic=args.enable_ic)
    pickle.dump(prot.pyramidal_ids_per_band, open(os.path.join(fig_dir, 'pyramidal_ids_per_band.pkl'), 'wb'))

    stimpar = {
        "dur": 0.26,
        "pip": 0.1,
        "start": [0.1],
        "baseline": [50, 100],
        "response": [100, 200],
    }

    input_fvals = [float(args.input_frequency)]
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

    pickle.dump(results, open(os.path.join(fig_dir, 'results_df.pkl'), 'wb'))

    avg_msfs = prot.save_stimulus_firing_rates(results, response=stimpar['response'], output_dir=fig_dir)
    prot.plot_stimulus_firing_rates(avg_msfs, cf_fvals, fig_dir)
    # prot.plot_results(nreps, results, baseline=stimpar['baseline'], response=stimpar['response'], output_dir=fig_dir)


if __name__ == '__main__':
    main()
