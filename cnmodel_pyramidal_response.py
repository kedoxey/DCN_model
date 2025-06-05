from cnmodel import populations
from cnmodel.util import sound, random_seed
from cnmodel.protocols import Protocol
from collections import OrderedDict
import os, sys, time
import pickle
import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from neuron import h
import multiprocessing as mp

class CNSoundStim(Protocol):
    def __init__(self, seed, temp=34.0, dt=0.025, hearing='normal', synapsetype="simple"):
        Protocol.__init__(self)

        self.seed = seed
        self.temp = temp
        self.dt = dt

        self.hearing = hearing  # normal or loss

        # Seed now to ensure network generation is stable
        random_seed.set_seed(seed)
        # Create cell populations.
        # This creates a complete set of _virtual_ cells for each population. No
        # cells are instantiated at this point.
        self.sgc = populations.SGC(model="dummy")
        self.dstellate = populations.DStellate()
        self.tstellate = populations.TStellate()
        self.tuberculoventral = populations.Tuberculoventral()
        self.pyramidal = populations.Pyramidal()

        pops = [
            self.sgc,
            self.dstellate,
            self.tuberculoventral,
            self.tstellate,
            self.pyramidal
        ]
        self.populations = OrderedDict([(pop.type, pop) for pop in pops])

        # set synapse type to use in the sgc population - simple is fast, multisite is slower
        # (eventually, we could do this for all synapse types..)
        self.sgc._synapsetype = synapsetype

        # Connect populations.
        # This only defines the connections between populations; no synapses are
        # created at this stage.
        self.sgc.connect(
            self.pyramidal, self.dstellate, self.tuberculoventral, self.tstellate
        )
        self.dstellate.connect(
            self.pyramidal, self.tstellate
        )  # should connect to dstellate as well?
        self.tuberculoventral.connect(self.pyramidal, self.tstellate) 
        self.tstellate.connect(self.pyramidal)

        # Select cells to record from.
        # At this time, we actually instantiate the selected cells.
        frequencies = [16e3]
        cells_per_band = 1
        for f in frequencies:
            pyramidal_cell_ids = self.pyramidal.select(cells_per_band, cf=f, create=True)

        # Now create the supporting circuitry needed to drive the cells we selected.
        # At this time, cells are created in all populations and automatically
        # connected with synapses.
        self.pyramidal.resolve_inputs(depth=2)
        self.target_cell = 'pyramidal'
        self.target_freq = frequencies[0]
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
        # TODO: implement hearing loss
        if 'loss' in self.hearing:
            loss_frac = 0.8
            loss_freq = 24e3
            sgc_ids = self.sgc.real_cells()
            for sgc_id in sgc_ids:
                sgc_cell = self.sgc.get_cell(sgc_id)
                if (sgc_cell.cf > loss_freq) and (len(sgc_cell._spiketrain) > 0):
                    ind_remove = set(random.sample(list(range(len(sgc_cell._spiketrain))), int(loss_frac*len(sgc_cell._spiketrain))))
                    sgc_cell._spiketrain = [n for i, n in enumerate(sgc_cell._spiketrain) if i not in ind_remove]
            temp = 5

        # set up recording vectors
        for pop in self.pyramidal, self.dstellate, self.tstellate, self.tuberculoventral:  # self.bushy, self.dstellate, self.tstellate, self.tuberculoventral:
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
                print("%0.2f / %0.2f" % (h.t, h.tstop))
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
        cell_ind = list(results_od[list(results_od.keys())[0]][1][0].keys())[0][1]

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
        axs.set_title(title)

        fig.colorbar(im)
        fig.tight_layout()
        filename = f'{self.target_cell}_{round(self.target_freq)}-{len(freqs)}fs_{len(levels)}dbs-response_map'
        if 'loss' in self.hearing:
            filename += '-loss'
        fig.savefig(os.path.join(output_dir, f'{filename}.png'), dpi=300)



def main():

    stims = []
    parallel = True

    nreps = 1
    fmin = 4e3
    fmax = 32e3
    octavespacing = 1 / 8.0
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

    n_levels = 11
    levels = np.linspace(20, 100, n_levels)

    print(("Frequencies:", fvals / 1000.0))
    print(("Levels:", levels))

    seed = 34657845
    temp = 34.0
    dt = 0.025
    hearing = 'normal'
    syntype = "multisite"

    cwd = os.getcwd() # os.path.dirname(__file__)
    cachepath = os.path.join(cwd, "cache")
    if 'loss' in hearing:
        cachepath += '_loss'
    print(cachepath)
    if not os.path.isdir(cachepath):
        os.mkdir(cachepath)

    prot = CNSoundStim(seed=seed, hearing=hearing, synapsetype=syntype)

    stimpar = {
        "dur": 0.2,
        "pip": 0.04,
        "start": [0.1],
        "baseline": [50, 100],
        "response": [100, 140],
    }
    tasks = []
    for f in fvals:
        for db in levels:
            for i in range(nreps):
                tasks.append((f, db, i))

    results = {}
    workers = 1 if not parallel else None
    tot_runs = len(fvals) * len(levels) * nreps

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

        cachefile = os.path.join(cachepath, f'seed={seed}_f0={f}_dbspl={db}_syntype={syntype}_iter={iteration}')
        force = False
        if (not os.path.isfile(cachefile)) or force:
            # result = prot.run(f, db, iteration, seed=i)  # parallelize
            result = mp.Manager().dict()
            p1 = mp.Process(target=prot.run, args=(stim, i, result))
            p1.start()
            p1.join()
            result = dict(result)
            pickle.dump(result, open(cachefile, 'wb'))
        else:
            result = pickle.load(open(cachefile, 'rb'))
        
        print(f'f = {f}, dbspl = {db}')
        results[(f, db, iteration)] = (stim, result)

    fig_dir = os.path.join(cwd, 'output', 'response_maps')
    if not os.path.exists(fig_dir):
        os.mkdir(fig_dir)
    prot.plot_results(nreps, results, baseline=stimpar['baseline'], response=stimpar['response'], output_dir=fig_dir)


if __name__ == '__main__':
    main()
