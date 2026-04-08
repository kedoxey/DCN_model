from cnmodel import populations
from cnmodel.util import sound, random_seed
from cnmodel.protocols import Protocol
from collections import OrderedDict
import os, sys, time
import pickle
import random
import time
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from neuron import h
import multiprocessing as mp
import seaborn as sns
import yaml
import itertools

start_t = time.time()

class CNSoundStim(Protocol):
    def __init__(self, seed, characteristic_frequency, temp=34.0, dt=0.025, 
                 hearing='normal', loss_limit=99e3, cohc=1.0, cihc=1.0,
                 synapsetype="simple", syn_opts=None, include_ic=False, pyr_erev=-62):
        Protocol.__init__(self)

        self.base_data = None

        self.seed = seed
        self.temp = temp
        self.dt = dt
        self.syn_opts = syn_opts
        self.hearing = hearing  # normal or loss
        self.loss_limit = loss_limit
        self.include_ic = include_ic

        # Seed now to ensure network generation is stable
        random_seed.set_seed(seed)
        # Create cell populations.
        # This creates a complete set of _virtual_ cells for each population. No
        # cells are instantiated at this point.
        self.sgc = populations.SGC(model="dummy", synapsetype=synapsetype, hearing=self.hearing, loss_limit=self.loss_limit, cohc=cohc, cihc=cihc)
        self.dstellate = populations.DStellate(synapsetype=synapsetype)
        # self.tstellate = populations.TStellate()
        # if self.include_ic:
        #     self.pyramidal = populations.Pyramidal(synapsetype='simple')
        #     self.tuberculoventral = populations.Tuberculoventral(synapsetype='simple')
        #     self.ic = populations.IC()
        # else:

        # syn_opts = {'sgc': {'weight': 0.0327},                 # default = 0.000327
        #             'tuberculoventral': {'weight': 0.002705}}  # default = 0.002705
        self.pyramidal = populations.Pyramidal(erev=pyr_erev, synapsetype=synapsetype, syn_opts=syn_opts)
        self.tuberculoventral = populations.Tuberculoventral(synapsetype=synapsetype)

        pops = [
            self.sgc,
            self.dstellate,
            self.tuberculoventral,
            # self.tstellate,
            self.pyramidal
        ]
        if self.include_ic:
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

        if self.include_ic:
            self.pyramidal.connect(self.ic)
            self.ic.connect(self.pyramidal, self.tuberculoventral)

        # Select cells to record from.
        # At this time, we actually instantiate the selected cells.
        # frequencies = [16e3]
        self.characteristic_frequency = characteristic_frequency
        cells_per_band = 1
        pyramidal_cell_ids = self.pyramidal.select(cells_per_band, cf=self.characteristic_frequency, create=True)
        self.pyramidal_id = pyramidal_cell_ids[0]
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

        self.sgc.set_sound_stim(stim, parallel=False)  #, hearing=self.hearing)
        # if 'loss' in self.hearing:
        #     loss_frac = 0.95
        #     loss_freq = 20e3
        #     sgc_ids = self.sgc.real_cells()
        #     for sgc_id in sgc_ids:
        #         sgc_cell = self.sgc.get_cell(sgc_id)
        #         if (sgc_cell.cf > loss_freq) and (len(sgc_cell._spiketrain) > 0):
        #             ind_remove = set(random.sample(list(range(len(sgc_cell._spiketrain))), int(loss_frac*len(sgc_cell._spiketrain))))
        #             sgc_cell._spiketrain = [n for i, n in enumerate(sgc_cell._spiketrain) if i not in ind_remove]

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

    def plot_results(self, nreps, results, baseline, response, input_type, output_dir):

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
        pickle.dump(results_od, open(os.path.join(output_dir, f'{len(freqs)}fs_{len(levels)}dbs_{self.characteristic_frequency}cf-results_od.pkl'), 'wb'))

        freqs = sorted(list(freqs))
        levels = sorted(list(levels))

        spont_spikes = 0
        spont_time = 0

        pop_type = 'pyramidal'
        cell_ind = self.pyramidal_id

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

        matrix_norm = matrix / np.max(np.abs(matrix))

        if 'map' in input_type:
            # plot response map

            cmap = sns.color_palette("icefire", as_cmap=True)

            for m_type, m in zip(['raw', 'norm'], [matrix, matrix_norm]):
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
                if self.include_ic:
                    title += ' with IC'
                axs.set_title(f'{title} - {m_type}')

                if 'norm' in m_type:
                    fig.colorbar(im, ticks=[-1,0,1])
                else:
                    fig.colorbar(im)
                fig.tight_layout()
                filename = f'{len(freqs)}fs_{len(levels)}dbs_{self.characteristic_frequency}cf-response_map'
                if 'loss' in self.hearing:
                    filename += '-loss'
                if self.include_ic:
                    filename += '-ic'
                filename += f'-{m_type}'
                fig.savefig(os.path.join(output_dir, f'{filename}.png'), dpi=300)

        elif 'rate' in input_type:
            # plot rate level curve

            cf_freq_idx = np.argmin(np.abs(np.array(freqs) - self.characteristic_frequency))

            fig, axs = plt.subplots(1,1,figsize=(5,4))

            axs.plot(levels, matrix[:,cf_freq_idx])

            axs.set_xlabel('Sound Level (dB SPL)')
            axs.set_ylabel('Evoked Firing Rate (spikes/s)')

            axs.set_title('Characteristic Frequency Rate Level Curve')

            fig.tight_layout()

            filename = f'{len(freqs)}fs_{len(levels)}dbs_{self.characteristic_frequency}cf-rate_level_curve'
            if 'loss' in self.hearing:
                filename += '-loss'
            if self.include_ic:
                filename += '-ic'
            fig.savefig(os.path.join(output_dir, f'{filename}.png'), dpi=300)

        else:
            # plot average voltage trace
            traces = []
            for stim, iterations in list(results_od.values()):
                for vec in list(iterations.values()):
                    trace = vec[(pop_type, cell_ind)][0]
                    traces.append(trace)

            t = list(results_od.values())[0][1][0]['t']
            average_trace = np.average(traces, axis=0)

            fig, axs = plt.subplots(1,1,figsize=(10,5))

            axs.axvspan(response[0], response[1], color='grey', alpha=0.3)
            axs.plot(t, average_trace)

            axs.set_ylabel('Voltage (mV)')
            axs.set_xlabel('Time (ms)')

            axs.set_title('Average Pyramidal Voltage Trace Across Trials')
            
            fig.tight_layout()
            filename = f'{len(freqs)}fs_{len(levels)}dbs_{self.characteristic_frequency}cf-pyramidal_trace'
            fig.savefig(os.path.join(output_dir, f'{filename}.png'), dpi=300)

        rm_data = {'freqs': freqs,
                   'freqs_log': freqs_log,
                   'levels': levels,
                   'matrix': matrix}
        pickle.dump(rm_data, open(os.path.join(output_dir, f'DATA-{filename}.pkl'), 'wb'))

        metadata = {
            'spont_rate': float(round(spont_rate, 5)),
            'populations': {'sgc': len(self.sgc.real_cells()),
                            'dstellate': len(self.dstellate.real_cells()),
                            'ventral': len(self.tuberculoventral.real_cells()),
                            'pyramidal': len(self.pyramidal.real_cells())}
        }
        # temp = 5

        return metadata


    def get_response_rate(self, stim):

        f0 = stim.opts['f0']
        db = stim.opts['dbspl']

        f0_ind = np.where(self.base_data['freqs'] == f0)[0][0]
        db_ind = np.where(self.base_data['levels'] == db)[0][0]

        response_rate = self.base_data['matrix'][db_ind, f0_ind]

        return response_rate


def run_simulation(prot, fvals, levels, args, parallel, seed, stimpar, syntype, cachepath, input_type, output_dir):

    tasks = []
    for f in fvals:
        for db in levels:
            for i in range(args.iterations):
                tasks.append((f, db, i))

    results = {}
    workers = 1 if not parallel else None
    tot_runs = len(fvals) * len(levels) * args.iterations

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

        cachefile = os.path.join(cachepath, f'seed={seed}_f0={f}_dbspl={db}_cf={args.characteristic_frequency}_syntype={syntype}_iter={iteration}')
    
        if (not os.path.isfile(cachefile)) or args.force_run:
            print(f'running - f = {f}, dbspl = {db}')
            print(cachefile)
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

    metadata = prot.plot_results(args.iterations, results, baseline =stimpar['baseline'], response=stimpar['response'], input_type=input_type, output_dir=output_dir)
    
    return metadata


def main():

    parser = argparse.ArgumentParser(description='run network with single pyramidal cell for various sound levels and frequencies')
    parser.add_argument('--hearing', type=str, choices=['normal', 'loss'], help='type of hearing')
    parser.add_argument('--loss_limit', default=99e3, type=int, help='lower frequency limit for hearing loss')
    parser.add_argument('-if', '--input_frequency', type=int, help='input sound frequency (Hz), if not generating response map')
    parser.add_argument('-idb', '--input_level', default=60, type=int, help='input level (dB SPL) of single input sound')
    parser.add_argument('-rm', '--response_map', action='store_true', help='generate response map')
    parser.add_argument('-rl', '--rate_level', action='store_true', help='generate rate level curve')
    parser.add_argument('-i', '--iterations', type=int, help='number of simulation iterations')
    parser.add_argument('-cf', '--characteristic_frequency', type=int, help='characteristic frequency (Hz) of pyramidal cell')
    parser.add_argument('-f', '--force_run', action='store_true', help='force run cell simulation')
    parser.add_argument('-ic', '--include_ic', action='store_true', help='include inferior colliculus')
    parser.add_argument('--cohc', type=float, default=1.0, help='level of impairment of outer hair cells in cochlea model [0, 1]')
    parser.add_argument('--cihc', type=float, default=1.0, help='level of impairment of inner hair cells in cochlea model [0, 1]')
    parser.add_argument('--cohcs', action='store_true', help='run batch iterations of level of outer hair cell impairment')
    parser.add_argument('--cihcs', action='store_true', help='run batch iterations of level of inner hair cell impairment')
    
    args = parser.parse_args()

    stims = []
    parallel = True

    if args.response_map:

        fmin = 4e3
        fmax = 32e3
        octavespacing = 1 / 8.0  # 2 -> 7, 4 -> 13, 8 -> 25, 16 -> 66, 32 -> , 64 -> 262
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

        n_levels = 11   # 11
        levels = np.linspace(20, 100, n_levels)  # 20, 100, n_levels

        input_type = 'response_map'

    elif args.rate_level:
        fvals = np.array([args.input_frequency])
        n_frequencies = len(fvals)

        n_levels = 13   # 11
        levels = np.linspace(4, 100, n_levels)
        levels = np.insert(levels,0,0)
        n_levels += 1

        input_type = 'rate_level'
    
    else:

        n_frequencies = 1
        fvals = np.array([args.input_frequency])
        
        n_levels = 1
        levels = np.array([args.input_level])

        input_type = 'single_sound'

    print(("Frequencies:", fvals / 1000.0))
    print(("Levels:", levels))

    seed = 34657845
    temp = 34.0
    dt = 0.025
    syntype = "simple"  # simple or multisite
    loss_method = 'hc'
    loss_frac = 70
    cohc = args.cohc
    cihc = args.cihc

    erev = -60  # -62 default

    stimpar = {
        "dur": 0.3,
        "pip": 0.1,
        "start": [0.125],       # equals response start in seconds
        "baseline": [25, 125],  # baseline duration has to equal response duration
        "response": [125, 225],
        }

    resp_type = 'IV'

    if args.response_map:
        sim_flag = f'{input_type}-{args.characteristic_frequency // 1000}kHZ_single_cell-{syntype}_syns-{len(fvals)}fs_{n_levels}ls-{args.iterations}nreps'
    elif args.rate_level:
        sim_flag = f'{input_type}-{args.characteristic_frequency // 1000}kHZ_single_cell-{syntype}_syns-{args.input_frequency}Hz_{n_levels}ls-{args.iterations}nreps'
    else:
        sim_flag = f'{input_type}-{args.characteristic_frequency // 1000}kHZ_single_cell-{syntype}_syns-{args.input_frequency}Hz_{args.input_level}dB-{args.iterations}nreps'

    if resp_type is not None:
        sim_flag += f'-{resp_type}'

    cwd = os.getcwd() # os.path.dirname(__file__)
    cachepath = os.path.join('/scratch/kedoxey', "cache")
    if 'loss' in args.hearing:
        sim_flag += f'_{args.loss_limit}loss-{loss_method}'
    if args.include_ic:
        sim_flag += '_ic'

    sim_flag += '-hf'
    cachepath += sim_flag

    print(cachepath)
    if not os.path.isdir(cachepath):
        os.mkdir(cachepath)

    print(sim_flag)

    top_dir = os.path.join('/data/scrook', 'dcnmodel_scratch', 'output', sim_flag)
    if not os.path.exists(top_dir):
        os.mkdir(top_dir)

    batch_weights = True

    if batch_weights is True:

        print('Running batch simulations')

        homeostasis_factors = [1, 2, 3]  # [0.3, 0.2, 0.1, 1, 1.5, 2, 2.5, 3]

        sgc_weight = 0.00038
        tv_weight = 0.0039
        ds_weight = 0.002228

        weight_combos = [(np.round(sgc_weight*hf,7), np.round(tv_weight/hf,7), np.round(ds_weight/hf,7)) for hf in homeostasis_factors]

        # sgc_weights = [0.00044, 0.0006, 0.0008]  # 0.0002, 0.00026, 0.000327, 0.00038 (IV), 0.00044 (III), 0.0005
        # tv_weights = [0.0003, 0.0001, 0.0006]  # 0.0006 (III), 0.0012, 0.0018, 0.0024, 0.002705, 0.003, 0.0033, 0.0036, 0.0039 (IV)
        # ds_weights = [0.001, 0.0005, 0.002228] # 0.002228 (default)

        # weight_combos = list(itertools.product(*[sgc_weights, tv_weights, ds_weights]))

        cachepath_og = cachepath
        # syn_opts = []
        for wc_i, weight_combo in enumerate(weight_combos): 

            sgc_weight = weight_combo[0]
            tv_weight = weight_combo[1]
            ds_weight = weight_combo[2]

            hear_dir = os.path.join(top_dir, f'COx{cohc}_CIx{cihc}') if 'loss' in args.hearing else top_dir
            if not os.path.exists(hear_dir):
                os.mkdir(hear_dir)

            sim_dir = os.path.join(hear_dir, f'SPx{sgc_weight}_VPx{tv_weight}_DPx{ds_weight}')
            if not os.path.exists(sim_dir):
                os.mkdir(sim_dir)

            cachepath += f'COx{cohc}_CIx{cihc}-SPx{sgc_weight}_VPx{tv_weight}_DPx{ds_weight}'
            if not os.path.isdir(cachepath):
                os.mkdir(cachepath)

            syn_opts = {'sgc': {'weight': sgc_weight},
                        'tuberculoventral': {'weight': tv_weight},
                        'dstellate': {'weight': ds_weight}}
            
            print(sim_dir)

            prot = CNSoundStim(seed=seed, characteristic_frequency=args.characteristic_frequency, 
                               hearing=args.hearing, loss_limit=args.loss_limit, cohc=cohc, cihc=cihc,
                               synapsetype=syntype, syn_opts=syn_opts, include_ic=args.include_ic, pyr_erev=erev)

            print(f'Synaptic weights: SGC->P x {sgc_weight}, V->P x {tv_weight}, D->P x {ds_weight}')
            
            metadata = run_simulation(prot, fvals, levels, args, parallel, seed, stimpar, syntype, cachepath, input_type, sim_dir)
            # syn_opts.append(syn_opt)

            metadata['stimpar'] = stimpar
            metadata['erev'] = erev
            metadata['inputs'] = {'n_freqs': len(fvals),
                                  'n_levels': n_levels}
            metadata['hearing'] = {'type': args.hearing,
                                'loss_limit': args.loss_limit}
            metadata['cf'] = args.characteristic_frequency
            metadata['syntype'] = syntype
            metadata['nresps'] = args.iterations
            metadata['homeostais_factor'] = homeostasis_factors[wc_i]
            metadata['hair_cell_impairment'] = {'Cohc': cohc,
                                                'Cihc': cihc}

            filename = os.path.join(sim_dir, f'{len(fvals)}fs_{len(levels)}dbs_{args.characteristic_frequency}cf-metadata')
            if 'loss' in args.hearing:
                filename += '-loss'
            if args.include_ic:
                filename += '-ic'

            end_t = time.time()
            elapsed_t = end_t-start_t
            hours, remainder = divmod(elapsed_t,3600)
            minutes, seconds = divmod(remainder,60)
            metadata['runtime'] = f'{int(hours)}h:{int(minutes)}m:{int(seconds)}s'

            with open(os.path.join(sim_dir, f'{filename}.yml'), 'w') as outfile:
                yaml.dump(metadata, outfile,)

            cachepath = cachepath_og
    else:

        prot = CNSoundStim(seed=seed, characteristic_frequency=args.characteristic_frequency, hearing=args.hearing, 
                           loss_limit=args.loss_limit, synapsetype=syntype, include_ic=args.include_ic, pyr_erev=erev)


        metadata = run_simulation(prot, fvals, levels, args, parallel, seed, stimpar, syntype, cachepath, top_dir)

        metadata['stimpar'] = stimpar
        metadata['erev'] = erev
        metadata['inputs'] = {'n_freqs': len(fvals),
                            'n_levels': n_levels}
        metadata['hearing'] = {'type': args.hearing,
                               'loss_limit': args.loss_limit}
        metadata['cf'] = args.characteristic_frequency
        metadata['syntype'] = syntype
        metadata['nresps'] = args.iterations
        
        filename = os.path.join(top_dir, f'{len(fvals)}fs_{len(levels)}dbs_{args.characteristic_frequency}cf-metadata')
        if 'loss' in args.hearing:
            filename += '-loss'
        if args.include_ic:
            filename += '-ic'

        end_t = time.time()
        elapsed_t = end_t-start_t
        hours, remainder = divmod(elapsed_t,3600)
        minutes, seconds = divmod(remainder,60)
        metadata['runtime'] = f'{int(hours)}h:{int(minutes)}m:{int(seconds)}s'

        with open(os.path.join(top_dir, f'{filename}.yml'), 'w') as outfile:
            yaml.dump(metadata, outfile,)



if __name__ == '__main__':
    main()
