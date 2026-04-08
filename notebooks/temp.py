cf = ['Below CF', 'CF', 'Above CF']

chcs = {'Cohc': [[0.5, 1.0], [0.25, 1.0]],
        'Cihc': [[1.0, 0.5], [1.0, 0.25]]}

fig, axs = plt.subplots(1,2,figsize=(12,6), sharex=True, sharey=True)
axs = axs.ravel()

for hc_i, chc in enumerate(['Cohc', 'Cihc']):

    for cohc, cihc in chcs[chc]:

        hear_dir = f'COx{cohc}_CIx{cihc}'

        for hf_i, hf in enumerate(homeostasis_factors):

            weight_combo = weight_combos[hf_i]
            sgc_w = weight_combo[0]
            tv_w = weight_combo[1]
            ds_w = weight_combo[2]

            # below_cf_sim_dir = os.path.join(below_cf_dir, hear_dir, f'SPx{sgc_w}_VPx{tv_w}_DPx{ds_w}')
            cf_sim_dir = os.path.join(cf_dir, hear_dir, f'SPx{sgc_w}_VPx{tv_w}_DPx{ds_w}')
            # above_cf_sim_dir = os.path.join(above_cf_dir, hear_dir, f'SPx{sgc_w}_VPx{tv_w}_DPx{ds_w}')

            for i, freq_dir in enumerate([cf_sim_dir]):
                i = 1

                subdir_results_od = pickle.load(open(os.path.join(freq_dir, '1fs_14dbs_22000cf-results_od.pkl'), 'rb'))
                _, evoked_rates = get_evoked_firing_rates('pyramidal', subdir_results_od, stimpar['response'])
                spont_rate = get_spontaneous_firing_rates('pyramidal', subdir_results_od, stimpar['baseline'])

                subdir_data_norm = pickle.load(open(os.path.join(freq_dir, 'DATA-1fs_14dbs_22000cf-rate_level_curve-loss.pkl'),'rb'))
                subdir_freq = subdir_data_norm['freqs'][0]
                subdir_levels = subdir_data_norm['levels']

                # subdir_levels.insert(0,0)
                # evoked_rates = {**{(subdir_freq, 0): spont_rate}, **evoked_rates}

                # if (cohc == 1.0) and (cihc == 1.0):
                #     axs[hc_i].plot(subdir_levels, list(evoked_rates.values()), label=f'Cohc={cohc}, Cihc={cihc}', color='k')
                # else:
                c_val = cohc if 'ohc' in chc else cihc
                axs[hc_i].plot(subdir_levels, list(evoked_rates.values()), label=c_val)

                axs[hc_i].set_xlabel('Sound Level (dB SPL)')
                axs[hc_i].set_title(f'{cf[i]} - {subdir_freq//1000} kHz')

    axs[hc_i].plot(subdir_levels, list(cf_normal_rates.values()), label='Normal', color='k')
    axs[hc_i].legend(loc='upper left', title=chc) #, bbox_to_anchor=[1.61,1.01])

fig.suptitle(f'HF = 1')
fig.tight_layout()