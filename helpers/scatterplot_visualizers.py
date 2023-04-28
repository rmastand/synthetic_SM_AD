import numpy as np

from sklearn.isotonic import IsotonicRegression
from sklearn.preprocessing import StandardScaler



def load_in_data(num_signal_to_inject, num_bkg, process, synth_ids, n_seed, scatterplot_dir):

    all_results_bkg = {iid:{} for iid in synth_ids}
    all_results_sig = {iid:{} for iid in synth_ids}

    
    for iid in synth_ids:
        for seed_NN in range(n_seed):

            if iid != "fullsup":
                data = np.load(f"{scatterplot_dir}/nsig_inj{num_signal_to_inject}_seed1/{iid}_results_seedNN{seed_NN}_nsig{num_signal_to_inject}.npy")
            else:
                data = np.load(f"{scatterplot_dir}/nsig_inj{num_signal_to_inject}_seed1/full_sup_results_seedNN{seed_NN}.npy")

            # currently standardizing each random seed independently. May want to revisit!
            if process == "StandardScale":
                data_stand = StandardScaler().fit_transform(data)

            elif process == "IsotonicReg":
                true_labels = np.concatenate((np.zeros((num_bkg,)), np.ones((num_sig,))))
                data_stand = IsotonicRegression().fit_transform(data, true_labels).reshape(-1, 1)
            elif process == "None":
                data_stand = data
                
            
            data_bkg = data_stand[:num_bkg]
            data_sig = data_stand[num_bkg:]

            all_results_bkg[iid][seed_NN] = data_bkg
            all_results_sig[iid][seed_NN] = data_sig
            
            
    return all_results_bkg, all_results_sig



def concatenate_scatterplot_data(all_results_bkg, all_results_sig, synth_ids, n_seed):
    
    concatenated_results_bkg = {iid:0 for iid in synth_ids}
    concatenated_results_sig = {iid:0 for iid in synth_ids}


    for iid in synth_ids:
        loc_holder_bkg = np.empty((0, 1))
        loc_holder_sig = np.empty((0, 1))

        for seed_NN in range(n_seed):
            loc_holder_bkg = np.concatenate((loc_holder_bkg, all_results_bkg[iid][seed_NN]), axis = 0)
            loc_holder_sig = np.concatenate((loc_holder_sig, all_results_sig[iid][seed_NN]), axis = 0)

        concatenated_results_bkg[iid] = loc_holder_bkg
        concatenated_results_sig[iid] = loc_holder_sig
        
    return concatenated_results_bkg, concatenated_results_sig
