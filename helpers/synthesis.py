import numpy as np

def get_sic_rejection(idd, seed, n, results_dir, num_bkg_events = -1):
    
    # if num_background_events > 0, then we do a cutoff for maxsic
    
    loc_dir = f"{results_dir}/nsig_inj{n}_seed1/"
    
    path_to_fpr = f"{loc_dir}/fpr_{idd}_{seed}.npy"
    path_to_tpr = f"{loc_dir}/tpr_{idd}_{seed}.npy"
   
    fpr = np.load(path_to_fpr)
    tpr = np.load(path_to_tpr)
    
    # get the nonzero entries of fpr
    fpr_nonzero_indices = np.where(fpr != 0)
    fpr_nonzero = fpr[fpr_nonzero_indices]
    
    tpr_nonzero = tpr[fpr_nonzero_indices]

    rejection = 1.0 / fpr_nonzero #np.divide(1.0, fpr, out=np.zeros_like(fpr), where=fpr!=0)
    sic = tpr_nonzero / np.sqrt(fpr_nonzero) #np.divide(tpr, np.sqrt(fpr), out=np.zeros_like(tpr), where=np.sqrt(fpr)!=0)


    if num_bkg_events > 0:
        eps_bkg = 1.0/((0.4**2)*num_bkg_events)
        fpr_cutoff_indices = np.where(fpr_nonzero > eps_bkg)
        maxsic = np.nanmax(sic[fpr_cutoff_indices])
    else:
        maxsic = np.nanmax(sic)
    return tpr_nonzero, sic, rejection, maxsic


def get_mean_std(loc_list):
    
    mean = np.nanmean(loc_list, axis = 0)
    std = np.nanstd(loc_list, axis = 0)
    
    return mean, std

def get_med_percentile(loc_list, lower_p = 16, upper_p = 84):
    
    med = np.nanmedian(loc_list, axis = 0)
    lower_p  = np.nanpercentile(loc_list, lower_p, axis = 0)
    upper_p = np.nanpercentile(loc_list, upper_p, axis = 0)
    
    return med, lower_p, upper_p


def select_n_events(samples, n_target_total, n_total_avail, weights = None):
    
    n_to_select = int(n_target_total*len(samples)/n_total_avail)
    
    # each of the 4 samples should have 1/4 of the total weight
    weight = float(0.25*n_target_total/n_to_select)
    indices_to_select = np.random.choice(len(samples), size = n_to_select, replace = False)
    selected_events = samples[indices_to_select]
    
    if weights is not None:
        selected_weights = weights[indices_to_select]
        weights_to_use = selected_weights*weight
        return selected_events, weights_to_use

    else:
        # create an array of weights
        selected_weights = np.ones((n_to_select, 1))
        weights_to_use = selected_weights*weight
        return selected_events, weights_to_use
    









