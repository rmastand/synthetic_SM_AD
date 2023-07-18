import numpy as np 
import torch
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
import os
from numba import cuda


from helpers.training import *
from helpers.utils import *
from helpers.nn import *

torch.set_default_dtype(torch.float64)

cuda_dev = 3
sample_size = 400000
epochs_NN =  100
batch_size_NN = 1000 
lr_NN = 0.001
patience_NN = 5 
train_networks = False
summarize_results = True

num_seed = 100
seed_start = cuda_dev

os.environ["CUDA_VISIBLE_DEVICES"]= str(cuda_dev)
device = cuda.get_current_device()
device.reset()
# set the number of threads that pytorch will use
torch.set_num_threads(2)
# set gpu device
device = torch.device( "cuda" if torch.cuda.is_available() else "cpu")
print( "Using device: " + str( device ), flush=True)


# loading and understanding the data
data_path = '/global/home/users/rrmastandrea/scaled_data_wide_07_11/'

cathode_data = np.load(f"{data_path}nsig_injected_0/cathode.npy")
curtains_data = np.load(f"{data_path}nsig_injected_0/curtains.npy")
feta_data = np.load(f"{data_path}nsig_injected_0/feta_o6.npy")
salad_data = np.load(f"{data_path}nsig_injected_0/salad.npy")
salad_weights = np.load(f"{data_path}nsig_injected_0/salad_weights.npy")

truth_data = np.load(f"{data_path}nsig_injected_0/data.npy")


"""
DATA PREPARATION
"""

np.random.seed(1618)
torch.manual_seed(1618)
# data preparation:
# select same number of events from datasets
# append weight for salad, 1 for the rest
# append label (cathode:0, curtains:1, feta: 2, salad: 3)
# split train/test/val
# make dataset

cathode_indices = np.random.choice(np.arange(len(cathode_data)), size=sample_size, replace=False)
cathode_data_cut = cathode_data[cathode_indices]
cathode_data_cut = np.concatenate((cathode_data_cut, np.ones((sample_size,1)),0.*np.ones((sample_size,1))), axis=1)

curtains_indices = np.random.choice(np.arange(len(curtains_data)), size=sample_size, replace=False)
curtains_data_cut = curtains_data[curtains_indices]
curtains_data_cut = np.concatenate((curtains_data_cut, np.ones((sample_size,1)), 1.*np.ones((sample_size,1))), axis=1)

feta_indices = np.random.choice(np.arange(len(feta_data)), size=sample_size, replace=False)
feta_data_cut = feta_data[feta_indices]
feta_data_cut = np.concatenate((feta_data_cut, np.ones((sample_size,1)), 2.*np.ones((sample_size,1))), axis=1)

salad_indices = np.random.choice(np.arange(len(salad_data)), size=sample_size, replace=False)
salad_data_cut = salad_data[salad_indices]
salad_data_cut = np.concatenate((salad_data_cut, salad_weights[salad_indices].reshape(sample_size,1), 3.*np.ones((sample_size,1))), axis=1)

truth_data_cut = np.concatenate((truth_data, 4.*np.ones((truth_data.shape[0], 1))), axis=1)

split_indices = [int(0.6*sample_size), int(0.2*sample_size), int(0.2*sample_size)]
assert np.sum(split_indices) == sample_size
split_indices = np.cumsum(split_indices)
print("Train / Test / Val split at indices ", split_indices)

train_cathode, test_cathode, val_cathode = np.split(cathode_data_cut, split_indices[:2])
train_curtains, test_curtains, val_curtains = np.split(curtains_data_cut, split_indices[:2])
train_feta, test_feta, val_feta = np.split(feta_data_cut, split_indices[:2])
train_salad, test_salad, val_salad = np.split(salad_data_cut, split_indices[:2])

train_data = np.concatenate((train_cathode, train_curtains, train_feta, train_salad))
test_data = np.concatenate((test_cathode, test_curtains, test_feta, test_salad))
val_data = np.concatenate((val_cathode, val_curtains, val_feta, val_salad))

train_dataset = TensorDataset(torch.tensor(train_data).to(device))
test_dataset = TensorDataset(torch.tensor(test_data).to(device))
val_dataset = TensorDataset(torch.tensor(val_data).to(device))
truth_dataset = TensorDataset(torch.tensor(truth_data_cut).to(device))

train_dataloader = DataLoader(train_dataset, batch_size=batch_size_NN, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size_NN, shuffle=False)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size_NN, shuffle=False)
truth_dataloader = DataLoader(truth_dataset, batch_size=batch_size_NN, shuffle=False)

"""
# Kish's effective sample size = (sum weights)**2 / sum (weights**2) => too large class weight, salad will dominate
# induced sample size: sum weights. 
for sample in [train_salad, test_salad, val_salad]:
    sum_of_square = (sample[:, 6]**2).sum()
    square_of_sum = sample[:, 6].sum()**2
    print("size of sample", sample.shape[0])
    print("sum of (weight**2)", sum_of_square)
    print("square of (weights.sum)", square_of_sum)
    print("((sum w)**2 / (sum w**2), which is Kish's effective sample size", (square_of_sum/sum_of_square))
    print("ratio of size to ((sum w)**2 / (sum w**2)", sample.shape[0] / (square_of_sum/sum_of_square))
    print("sum of |weights| and n/sum|weights|: ", sample[:, 6].sum(), sample.shape[0]/sample[:, 6].sum())
    print(" * * * ")
"""
# will use 1.021 for the subsequent analysis



# Looking at these numbers, we see that we can train a **balanced** multiclass classifier of CATHODE vs CURTAINS vs FETA vs SALAD  using 400k events of each method. We evaluate on the 120k truth events in the end. 
# 
# For training, a train / test / val split of 60/20/20 would mean 240k train, 80k test, and 80k val events. 
# 

# # balanced data

if train_networks:

    for seed in range(seed_start, num_seed, 4):
        
        print(f"On seed {seed} of {num_seed}...")

        np.random.seed(seed)
        torch.manual_seed(seed)

        """MAKE THE NETWORK"""

        dense_net = MulticlassNet(input_shape=5)
        dense_net.to(device)

        #print("model architecture ")
        #print(dense_net)
        #total_parameters = sum(p.numel() for p in dense_net.parameters() if p.requires_grad)
        #print(f"Model has {total_parameters:d} trainable parameters")

        criterion = nn.NLLLoss(reduction='none', weight=torch.tensor([1., 1., 1., 1.021]).to(device)) #, weight=torch.tensor([1., 1., 1., 9./8.]).to(device)
        optimizer = torch.optim.Adam(dense_net.parameters(), lr=lr_NN)
        name_appendix = f"multiclass_models/run{seed}_balanced_original_5_classwgt_1.021"
        early_stopping = EarlyStopping(patience = patience_NN)
        best_val_loss = 1e6

        #accuracy = []
        _, acc = evaluate(dense_net, test_dataloader, criterion)
        #accuracy.append(acc.mean())

        #train_losses = []
        #eval_losses = []

        for epoch in range(epochs_NN):
            print(f"Epoch {epoch+1} / {epochs_NN}")
            train_loss = train_multiclass_model(dense_net, train_dataloader, optimizer, criterion)
            eval_loss, acc = evaluate(dense_net, test_dataloader, criterion)
            #train_losses.append(train_loss.mean())
            #eval_losses.append(eval_loss)
            #accuracy.append(acc.mean())

            early_stopping(eval_loss.mean())
            if eval_loss.mean() < best_val_loss:
                save_weights(dense_net, appendix=name_appendix)
                best_val_loss = eval_loss.mean()
            if early_stopping.early_stop:
                print("Early stopping")
                break
            print("   - - - - -   ")
        print(3*"\n")

        

"""
LOAD IN PREDICTIONS
"""

if summarize_results:
    output_str = "log posterior of {} is {}. Argmax is at {}"
    output_str_2 = "log posterior of {} is {} +/- {}. Argmax is at {}"
    log_posterior_dict = {'CATHODE': [], 'CURTAINS': [], 'FETA': [], 'SALAD': [], 'TRUTH': []}

    preds_cathode_list = []
    preds_curtains_list = []
    preds_feta_list = []
    preds_salad_list = []
    weights_salad_list = []
    preds_truth_list = []

    for run_num in range(100):

        print(f"Run number {run_num}")
        name_appendix = f"multiclass_models/run{run_num}_balanced_original_5_classwgt_1.021"
        
        dense_net = MulticlassNet(input_shape=5)
        dense_net.to(device)
        
        load_weights(dense_net, device, appendix=name_appendix)
        preds_models, weights_models = get_prediction(dense_net, test_dataloader)
        preds_truth, _ = get_prediction(dense_net, truth_dataloader)
        preds_cathode_list.append(preds_models[preds_models[:, -1] == 0.][:, :4])
        preds_curtains_list.append(preds_models[preds_models[:, -1] == 1.][:, :4])
        preds_feta_list.append(preds_models[preds_models[:, -1] == 2.][:, :4])
        preds_salad_list.append(preds_models[preds_models[:, -1] == 3.][:, :4])
        weights_salad_list.append(weights_models[preds_models[:, -1] == 3.].reshape(-1,1))
        preds_truth_list.append(preds_truth[:, :4])

        
        print(output_str.format("CATHODE samples", 
                                log_posterior(preds_cathode_list[-1]), 
                                np.argmax(log_posterior(preds_cathode_list[-1]))))
        log_posterior_dict['CATHODE'].append(log_posterior(preds_cathode_list[-1])/len(preds_cathode_list[-1]))

        print(output_str.format("CURTAINS samples", 
                                log_posterior(preds_curtains_list[-1]), 
                                np.argmax(log_posterior(preds_curtains_list[-1]))))
        log_posterior_dict['CURTAINS'].append(log_posterior(preds_curtains_list[-1])/len(preds_curtains_list[-1]))

        print(output_str.format("FETA samples", 
                                log_posterior(preds_feta_list[-1]), 
                                np.argmax(log_posterior(preds_feta_list[-1]))))
        log_posterior_dict['FETA'].append(log_posterior(preds_feta_list[-1])/len(preds_feta_list[-1]))

        print(output_str.format("SALAD samples", 
                                log_posterior(preds_salad_list[-1], weights=weights_salad_list[-1]), 
                                np.argmax(log_posterior(preds_salad_list[-1]))))
        log_posterior_dict['SALAD'].append(log_posterior(preds_salad_list[-1])/len(preds_salad_list[-1]))

        print(output_str.format("True samples", 
                                log_posterior(preds_truth_list[-1]), 
                                np.argmax(log_posterior(preds_truth_list[-1]))))
        log_posterior_dict['TRUTH'].append(log_posterior(preds_truth_list[-1])/len(preds_truth_list[-1]))
        


    preds_cathode_list = np.array(preds_cathode_list)
    preds_curtains_list = np.array(preds_curtains_list)
    preds_feta_list = np.array(preds_feta_list)
    preds_salad_list = np.array(preds_salad_list)
    weights_salad_list = np.array(weights_salad_list)
    preds_truth_list = np.array(preds_truth_list)

    
    for key in log_posterior_dict:
        log_posterior_dict[key] = np.array(log_posterior_dict[key])

    """
    print("averaged scores: ")
    print(output_str.format("CATHODE samples", 
                            log_posterior(preds_cathode_list.mean(0)),
                            np.argmax(log_posterior(preds_cathode_list.mean(0)))))
    print(output_str.format("CURTAINS samples", 
                            log_posterior(preds_curtains_list.mean(0)), 
                            np.argmax(log_posterior(preds_curtains_list.mean(0)))))
    print(output_str.format("FETA samples", 
                            log_posterior(preds_feta_list.mean(0)), 
                            np.argmax(log_posterior(preds_feta_list.mean(0)))))
    print(output_str.format("SALAD samples", 
                            log_posterior(preds_salad_list.mean(0)), 
                            np.argmax(log_posterior(preds_salad_list.mean(0)))))
    print(output_str.format("True samples", 
                            log_posterior(preds_truth_list.mean(0)), 
                            np.argmax(log_posterior(preds_truth_list.mean(0)))))
    print("Mean and std of individual runs: ")
    """

    to_plot_median = []
    to_plot_err_lower = []
    to_plot_err_higher = []

    for method in ['CATHODE', 'CURTAINS', 'FETA', 'SALAD', 'TRUTH']:
        #print(f"Based on {len(log_posterior_dict[method])} runs.")

        #cen = log_posterior_dict[method].mean(0)
        #err = log_posterior_dict[method].std(0)
        median= np.median(log_posterior_dict[method], axis = 0) 
        percentile_16= np.percentile(log_posterior_dict[method],16, axis = 0) 
        percentile_84= np.percentile(log_posterior_dict[method],84, axis = 0) 

        #print(output_str_2.format(f"{method} samples", 
        #                          cen,
         #                         err,
        #                          np.argmax(log_posterior_dict[method].mean(0))))
  
        to_plot_median.append(median)
        to_plot_err_lower.append(percentile_16)
        to_plot_err_higher.append(percentile_84)

    to_plot_median = np.array(to_plot_median).flatten()
    to_plot_err_lower = np.array(to_plot_err_lower).flatten()
    to_plot_err_higher = np.array(to_plot_err_higher).flatten()

    with open('log_posterior_med.npy', 'wb') as f:
        np.save(f, to_plot_median)
    with open('log_posterior_err_lower.npy', 'wb') as f:
        np.save(f, to_plot_err_lower)
    with open('log_posterior_err_higher.npy', 'wb') as f:
        np.save(f, to_plot_err_higher)
        
    print("Done!")

