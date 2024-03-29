import numpy as np
from helpers.nn import *

from sklearn.utils import class_weight, shuffle
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split, KFold

from sklearn.metrics import roc_auc_score, roc_curve




def discriminate_datasets_weighted(dir_to_save, idd, 
                                        train_samp_1, train_samp_2, 
                                        weights_samp_1, weights_samp_2,
                                        test_samp_1, test_samp_2, 
                                        n_features, n_epochs, 
                                        batch_size, lr, patience, device, 
                                        update_epochs = 1, early_stop = True, visualize = True, seed = None):
    
    
    # save the best model
    val_loss_to_beat = 10000
    best_epoch = -1
    
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)
                    
    dense_net = NeuralNet(input_shape = n_features)
    criterion = F.binary_cross_entropy #nn.BCELoss()
    optimizer = torch.optim.Adam(dense_net.parameters(), lr=lr)

    dense_net.to(device)
    
    if early_stop:
        early_stopping = EarlyStopping(patience=patience)
    
    # transformed SIM has label 0, DAT has label 1
    # make the input and output data
    nn_train_data = np.concatenate((train_samp_1, train_samp_2))
    nn_train_labs = np.concatenate((torch.zeros((train_samp_1.shape[0], 1)), torch.ones((train_samp_2.shape[0],1))))
    nn_train_weights =  np.concatenate((weights_samp_1, weights_samp_2))
    
    print(f"Sample 1 shape: {train_samp_1.shape}")
    print(f"Sample 2 shape: {train_samp_2.shape}")
          
    # get CLASS weights
    class_weights = class_weight.compute_class_weight('balanced', np.unique(nn_train_labs.reshape(-1)), nn_train_labs.reshape(-1))
    class_weights = dict(enumerate(class_weights))
    print(class_weights)

    # train-test split
    val_size = 0.2
    
    # train-val split
    X_train, X_val, w_train, w_val, y_train, y_val = train_test_split(nn_train_data, nn_train_weights, nn_train_labs, test_size=val_size)
    
    # if no test data provided, use the val data
    if (test_samp_1 is None) or (test_samp_2 is None):
        print("Using val data as test data...")
        X_test = X_val
        y_test = y_val
    else:
        nn_test_data = np.concatenate((test_samp_1, test_samp_2))
        nn_test_labs = np.concatenate((torch.zeros((test_samp_1.shape[0], 1)), torch.ones((test_samp_2.shape[0],1))))
        # shuffle the data
        nn_train_data, nn_train_labs = shuffle(nn_train_data, nn_train_labs)
        X_test, y_test = shuffle(nn_test_data, nn_test_labs)
    
    print("Train data, labels shape:", X_train.shape, y_train.shape)
    print("Val data, labels shape:", X_val.shape, y_val.shape)
    print("Test data, labels  shape:", X_test.shape, y_test.shape)
    
    # preprocess the data
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)
    
    # send to device
    X_train = np_to_torch(X_train, device)
    X_val = np_to_torch(X_val, device)
    X_test = np_to_torch(X_test, device)
    y_train = np_to_torch(y_train, device)
    y_val = np_to_torch(y_val, device)
    w_train = np_to_torch(w_train, device)
    w_val = np_to_torch(w_val, device)

    epochs, epochs_val = [], []
    losses, losses_val = [], []

    for epoch in tqdm(range(n_epochs)):
        # batching
        indices_list = torch.split( torch.randperm( X_train.shape[0] ), batch_size )
        # initialise lists to store batch stats
        losses_batch_per_e = []
        
        for i, indices in enumerate( indices_list ): # going through the batches
            # calculate the loss, backpropagate
            optimizer.zero_grad()
            
            batch_data = X_train[indices]
            batch_labels = y_train[indices]
            batch_weights_salad = w_train[indices]
            
            # get the weights
            batch_weights_class = (torch.ones(batch_labels.shape, device=device)
                        - batch_labels)*class_weights[0] \
                        + batch_labels*class_weights[1]
            
            batch_weights = batch_weights_salad*batch_weights_class


            loss = criterion(dense_net(batch_data), batch_labels, weight = batch_weights)
            
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            
            losses_batch_per_e.append(loss.detach().cpu().numpy())

        epochs.append(epoch)
        losses.append(np.mean(losses_batch_per_e))
        
        # validation
        if epoch % update_epochs == 0:
            
            with torch.no_grad():
                # batching
                val_indices_list = torch.split( torch.randperm( X_val.shape[0] ), batch_size )
                # initialise lists to store batch stats
                val_losses_batch_per_e = []

                for i, indices in enumerate( val_indices_list ): # going through the batches
                    # calculate the loss, backpropagate
                    optimizer.zero_grad()
                    
                    batch_data = X_val[indices]
                    batch_labels = y_val[indices]
                    batch_weights_salad = w_val[indices]

                    # get the weights
                    batch_weights_class = (torch.ones(batch_labels.shape, device=device)
                                - batch_labels)*class_weights[0] \
                                + batch_labels*class_weights[1]
                    
                    batch_weights = batch_weights_salad*batch_weights_class

                    
                    val_loss = criterion(dense_net(batch_data), batch_labels, weight = batch_weights) 

                    val_losses_batch_per_e.append(val_loss.detach().cpu().numpy())

            epochs_val.append(epoch)
            losses_val.append(np.mean(val_losses_batch_per_e))
            
            # see if the model has the best val loss
            if np.mean(val_losses_batch_per_e) < val_loss_to_beat:
                val_loss_to_beat = np.mean(val_losses_batch_per_e)
                # save the model
                model_path = f"{dir_to_save}/.{idd}_best_model.pt"
                torch.save(dense_net, model_path)
                best_epoch = epoch
                
            if early_stop:
                early_stopping(np.mean(val_losses_batch_per_e))
                
        if early_stopping.early_stop:
            break

    print("Done training!")
    if visualize:
        fig, ax = plt.subplots(1, 1, figsize=(7, 5))
        ax.plot(epochs, losses)
        ax.plot(epochs_val, losses_val, label = "val")
        ax.legend()
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.set_title(idd)
        fname = f"{dir_to_save}/losses_{idd}"
        fig.savefig(fname)

    # evaluate
    # load in the model with the best val loss
    
    print(f"Loading in best model for {model_path}, val loss {val_loss_to_beat} from epoch {best_epoch}")
    dense_net_eval = torch.load(model_path)
    dense_net_eval.eval()

    # since we're not training, we don't need to calculate the gradients for our outputs
    with torch.no_grad():
        outputs = dense_net_eval(X_test).detach().cpu().numpy()
        predicted = np.round(outputs)

        # calculate auc 
        auc = roc_auc_score(y_test, outputs)
        fpr, tpr, _ = roc_curve(y_test, outputs)

    if visualize:
        fig, ax = plt.subplots(1, 1, figsize=(7, 5))
        ax.plot(fpr, tpr)
        ax.set_xlabel("FPR")
        ax.set_ylabel("TPR")
        ax.set_title("ROC: " + str(auc))
        fname = f"{dir_to_save}/roc_{idd}"
        fig.savefig(fname)
        
    np.save(f"{dir_to_save}/fpr_{idd}", fpr)
    np.save(f"{dir_to_save}/tpr_{idd}", tpr)
        
    if auc < 0.5:
        auc = 1.0 - auc
    
    return auc, outputs


def discriminate_for_scatter_kfold(results_dir, idd, train_samp_1, train_samp_2, weights_samp_1, weights_samp_2, test_samp_1, test_samp_2, n_features, n_epochs, batch_size, lr, patience, device, early_stop = True, visualize = True, seed = None, k_folds = 5):
    
    if seed is not None:
        print(f"Using seed {seed}...")
        torch.manual_seed(seed)
        np.random.seed(seed)
    
    
    # transformed SIM has label 0, DAT has label 1
    # make the input and output data
    X_train = np.concatenate((train_samp_1, train_samp_2))
    y_train = np.concatenate((torch.zeros((train_samp_1.shape[0], 1)), torch.ones((train_samp_2.shape[0],1))))    
    w_train = np.concatenate((weights_samp_1, weights_samp_2))
    
    # get weights in case we're oversampling
    class_weights = class_weight.compute_class_weight('balanced', np.unique(y_train.reshape(-1)), y_train.reshape(-1))
    class_weights = dict(enumerate(class_weights))   
    
    X_test = np.concatenate((test_samp_1, test_samp_2))
    y_test = np.concatenate((torch.zeros((test_samp_1.shape[0], 1)), torch.ones((test_samp_2.shape[0],1))))
    
    print("Train data, labels shape:", X_train.shape, y_train.shape)
    print("Test data, labels  shape:", X_test.shape, y_test.shape)
    
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # send to device
    X_train = np_to_torch(X_train, device)
    X_test = np_to_torch(X_test, device)
    y_train = np_to_torch(y_train, device)
    w_train = np_to_torch(w_train, device)
    
    # Define the K-fold Cross Validator
    kfold = KFold(n_splits=k_folds, shuffle=True)
    fold_best_val_losses = []
    
    # K-fold Cross Validation model evaluation
    for fold, (train_ids, val_ids) in enumerate(kfold.split(X_train)):     
    
        # Print
        print(f'FOLD {fold}')
        print('--------------------------------')

        X_train_fold = X_train[train_ids]
        y_train_fold = y_train[train_ids]
        w_train_fold = w_train[train_ids] 
        
        X_val_fold = X_train[val_ids]
        y_val_fold = y_train[val_ids]
        w_val_fold = w_train[val_ids] 
                
        train_set = torch.utils.data.TensorDataset(X_train_fold, y_train_fold, w_train_fold)
        val_set = torch.utils.data.TensorDataset(X_val_fold, y_val_fold, w_val_fold)
        train_loader = torch.utils.data.DataLoader(train_set, batch_size = batch_size, shuffle = True)
        val_loader = torch.utils.data.DataLoader(val_set, batch_size = batch_size, shuffle = True)
        
        # initialze the network
        dense_net = NeuralNet(input_shape = n_features)
        criterion = F.binary_cross_entropy 
        optimizer = torch.optim.Adam(dense_net.parameters(), lr=lr)
        dense_net.to(device)
        
        if early_stop:
            early_stopping = EarlyStopping(patience=patience)
        
         # save the best model
        val_loss_to_beat = 10000
        best_epoch = -1

        epochs, losses, losses_val = [], [], []

        for epoch in tqdm(range(n_epochs)):
            losses_batch_per_e = []
            # batching    
            for batch_index, (batch_data, batch_labels, batch_salad_weights) in enumerate(train_loader):

                # calculate the loss, backpropagate
                optimizer.zero_grad()

                # get the weights
                batch_weights_class = (torch.ones(batch_labels.shape, device=device)
                            - batch_labels)*class_weights[0] \
                            + batch_labels*class_weights[1]
                
                batch_weights = batch_weights_class*batch_salad_weights
         
                
                loss = criterion(dense_net(batch_data), batch_labels, weight = batch_weights)

                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

                losses_batch_per_e.append(loss.detach().cpu().numpy())

            epochs.append(epoch)
            losses.append(np.mean(losses_batch_per_e))

            # validation
            with torch.no_grad():
                val_losses_batch_per_e = []
                
                for batch_index, (batch_data, batch_labels, batch_salad_weights) in enumerate(val_loader):
                    # calculate the loss, backpropagate
                    optimizer.zero_grad()

                    # get the weights
                    batch_weights_class = (torch.ones(batch_labels.shape, device=device)
                                - batch_labels)*class_weights[0] \
                                + batch_labels*class_weights[1]
                    
                    batch_weights = batch_weights_class*batch_salad_weights

                    val_loss = criterion(dense_net(batch_data), batch_labels, weight = batch_weights) 
                    val_losses_batch_per_e.append(val_loss.detach().cpu().numpy())

                losses_val.append(np.mean(val_losses_batch_per_e))

                # see if the model has the best val loss
                if np.mean(val_losses_batch_per_e) < val_loss_to_beat:
                    val_loss_to_beat = np.mean(val_losses_batch_per_e)
                    # save the modelnp.rando
                    model_path = f"{results_dir}/.{idd}_fold{fold}.pt"
                    torch.save(dense_net, model_path)
                    best_epoch = epoch

                if early_stop:
                    early_stopping(np.mean(val_losses_batch_per_e))

            if early_stopping.early_stop:
                break
        
        print(f"Done training fold {fold}. Best val loss {val_loss_to_beat} at epoch {best_epoch}")
        """
        if visualize:
            fig, ax = plt.subplots(1, 1, figsize=(7, 5))
            ax.plot(epochs, losses)
            ax.plot(epochs, losses_val, label = "val")
            ax.legend()
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Loss")
            ax.set_title(f"{idd}_fold{fold}")
        """

        # evaluate
        fold_best_val_losses.append(val_loss_to_beat)
        print(fold_best_val_losses)
            
    
    # load in the model / fold with the best val loss 
    best_model_index = np.argmin(fold_best_val_losses)
    best_model_path = f"{results_dir}/.{idd}_fold{best_model_index}.pt"
    print(f"Loading in best model for {best_model_path}, val loss {np.min(fold_best_val_losses)} from fold {best_model_index}")
    
    dense_net_eval = torch.load(best_model_path)
    dense_net_eval.eval()
    # since we're not training, we don't need to calculate the gradients for our outputs
    with torch.no_grad():
        outputs = dense_net_eval(X_test).detach().cpu().numpy()
        predicted = np.round(outputs)

        # calculate auc 
        auc = roc_auc_score(y_test, outputs)
        fpr, tpr, _ = roc_curve(y_test, outputs)

    if visualize:
        fig, ax = plt.subplots(1, 1, figsize=(7, 5))
        ax.plot(fpr, tpr)
        ax.set_xlabel("FPR")
        ax.set_ylabel("TPR")
        ax.set_title("ROC: " + str(auc))
        fname = f"{results_dir}/roc_{idd}"
        fig.savefig(fname)
        
    np.save(f"{results_dir}/fpr_{idd}", fpr)
    np.save(f"{results_dir}/tpr_{idd}", tpr)
        
    if auc < 0.5:
        auc = 1.0 - auc
    
    return auc, outputs



"""
multiclass helpers
"""

def find_accuracy(pred, label, individual=False):
    overall = (np.argmax(pred, axis=-1) == label).astype(float) #.mean()
    if individual:
        cathode = (np.argmax(pred[label==0], axis=-1) == 0).astype(float)
        curtains = (np.argmax(pred[label==1], axis=-1) == 1).astype(float)
        feta = (np.argmax(pred[label==2], axis=-1) == 2).astype(float)
        salad = (np.argmax(pred[label==3], axis=-1) == 3).astype(float)
        return overall, (cathode, curtains, feta, salad)
    else:
        return overall


def train_multiclass_model(model, dataloader, optimizer, criterion):
    model.train()
    train_loss = []
    train_weights = []
    for i, batch in enumerate(dataloader):
        optimizer.zero_grad()
            
        batch_data = batch[0][:, :5]
        batch_weight = batch[0][:, 6]
        batch_labels = batch[0][:, -1].to(torch.long)
            
        loss = batch_weight*criterion(model(batch_data), batch_labels)
        train_loss.append(loss.tolist())
        train_weights.append(batch_weight.tolist())
        
        #loss = loss.sum()/batch_weight.sum()
        loss = loss.mean()
        
        loss.backward()
        optimizer.step()
        #if i % 40 == 0:
            #print(f"step {i:4d} / {len(dataloader)}; loss {loss.item():.4f}")
    train_loss = np.array(train_loss).flatten()
    train_weights = np.array(train_weights).flatten()
    #return train_loss.sum()/train_weights.sum()
    return train_loss.mean()

@torch.no_grad()
def evaluate(model, dataloader, criterion):
    model.eval()
    eval_loss = []
    eval_weights = []
    acc = []
    acc_individual = {'cathode': [], 'curtains': [], 'feta': [], 'salad': []}
    for batch in dataloader:            
        batch_data = batch[0][:, :5]
        batch_weight = batch[0][:, 6]
        batch_labels = batch[0][:, -1].to(torch.long)
        pred=model(batch_data)
        loss = batch_weight*criterion(pred, batch_labels)
        eval_loss.append(loss.tolist())
        eval_weights.append(batch_weight.tolist())
        local_acc, local_acc_individual = find_accuracy(np.exp(pred.cpu().numpy()), batch_labels.cpu().numpy(), individual=True)
        acc.append(local_acc)
        acc_individual['cathode'].append(local_acc_individual[0])
        acc_individual['curtains'].append(local_acc_individual[1])
        acc_individual['feta'].append(local_acc_individual[2])
        acc_individual['salad'].append(local_acc_individual[3])
    eval_loss = np.array(eval_loss).flatten()
    eval_weights = np.array(eval_weights).flatten()
    acc = np.array(acc).flatten()
    for key in acc_individual:
        acc_individual[key] = np.concatenate([*acc_individual[key]]).flatten()
    #print(f"Evaluation loss {eval_loss.sum()/eval_weights.sum()}, accuracy {acc.mean()}, acc CATHODE: {acc_individual['cathode'].mean():.3f}, acc CURTAINS: {acc_individual['curtains'].mean():.3f}, acc FETA: {acc_individual['feta'].mean():.3f}, acc SALAD: {acc_individual['salad'].mean():.3f}")
    #return eval_loss.sum()/eval_weights.sum(), acc
    print(f"Evaluation loss {eval_loss.mean()}, accuracy {acc.mean()}, acc CATHODE: {acc_individual['cathode'].mean():.3f}, acc CURTAINS: {acc_individual['curtains'].mean():.3f}, acc FETA: {acc_individual['feta'].mean():.3f}, acc SALAD: {acc_individual['salad'].mean():.3f}")
    return eval_loss.mean(), acc

@torch.no_grad()
def get_prediction(model, dataloader):
    model.eval()
    preds = []
    weights = []
    for batch in dataloader:            
        batch_data = batch[0][:, :5]
        weights.append(np.array(batch[0][:, 6].tolist()))
        batch_labels = batch[0][:, -1]
        batch_preds = np.array(model(batch_data).tolist())
        
        preds.append(np.concatenate((batch_preds, batch_labels.reshape(-1, 1).tolist()), axis=1))
    weights = np.concatenate([*weights])
    preds = np.concatenate([*preds])
    preds[:, :4] = np.exp(preds[:, :4])
    return preds, weights

def save_weights(model, appendix=None):
    """ saves the model to file """
    if appendix is not None:
        file_name = f'multiclass_weights_{appendix}.pt'
    else:
        file_name = f'multiclass_weights.pt'
    torch.save({'model_state_dict': model.state_dict()}, file_name)
    print("Model saved")

def load_weights(model, device, appendix=None):
    """ loads the model from file """
    if appendix is not None:
        file_name = f'multiclass_weights_{appendix}.pt'
    else:
        file_name = f'multiclass_weights.pt'
    checkpoint = torch.load(file_name, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    print(f"Model {file_name} loaded")
    
def log_posterior(preds, weights=None):
    """ log posterior scores = sum_{x in geant} log p(C|x) where C = CATHODE, CURTAINS, FETA, SALAD """
    if weights is None:
        return np.log(preds).sum(axis=0)
    else:
        return (weights*np.log(preds)).sum(axis=0)


