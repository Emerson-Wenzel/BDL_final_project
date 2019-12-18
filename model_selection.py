import torch
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, fbeta_score, accuracy_score
import pandas as pd
import numpy as np
from bnn import BNNBayesbyBackprop
import sys


def load_data(frac_test):
    X = pd.read_csv('data/smallTrainCleaned.csv')
    y = pd.read_csv('data/y_labels.csv', header=None)
    y[y == -1] = 0
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=.80, test_size=.20)
    X_train, X_test = X_train.to_numpy(), X_test.to_numpy()
    y_train, y_test = y_train.to_numpy(), y_test.to_numpy()
    y_train = y_train.reshape(-1)
    y_test = y_test.reshape(-1) 
    return X_train, X_test, y_train, y_test

def build_hidden_layer_params(num_h_layers_all, hidden_size):
    num_core_h_layers = num_h_layers_all[0] 
    num_mu_h_layers = num_h_layers_all[1] 
    num_log_s_h_layers = num_h_layers_all[2]
    # construct hidden layer lists
    core_h_layers = [hidden_size] * num_core_h_layers
    mu_h_layers = [hidden_size] * num_mu_h_layers
    log_s_h_layers = [hidden_size] * num_log_s_h_layers
    return core_h_layers, mu_h_layers, log_s_h_layers 


def append_results_dict(results_dict, metrics, cur_params, i, cv):
    # on every round, want to update val_{}_fold_{}
    for metric in metrics.keys():
        results_dict["val_{}_fold_{}".format(metric, i)].append(metrics[metric][i])
 

    # on last round per CV cycle, update everything else
    if i == (cv - 1):
        param_list = ['hidden_size', 'num_core_h_layers', 'num_mu_h_layers',
                      'num_log_s_h_layers', 'learning_rate', 'n_epochs', 'homoscedastic_var']
        for param in param_list: 
            results_dict[param].append(cur_params[param])

        for metric in metrics.keys():
            mean = np.mean(np.array(metrics[metric]))
            std = np.std(np.array(metrics[metric]))
            results_dict['val_{}_mean'.format(metric)].append(mean)
            results_dict['val_{}_std'.format(metric)].append(std)
        
    return results_dict
    

# Takes model to train, param_grid --dict of parameters, cv--number of folds
# param_grid MUST have the fields:
#   hidden_size --> list of hidden sizes
#   num_core_h_layers, num_mu_h_layers, num_log_s_h_layers --> parallel lists of num hidden layers
#   lr          --> list of learning rates
#   n_epochs  --> list of num_epochs
#   homoscedastic_vars  --> list of homoscedastic_vars
# Returns dataframe with performance given each hyperparameter setting
def grid_search(X_train, y_train, param_grid, filename, cv=3):
    num_instances = y_train.shape[0]
    test_fold_size = int(num_instances / cv)
    # use in splitting data
    perm = np.random.permutation(num_instances) 
    # initialize results_dict
    results_dict = {'hidden_size':[], 'num_core_h_layers':[],
                    'num_mu_h_layers':[], 'num_log_s_h_layers':[],
                    'learning_rate':[], 'n_epochs':[],
                    'homoscedastic_var':[]}
    for metric in ['prec', 'rec', 'f_beta', 'acc']:
        for i in range(cv):
            results_dict['val_{}_fold_{}'.format(metric, i)] = []
        results_dict['val_{}_mean'.format(metric)] = []
        results_dict['val_{}_std'.format(metric)] = []

    num_h_layers_all_list = list(zip(param_grid['num_core_h_layers'],
                                param_grid['num_mu_h_layers'],
                                param_grid['num_log_s_h_layers']))

    num_hidden_size = len(param_grid['hidden_size'])
    h_layers_all_cnt = len(list(num_h_layers_all_list))
    num_lr = len(param_grid['lr'])
    num_n_epochs = len(param_grid['n_epochs'])
    num_total_trains = num_hidden_size * h_layers_all_cnt * num_lr * num_n_epochs * cv

    
    count = 1
    # hidden_size
    for hidden_size in param_grid['hidden_size']:
        # arch
        for num_h_layers_all in num_h_layers_all_list:
            (core_h_layers, mu_h_layers, log_s_h_layers) = build_hidden_layer_params(num_h_layers_all, hidden_size) 
            # learning rate
            for lr in param_grid['lr']:

                # num_epochs
                for n_epochs in param_grid['n_epochs']:

                    for homoscedastic_var in param_grid['homoscedastic_vars']:

                        # initialize metric lists for cv
                        metrics = {'prec':[], 'rec':[], 'f_beta':[], 'acc': []}
                        for i in range(cv):
                            print('{} / {} total trains\t {}% done'.format(count, num_total_trains, int((count/num_total_trains) * 100)))
                            start = i * test_fold_size
                            if i == (cv - 1):
                                # if not even division, last split goes all the way to end
                                end = None
                            else:
                                end = (i+1) * test_fold_size
                        
                            val_mask = np.full(shape=(num_instances,), fill_value=False)
                            val_mask[start:end] = True
                            val_ind = perm[val_mask]
                            train_ind = perm[~val_mask]

                            X_train_i, y_train_i = X_train[train_ind], y_train[train_ind]
                            X_val_i, y_val_i = X_train[val_ind], y_train[val_ind]


                            bnn = BNNBayesbyBackprop(input_dim=X_train.shape[1], 
                                                     core_hidden_layers=core_h_layers,
                                                     mu_hidden_layers=mu_h_layers,
                                                     log_s_hidden_layers=log_s_h_layers,
                                                     homoscedastic_var=homoscedastic_var,
                                                     prior_mu=0.0, prior_s=1.0,
                                                     num_MC_samples=100, 
                                                     classification=True)
                            bnn.fit(X_train_i, y_train_i, plot=False, n_epochs=n_epochs, learning_rate=lr, batch_size=1000)
                            preds = bnn.model(X_val_i, predict=True)
                            precision, recall = precision_score(y_val_i, preds), recall_score(y_val_i, preds)
                            f_beta, acc = fbeta_score(y_val_i, preds, beta=1.0), accuracy_score(y_val_i, preds)
                            metrics['prec'].append(precision)
                            metrics['rec'].append(recall)
                            metrics['f_beta'].append(f_beta)
                            metrics['acc'].append(acc)
                            cur_params = {'hidden_size': hidden_size, 'learning_rate': lr, 'n_epochs': n_epochs,
                                          'num_core_h_layers': num_h_layers_all[0], 
                                          'num_mu_h_layers': num_h_layers_all[1],
                                          'num_log_s_h_layers': num_h_layers_all[2],
                                          'homoscedastic_var': homoscedastic_var}
                            results_dict = append_results_dict(results_dict, metrics, cur_params, i, cv)
                            count += 1

    results_df = pd.DataFrame(results_dict)
    results_df.to_csv(filename)
    return results_df


