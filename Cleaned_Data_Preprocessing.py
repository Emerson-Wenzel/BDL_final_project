import numpy as np
import pandas as pd
import copy

def shuffle_X_y(X, y):
    random_mask = np.random.permutation(len(y))
    X_rand = X.iloc[random_mask]
    y_rand = y.iloc[random_mask]
    
    return X_rand, y_rand

# Splits given X and y into X_train, y_train, X_test, and y_test
# Resets index of dataframes.
def train_test_split(X, y, train_ratio=0.8):
    X_rand, y_rand = shuffle_X_y(X, y)
    
    split_ind = int(train_ratio * len(y))
    X_train = X_rand.iloc[:split_ind].reset_index()
    y_train = y_rand.iloc[:split_ind].reset_index()
    
    X_test = X_rand.iloc[split_ind:].reset_index()
    y_test = y_rand.iloc[split_ind:].reset_index()
    
    return X_train, y_train, X_test, y_test
    
# Balances binary dataset passed in: increases number of positive labels by duplicating data with positive labels
# Shuffles the dataframes and resets the indexes.
def balance_data_by_label(X, y, target_1_0_ratio):
    mask_1 = pd.Series([y['y'] == 1][0])
    X_1 = X.loc[mask_1]
    y_1 = y.loc[mask_1] # Unnecessary: We know these are all 1's
    num_1 = sum(mask_1)
    num_0 = len(y) - num_1
    
    curr_1_0_ratio = num_1 / num_0
    # Float value
    num_dupes = float(target_1_0_ratio) / curr_1_0_ratio - 1
    
    while num_dupes > 1:
        
        # @TODO: append X_1 to X, and 1's to y
#         X = np.vstack((X, copy.deepcopy(X_1)))
#         y = np.vstack((y, np.ones((num_1, 1))))
        X = X.append(copy.deepcopy(X_1))
        y = y.append(copy.deepcopy(y_1))
        num_dupes -= 1
    
    #Adding on additional individual duplicates chosen at random from dupe_1 to meet target_1_total_ratio:
    if num_dupes > 0:
        # @TODO: append X_to_append to X, and 1's to y
#         X_to_append = copy.deepcopy(X_1[:int(num_dupes * num_1)])
#         y_to_append = np.ones((int(num_dupes * num_1), 1))
#         X = np.vstack((X, X_to_append))
#         y = np.vstack((y, y_to_append))
        X = X.append(copy.deepcopy(X_1.iloc[:int(num_dupes * num_1)]))
        y = y.append(copy.deepcopy(y_1.iloc[:int(num_dupes * num_1)]))
        
        X_rand, y_rand = shuffle_X_y(X, y)
        
    return X_rand.reset_index(), y_rand.reset_index()