from model_selection import grid_search
from sklearn.model_selection import train_test_split
import pandas as pd

def main():
    X = pd.read_csv('data/X.csv').to_numpy()
    y = pd.read_csv('data/y.csv').to_numpy().reshape(-1)
    param_grid = {'hidden_size': [512, 1024, 2048, 4096], 'num_core_h_layers': [2], 
                  'num_mu_h_layers': [3], 'num_log_s_h_layers': [3],
                  'lr': [9e-3, 7e-3, 5e-3, 3e-3, 1e-3], 'n_epochs': [1000], 
                  'homoscedastic_vars': [None]}
    grid_search(X, y, param_grid, 'gs_results/focus_gs_lr2_hs2.csv', cv=3)

if __name__ == "__main__":
   main()


