from model_selection import grid_search
import pandas as pd
from Cleaned_Data_Preprocessing import balance_data_by_label, train_test_split

def main():
    X = pd.read_csv('data/X.csv')
    y = pd.read_csv('data/y.csv')
    X_train, y_train, _, _ = train_test_split(X, y)
    X_train, y_train = balance_data_by_label(X_train, y_train, 0.4)

    param_grid = {'hidden_size': [16], 'num_core_h_layers': [2], 
                  'num_mu_h_layers': [3], 'num_log_s_h_layers': [3],
                  'lr': [5e-4], 'n_epochs': [3], 
                  'homoscedastic_vars': [None]}
    grid_search(X_train, y_train, param_grid, 'gs_results/gs_focus_TEST.csv', cv=2)

if __name__ == "__main__":
   main()

