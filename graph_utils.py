import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve
import torch
import numpy as np




"""
w1_1,w1_2,w2_1,w2_2,
w1_1_grad,w1_2_grad,w2_1_grad,w2_2_grad,
b_1,b_2,
b_1_grad,b_2_grad
"""


longColName = {
    'w1_1': 'mean output weight 1',
    'w1_2': 'mean output weight 2',
    'w2_1': 'variance output weight 1',
    'w2_2': 'variance output weight 2',
    'w1_1_grad': 'mean output weight 1 gradient',
    'w1_2_grad': 'mean output weight 2 gradient',
    'w2_1_grad': 'variance output weight 1 gradient',
    'w2_2_grad': 'variance output weight 2 gradient',
}



def graphCols(df, cols, ylabel, plotDim1, plotDim2):
    X = list(range(df.shape[0]))

    #fig, axs = plt.subplots(plotDim1, plotDim2)


    for i, c in enumerate(cols):
    
        plt.plot(X, df[c])
        plt.title(longColName[c])
        plt.show()
        
    
    """
        row = i // plotDim2
        col = i % plotDim2
        axs[row, col].plot(X, df[c])
        axs[row, col].title(longColName[c])
        axs[row, col].ylabel(ylabel)
        axs[row, col].xlabel("Epoch")

    plt.show()

"""

def calibrationPlot(trainedModel, y_test, X_test, title=None, n_bins=10, pred_prob_hist=False, filename=None):
    n_bins = 10
    calibration_mc_samples = 1000

    sample_pred_list = []
    for s in range(calibration_mc_samples):
        sample_pred = trainedModel.model(torch.tensor(X_test, dtype=torch.float), predict=True).detach().numpy()
        sample_pred_list.append(sample_pred)
        
    # Each row is a sample, columns are datapoints; average across columns for probability
    sample_preds = np.vstack(sample_pred_list).T
    y_prob = np.mean(sample_preds, axis=1)
    true_prob, pred_prob = calibration_curve(y_test, y_prob, n_bins=n_bins, strategy='uniform')

    if pred_prob_hist:
        fig = plt.figure(figsize=(10,5))
        ax = fig.add_subplot(121)
    else:
        fig = plt.figure(figsize=(5,5))
        ax = fig.add_subplot(111)
        

    if title:
        ax.set_title("Calibration Plot\n{}".format(title))
    else:
        ax.set_title("Calibration Plot")
    ax.set_xlabel("Predicted Probability")
    ax.set_ylabel("True Probability")
    ax.plot([i/10 for i in range(0, 11)], [i /10 for i in range(0, 11)], c='black')
    ax.plot(pred_prob, true_prob)
    
    if pred_prob_hist:
        ax = fig.add_subplot(122)
        ax.set_title("Histogram of Predicted Probabilities")
        ax.hist(y_prob)

    plt.show()
    if filename:
        plt.savefig(filename)

