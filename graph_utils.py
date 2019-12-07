
import matplotlib.pyplot as plt


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
