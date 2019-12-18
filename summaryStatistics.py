

def summaryStatistics(matrix, beta=5, printResults=False):
    tp = matrix[0,0]
    fp = matrix[0,1]
    fn = matrix[1,0]
    tn = matrix[1,1]

    precision = tp / (fp + tp)
    recall = tp / (fn + tp)

    f1 = 2*((precision*recall)/(precision+recall))

    fBeta = (1 + beta**2) * ((precision*recall)/((precision * beta**2) + recall))
    
    if printResults == True:
        print("Precision:\t{}".format(precision))
        print("Recall:\t{}".format(recall))
        print("F1:\t{}".format(f1))
        print("F-Beta (beta={}):\t{}".format(beta, fBeta))
    
    return precision, recall, f1, fBeta
    
    
def prCurve(bnn):
    from sklearn.metrics import confusion_matrix    
    X_test, y_test = generateData()
    pList = []; rList = []; f1List = []; fBList = []
    
    thresholds = np.arange(.01, .99, .05)    
    y_prob = bnn.model.forward(torch.Tensor(X_test), predict = True, threshold=False)
    
    for t in thresholds:
        y_pred = y_prob > t
        
        matrix = confusion_matrix(y_pred, y_test)
        print(matrix)
        print("")
        precision, recall, f1, fBeta = summaryStatistics(matrix, beta = 5)
        pList.append(precision)
        rList.append(recall)
        f1List.append(f1)
        fBList.append(fBeta)
        
    return pList, rList, f1List, fBList
    