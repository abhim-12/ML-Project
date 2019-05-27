import pandas as pd

data = pd.read_csv('sds.csv')
labels = ['low','medium','high']

for j in data.columns[:-1]:
    mean = data[j].mean()
    data[j] = data[j].replace(0,mean)
    data[j] = pd.cut(data[j],bins=len(labels),labels=labels)
    
#Split data set for training. 70 is 70% of data set used for training
split_per = [70]

def count(data,colname,label,target):
    condition = (data[colname] == label) & (data['Outcome'] == target)
    return len(data[condition])

for i in split_per:
    predicted = []
    probabilities = {0:{},1:{}}

    #amount of data set used for training
    train_len = int((i*len(data))/100)
    #print(train_len)
    train_X = data.iloc[:train_len,:]
    #extracting as many number of rows of data as specified by train_len 
    test_X = data.iloc[train_len+1:,:-1]
    #extracting the remaining rows for testing
    test_y = data.iloc[train_len+1:,-1]

    count_0 = count(train_X,'Outcome',0,0)
    count_1 = count(train_X,'Outcome',1,1)
    
    #Find probability of occurence of 0 and 1
    prob_0 = count_0/len(train_X)
    prob_1 = count_1/len(train_X)

    for j in train_X.columns[:-1]:
        probabilities[0][j] = {}
        probabilities[1][j] = {}

        for k in labels:
            count_k_0 = count(train_X,j,k,0)
            count_k_1 = count(train_X,j,k,1)
            probabilities[0][j][k] = count_k_0 / count_0
            probabilities[1][j][k] = count_k_1 / count_1

    for row in range(0,len(test_X)):
        prod_0 = prob_0
        prod_1 = prob_1

        #Find the probabilities in the test set for each column
        for feature in test_X.columns:
            prod_0 *= probabilities[0][feature][test_X[feature].iloc[row]]
            prod_1 *= probabilities[1][feature][test_X[feature].iloc[row]]

        if prod_0 > prod_1:
             predicted.append(0)
        else:
             predicted.append(1)

    tp,tn,fp,fn = 0,0,0,0

    #Calculate Accuracy Metrics in test set
    for j in range(0,len(predicted)):
        if predicted[j] == 0:
            if test_y.iloc[j] == 0:
                 tp += 1
            else:
                fp += 1
        else:
            if test_y.iloc[j] == 1:
                tn += 1
            else:
                fn += 1

    print('Predicted Output: ')
    print(predicted)
    #for i in predicted :
    #    if(i==1):
     #       print('Has Diabetes')
     #   else:
      #       print('Does not have Diabetes')
    print()
    print('For Training Set ' + str(i) + '%')
    print('Accuracy: ',((tp+tn)/len(test_y))*100)
    print('TP:' + str(tp) +  '      ' + 'FN:'+ str(fn))
    print('FP:' + str(fp) +  '      ' + 'TN:'+ str(tn))
    print()
    print('Specificity: ',(tn/(tn+fp))*100)
    print('Sensitivity: ',(tp/(tp+fn))*100)
    print('Precision: ',(tp/(tp+fp))*100)
    print('Recall: ',(tp/(tp+fn))*100)
    print('F1: ',(tp/(tp+((tp+fn)/2)))*100)
    
