import csv
import random
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# Loads data from csv file
#   filename - string of the file name
def loadCsv(filename):
    lines = csv.reader(open(filename, newline=''), delimiter=',', quotechar='|')
    next(lines)
    dataset = []
    for row in lines:
        dataset.append([float(x) for x in row])
    return dataset

# Splits data into training and validation portions
#   dataset - matrix of the whole dataset
#   splitRatio - float of the ratio
def splitDataset(dataset, splitRatio):
    trainSize = int(len(dataset) * splitRatio)
    trainSet = []
    copy = list(dataset)
    while len(trainSet) < trainSize:
        index = random.randrange(len(copy))
        trainSet.append(copy.pop(index))
    return [trainSet, copy]

# Oversamples the minority data
def oversampling(trainSet):
    labels = list(zip(*[reversed(el) for el in trainSet]))[0]
    label0 = np.array([i for i in trainSet if i[-1] == 0])
    label1 = np.array([j for j in trainSet if j[-1] == 1])
    a = len(label0)
    b = len(label1)
    if a > b:
        factor = a // b
        if factor > 1:
            factor -= 1
        label1 = np.repeat(label1, repeats=factor, axis=0)
    elif a < b:
        factor = b // a
        if factor > 1:
            factor -= 1
        label0 = np.repeat(label0, repeats=factor, axis=0)
    print(label0.shape)
    print(label1.shape)
    new_set = np.concatenate([label0,label1])
    return new_set

# Initializes a np.array of weights of 0 with weights[-1] being w_0
#   trainSet - matrix of the training set
def initializeWeights(trainSet):
    weights = np.zeros(len(trainSet[0]))
    return weights

# Sigmoid function
#   a - number to plug into the sigmoid function
def sigmoid(a):
    return (1.0 / (1 + np.exp(-a)))

# Returns the prediction probability for one individual
#   features - list of the features' values for one patient
#   weights - np.array of the weights
def predict(features, weights):
    a = np.dot(np.append(features,np.array([1])), weights)
    return sigmoid(a)

# Returns a list of prediction probabilities of all individuals
def getPredictions(weights, trainSet):
    predictions = []
    for t in trainSet:
        p = predict(t[:-1],weights)
        predictions.append(p)
    return predictions

# Calculates the cross entropy loss
#   batch - (x,69) matrix of batch size of x individuals 
def crossEntropyLoss(weights, batch):
    y_true = [t[-1] for t in batch]
    y_predict = getPredictions(weights, batch)
    num_data = len(batch)
    total = 0
    for i in range(len(batch)):
        if y_true[i] == 1:
            total += -np.log(y_predict[i])
        else:
            total += -np.log(1-y_predict[i])
    loss = total / num_data
    return loss

# Updates the weights after each batch
#   predicted - (x,1) array of the predicted labels
#   labels - (x,1) array of the actual labels from the data
#   weights - (1,69) array of weights
#   lr - learning rate
def gradientDescent(batch, predicted, labels, weights, lr):
    bias_update = 0
    add_weights = np.zeros(len(weights)-1)
    for i in range(len(batch)):
        y_hat = predicted[i]
        y_i = labels[i]
        bias_update += ((y_hat-y_i)*(y_hat*(1-y_hat)))
        add_weights += ((y_hat-y_i)*(y_hat*(1-y_hat)))*batch[i][:-1]
    add_weights = np.append(add_weights,np.array([bias_update]))
    add_weights /= len(batch)
    add_weights *= lr
    weights -= add_weights
    return weights

# Splits the training set into batches of equal to near equal size
#   batchSize - int for the size of the batch
def splitIntoBatches(trainSet, batchSize):
    trainSet = np.array(trainSet)
    batches = np.array_split(trainSet, batchSize)
    return batches

# Trains the model
#   nEpoch - int for number of epochs
def train(weights, trainSet, batchSize, nEpoch, lr):
    batches = splitIntoBatches(trainSet, batchSize)
    losses = []
    epochs = []
    loss_plot = []
    for i in range(nEpoch):
        for b in batches:
            predicted = getPredictions(weights, b)
            new_weights = gradientDescent(b, predicted, b[:,-1], weights, lr)
            loss = crossEntropyLoss(new_weights, b)
            losses.append(loss)
        # Checkpoint: prints out the loss every 50 epochs
        if i % 50 == 0:
            print ("iter: " + str(i) + " loss: " + str(loss))
            epochs.append(i)
            loss_plot.append(loss)
#     plt.plot(epochs, loss_plot)
#     plt.show()
    return new_weights, losses

# Calculates the total accuracy and accuracies of each class
def accuracy(labels, predicted):
    total = 0
    total_0 = 0
    total_1 = 0
    for i in range(len(predicted)):
        p = predicted[i]
        a = labels[i][-1]
        if p == a:
            total += 1
            if p == 0 and a == 0:
                total_0 += 1
            elif p == 1 and a == 1:
                total_1 += 1
    acc = total / len(predicted)
    acc_0 = total_0 / list(list(zip(*map(reversed, labels)))[0]).count(0)
    acc_1 = total_1 / list(list(zip(*map(reversed, labels)))[0]).count(1)
    output = "Total accuracy = " + str(round(acc,4))
    output += "\nClass 0 accuracy = " + str(round(acc_0,4))
    output += "\nClass 1 accuracy = " + str(round(acc_1,4))
    print(output)

# Computes performance in terms of F1 scores
#   predictions - (x,1) array of the predicted labels
def evaluate(testSet, predictions):
    tp1 = 0.0
    fp1 = 0.0
    fn1 = 0.0
    tp0 = 0.0
    fp0 = 0.0
    fn0 = 0.0
    for i in range(len(testSet)):
        if testSet[i][-1] == 0:
            if predictions[i] == 0:
                tp0 += 1.0
            else:
                fp1 += 1.0
                fn0 += 1.0
        else:
            if predictions[i] == 1:
                tp1 += 1.0
            else:
                fp0 += 1.0
                fn1 += 1.0
    p0 = tp0/(tp0+fp0)
    p1 = tp1/(tp1+fp1)
    r0 = tp0/(tp0+fn0)
    r1 = tp1/(tp1+fn1)
    print("F1 (Class 0): " + str((2.0*p0*r0)/(p0+r0)))
    print("F1 (Class 1): " + str((2.0*p1*r1)/(p1+r1)))

# Main function
def main():
    random.seed(13)
    filename_train = 'readmission.csv'
    splitRatio = 0.80
    dataset = loadCsv(filename_train)
    
    # if the training and test sets are in the same file
#     trainingSet, valSet = splitDataset(dataset, splitRatio)
#     print('Split '+str(len(dataset)) + ' rows into train = ' + str(len(trainingSet))
#             + ' and test = ' + str(len(valSet)) + ' rows.\n')
    
    # if the training and test sets are in different files
    filename_test = 'readmission_test.csv'
    trainingSet = dataset
    valSet = loadCsv(filename_test)
    
    # PCA attempt
    pca = PCA(n_components = 50)
    trainingSet2 = [i[:-1] for i in trainingSet]
    pca.fit(trainingSet2)
    reduced_training = pca.transform(trainingSet2)
    valSet2 = [j[:-1] for j in valSet]
    reduced_val = pca.transform(valSet2)
    
    print(len(reduced_training))
    for i in range(len(reduced_training)):
        reduced_training[i][-1] = int(round(trainingSet[i][-1]))
        
    for i in range(len(reduced_val)):
        reduced_val[i][-1] = int(round(valSet[i][-1]))
    
#     plt.figure()
#     plt.plot(np.cumsum(pca.explained_variance_ratio_))
#     plt.xlabel('Number of Components')
#     plt.ylabel('Variance (%)') 
#     plt.title('Explained Variance')
#     plt.show()
#     how does the pca know what the correct labels are?

#     when doing this ^, do PCA(n_components = 68)
#     used to find the proper number of components to keep. Variance loss begins at about lower than 50 components, 
#     so we picked ~50 components for pca to minimize info loss during dimensionality reduction

    # Shuffles the oversampled data
    adjusted_training = oversampling(reduced_training)
    random.shuffle(adjusted_training)
    
    # Initializes weights
    weights = initializeWeights(adjusted_training)

    # Trains the dataset with these hyperparameters
    batch_size = 1200
    n_epoch = 500
    learning_rate = 0.0015
    finalWeights, losses = train(weights, adjusted_training, batch_size, n_epoch, learning_rate)
    print("Weights: " + str(finalWeights))

    # Gets the predictions after weights are set
    valSet_before = getPredictions(finalWeights, valSet)
    f = lambda x: 0 if x < 0.5 else 1
    valSet_after = np.array(list(map(f,valSet_before)))
    print("Predicted labels: " + str(list(valSet_after)))

    # Test model
    accuracy(valSet, valSet_after)
    evaluate(valSet, valSet_after)
    
main()