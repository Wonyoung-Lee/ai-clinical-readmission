#Code framework courtesy of Jason Brownlee
#https://machinelearningmastery.com/naive-bayes-classifier-scratch-python/

import csv
import random
import math

#Load data from csv file
def loadCsv(filename):
    lines = csv.reader(open(filename, newline=''), delimiter=',', quotechar='|')
    next(lines)
    dataset = []
    for row in lines:
        dataset.append([float(x) for x in row])
    return dataset

#Split data into training and validation portions
def splitDataset(dataset, splitRatio):
    trainSize = int(len(dataset) * splitRatio)
    trainSet = []
    copy = list(dataset)
    while len(trainSet) < trainSize:
        index = random.randrange(len(copy))
        trainSet.append(copy.pop(index))
    return [trainSet, copy]

#Group data points by class
def separateByClass(dataset):
    separated = {}
    for i in range(len(dataset)):
        vector = dataset[i]
        if (vector[-1] not in separated):
            separated[vector[-1]] = []
        separated[vector[-1]].append(vector)
    return separated

#Compute the mean across a collection of numbers
def mean(numbers):
    return sum(numbers)/float(len(numbers))

#Compute the standard deviation across a collection of numbers
def stdev(numbers):
    avg = mean(numbers)
    variance = sum([pow(x-avg,2) for x in numbers])/float(len(numbers)-1)
    return math.sqrt(variance)

#Compute summary statistics for the entire dataset
def summarize(dataset):
    summaries = [(mean(attribute), stdev(attribute)) for attribute in zip(*dataset)]
    del summaries[-1]
    return summaries

#Compute summary statistics per class
def summarizeByClass(dataset):
    separated = separateByClass(dataset)
    summaries = {}
    print (separated)
    for classValue, instances in separated.items():
        summaries[classValue] = summarize(instances)
    return summaries

#Compute per-attribute probabilities
def calculateProbability(x, mean, stdev):
    if stdev == 0:
        return 0.000001
    exponent = math.exp(-(math.pow(x-mean,2)/(2*math.pow(stdev,2))))
    return max(0.000001,(1 / (math.sqrt(2*math.pi) * stdev)) * exponent)

#Calculate global calculateProbability
#To prevent data type underflow, we use log probabilities
def calculateClassProbabilities(summaries, inputVector):
    probabilities = {}
    for classValue, classSummaries in summaries.items():
        probabilities[classValue] = 0
        for i in range(len(classSummaries)):
            mean, stdev = classSummaries[i]
            x = inputVector[i]
            probabilities[classValue] += math.log10(calculateProbability(x, mean, stdev))
    return probabilities

#Determine the most likely class label
def predict(summaries, inputVector):
    probabilities = calculateClassProbabilities(summaries, inputVector)
    bestLabel, bestProb = None, -1
    for classValue, probability in probabilities.items():
        if bestLabel is None or probability > bestProb:
            bestProb = probability
            bestLabel = classValue
    return bestLabel

#Make predictions on the unseen data
def getPredictions(summaries, testSet):
    predictions = []
    for i in range(len(testSet)):
        result = predict(summaries, testSet[i])
        predictions.append(result)
    return predictions

#Compute performance in terms of F1 scores
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
    print("F1 (Class 0): "+str((2.0*p0*r0)/(p0+r0)))
    print("F1 (Class 1): "+str((2.0*p1*r1)/(p1+r1)))

def main():
    random.seed(13)
    filename = 'readmission.csv'
    splitRatio = 0.80
    dataset = loadCsv(filename)
    trainingSet, valSet = splitDataset(dataset, splitRatio)
    print('Split '+str(len(dataset))+' rows into train = '+str(len(trainingSet))+' and test = '+str(len(valSet))+' rows.\n')
    # prepare model
    summaries = summarizeByClass(trainingSet)
    # test model
    predictions = getPredictions(summaries, valSet)
    evaluate(valSet, predictions)

main()
