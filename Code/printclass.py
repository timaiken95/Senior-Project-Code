def printClassifier(prefix, score, truePositive, falseNegative, falsePositive, trueNegative):
    print(prefix + " overall accuracy: " + str(score))
    print(prefix + " true positive: " + str(truePositive))
    print(prefix + " false negative: " + str(falseNegative))
    print(prefix + " false positive: " + str(falsePositive))
    print(prefix + " true negative: " + str(trueNegative))