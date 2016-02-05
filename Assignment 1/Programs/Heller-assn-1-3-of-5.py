# Author: Miriam Heller <mheller8@gatech.edu>
# CS 7641, Spring 2016 GA Tech OMSCS
# Asssignment 1 - due February 7, 2016

# Comparison of 5 Supervised Learning Algorithms on 2 data sets

# Import digits dataset from sk-learns's built in 0-9 digit dataset
from sklearn.datasets import load_digits

# Import sk-learn supervised learning algorithms
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn import tree
from sklearn import cross_validation
from sklearn.metrics import accuracy_score, recall_score, confusion_matrix

# Import standard scientific Python packages
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
# get ipython().magic(u'matplotlib inline')

######################
#
# Dataset 1
#
# Sci-kit subset of the MNIST database of handwritten digits 0-9. This subset consists of
# 1797 samples with each digit depicted/store as 8x8 16 grey-scale # images. Thus each sample has 64
# features.
#
# Hyperparameters to vary for model selection:
#
#     k - number of nearest neighbors to use to estimate target to start
#
#  If time permits explore:
#
#     Learning (Split on training/CV/test) - 70%/30%, 80%/20%, 85%/15%, 90%/10% training/test set scheme. Reduce
#        overfitting by constraining minimum samples_split to 5% of sample to be fit (90%).
#
#     d(x,q) - definition of distance, e.g., Manhatten, Euclidean, Distance-Weighted, etc.
#
#  Models will be evaluated and selected based on the analysis of the results of parameter variation in
#  terms of bias, variance and learning curves.
#


######################
#
#  Exploratory data analysis to verify dataset
#
# Load MNIST digit data and verify dimensions
data_set_name = "MNIST Digit Dataset"
digits = load_digits()
samples = digits.data.shape[0]
features = digits.data.shape[1]
classes = digits.target_names.shape[0]
print samples
print features
print classes
print digits.target_names
print digits.data[1:2]
print digits.target.shape[0]

# Print out examples for 0 - 9 from dataset for report
#

#pl.matshow(digits.images[12], cmap = pl.cm.gray)
#print digits.target[12]
#pl.show()







kf = cross_validation.KFold(len(digits.target), n_folds=5)
print kf


# In[99]:

means = []
for neighs in range(1,11):
    accuracies = []
    for train_index, test_index in kf:
        neigh = KNeighborsClassifier(n_neighbors=neighs)
        X_train, X_test = digits.data[train_index], digits.data[test_index]
        y_train, y_test = digits.target[train_index], digits.target[test_index]
        neigh.fit(X_train, y_train) 
        trues, preds = [], []
        for i in range(len(X_test)):
            preds.append(neigh.predict(X_test[i]))
            trues.append(y_test[i])
        acc = accuracy_score(trues, preds)
#         print "Accuracy: ", acc, "Neighbors: ", neighs
        accuracies.append(acc)
    # accuracies has 5 values
    print "Mean Accuracy for ", neighs, " neighbors: ", np.mean(accuracies)
    means.append(np.mean(accuracies))

def percentFormatter(y,position):
    s = "{0:.1f}".format(y*100)

    if matplotlib.rcParams['text.usetex']:
        return s + r'$\%$'
    else:
        return s + "%"

# red dashes, blue squares and green triangles
# plt.plot(t, t, 'r--', t, t**2, 'bs', t, t**3, 'g^')

#means has 10 values
fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(range(1,11), means, '-ro')
# plt.plot(range(1,11), test_means, '-go')
# plt.show()
# plt.scatter(range(1,11), means)
ax.set_title("k-NN Accuracy vs Complexity")
ax.set_xlabel("k-nearest neighbors")
ax.set_ylabel("Accuracy")
ax.get_yaxis().set_major_formatter(matplotlib.ticker.FuncFormatter(percentFormatter))
plt.savefig("knn.jpg")


# Exploring SVC

means = []
for c_value in range(-19,20,2):
    accuracies = []
    for train_index, test_index in kf:
        model = SVC(C=10**(c_value * -1))
        X_train, X_test = digits.data[train_index], digits.data[test_index]
        y_train, y_test = digits.target[train_index], digits.target[test_index]
        model.fit(X_train, y_train) 
        trues, preds = [], []
        for i in range(len(X_test)):
            preds.append(model.predict(X_test[i]))
            trues.append(y_test[i])
        acc = accuracy_score(trues, preds)
    #         print "Accuracy: ", acc, "Neighbors: ", neighs
        accuracies.append(acc)
    print "Mean Accuracy for C: ", 10**(c_value * -1), " Average:", np.mean(accuracies)
    means.append(np.mean(accuracies))

plt.scatter([10**(r * -1) for r in range(-19,20,2)], means)


# In[102]:

means = []
for depth in range(1,20,2):
    accuracies = []
    for train_index, test_index in kf:
        model = tree.DecisionTreeClassifier(max_depth=depth)
        X_train, X_test = digits.data[train_index], digits.data[test_index]
        y_train, y_test = digits.target[train_index], digits.target[test_index]
        model.fit(X_train, y_train) 
        trues, preds = [], []
        for i in range(len(X_train)):
            preds.append(model.predict(X_train[i]))
            trues.append(y_train[i])
        acc = accuracy_score(trues, preds)
    #         print "Accuracy: ", acc, "Neighbors: ", neighs
        accuracies.append(acc)
    print "Mean Accuracy for max depth: ", depth, " Average:", np.mean(accuracies)
    means.append(np.mean(accuracies))

plt.scatter(range(1,20,2), means)


# In[ ]:



