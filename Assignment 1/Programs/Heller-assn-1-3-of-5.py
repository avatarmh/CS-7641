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
import math
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
# Exploratory data analysis to verify dataset
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
#import pylab as pl
#
# for img in range(10):
#     pl.matshow(digits.images[img], cmap=pl.cm.gray)
#     print digits.target[img]
#     filename = "digitImg" + str(img) + '.jpg'
#     plt.savefig(filename)

######################
#
# Explorating k-NN
#
# Goal: Apply the k-NN classifier to two datasets. Evaluate the models while varying hyperparameter values.
# Learn how to select the best model for each algorithm and dataset. Compare the performance of the k-NN
# algorithm on the different datasets and determine whether one type of data was more amenable to the
# algorithm than the other. Through analysis speculate or explain why.
#
# Parameters to vary for model selection:
#
#  k - number of nearest neighbors to use to estimate target to start
#
#  If time permits explore:
#
#  Learning (Split on training/CV/test) - 70%/30%, 80%/20%, 85%/15%, 90%/10% training/test set scheme.
#   Reduce overfitting by constraining minimum samples_split to 5% of sample to be fit (90%).
#
#  d(x,q) - definition of distance, e.g., Manhatten, Euclidean, Distance-Weighted, etc.
#
# Models will be evaluated and selected based on the analysis of the results of parameter variation in
# terms of bias, variance and learning curves.
#


######################
#
# Design plan
#
#
# For k-NN, we chose to examine two cross-validation methods: . The first analysis follows the holdout method where
# the data are randomly separated into 3 sets: 1) training set; 2) validation set; and 3) a test set. This will be
# compare with the KFolds method of cross validation afterwards.
#
hold_out =.4
rand = 42
X_train, X_holdout, y_train, y_holdout = cross_validation.train_test_split(
    digits.data, digits.target, test_size=hold_out, random_state=rand)
# Split holdout evenly into a cross-validation set and a final test set
X_CV, X_test, y_CV, y_test = cross_validation.train_test_split(
    X_holdout, y_holdout, test_size=.5, random_state=rand)

# Verify sample sizes

print 'samples & targets = ', digits.data.shape[0]
print 'Train_set_size=', X_train.shape[0]
print 'Validation_set_size=', X_CV.shape[0]
print 'Test_set_size=', X_test.shape[0]

# Fit the model with the training set, varying the number k-nearest neighbors
# Simultaneously calculate the mean accuracy of each model defined by k on the validation set data

# Test models for k = 1,max_k_train = 5% of training set size

max_k = int(min(math.floor(X_train.shape[0]*.05),15))

#accuracy_scores_train = []
accuracy_scores_CV = []
ks = []

for i in range(max_k):

    ks.append(i+1)
    neigh = KNeighborsClassifier(n_neighbors=i+1)
    neigh.fit(X_train,y_train)

# Predict class for each example in training set (for fixed k) and calculate corresponding accuracy
#    y_train_pred = neigh.predict(X_train)
#    accuracy_scores_train.append(accuracy_score(y_train, y_train_pred))

# Predict class for each example in validation set (for fixed k) and calculate corresponding accuracy
    y_CV_pred = neigh.predict(X_CV)
    accuracy_scores_CV.append(accuracy_score(y_CV, y_CV_pred))

# Choose the ks associated with the optimal accuracy_score over all k to select model to get accuracy on test set
k_max = np.argmax(accuracy_scores_CV) + 1

# Predict y_test using training set according to optimal k from validation
neigh = KNeighborsClassifier(n_neighbors=k_max)
neigh.fit(X_train,y_train)

# Get accuracy of optimal model defined by optimal k on training set
test_set_accuracy = accuracy_score(y_test, neigh.predict(X_test))

# Generate descriptive filename
train_pct = str(int((1.-hold_out)*100))
test_pct = str(int((hold_out)*100/2))

filename = 'kNN_' + data_set_name + '_train_' + train_pct + '_' + test_pct + '_' + test_pct

def percentFormatter(y, position):
    s = "{0:.1f}".format(y * 100)

    if matplotlib.rcParams['text.usetex']:
        return s + r'$\%$'
    else:
        return s + "%"

# red dashes, blue squares and green triangles
# plt.plot(t, t, 'r--', t, t**2, 'bs', t, t**3, 'g^')

# Accuracy_scores_CV has max_k values so add 1
fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(range(1, max_k+1), accuracy_scores_CV, '-bo')
# plt.plot(range(1,11), test_means, '-go')
# # plt.show()
# plt.scatter(range(1,11), means)
ax.set_title("k-NN Accuracy vs Complexity:\nHoldout Cross Validation")
ax.set_xlabel("k-nearest neighbors")
ax.set_ylabel("Accuracy")
ax.set_ylim(.95,1.)
ax.get_yaxis().set_major_formatter(matplotlib.ticker.FuncFormatter(percentFormatter))
plt.savefig(filename + '.jpeg', format='jpeg')


# print 'Accuracy scores based on training', accuracy_scores_train
print 'Accuracy scores based on validation', accuracy_scores_CV

print 'Optimal model: k=', k_max, "Maximum accuracy = ", test_set_accuracy

#np.savetxt(filename,'Optimal model: k=', k_max, ' and maximum accuracy = ', test_set_accuracy)
np.savetxt(filename,zip(ks,accuracy_scores_CV),delimiter=',')


# finished hold out method of cross-validation

######################
#
# KFold method for cross-validation
#

kf = cross_validation.KFold(len(digits.target), n_folds=5)
print kf

# In[99]:

means = []
for neighs in range(1, 11):
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

# red dashes, blue squares and green triangles
# plt.plot(t, t, 'r--', t, t**2, 'bs', t, t**3, 'g^')

# means has 10 values
fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(range(1, 11), means, '-ro')
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
for c_value in range(-19, 20, 2):
    accuracies = []
    for train_index, test_index in kf:
        model = SVC(C=10 ** (c_value * -1))
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
    print "Mean Accuracy for C: ", 10 ** (c_value * -1), " Average:", np.mean(accuracies)
    means.append(np.mean(accuracies))

plt.scatter([10 ** (r * -1) for r in range(-19, 20, 2)], means)

# In[102]:

means = []
for depth in range(1, 20, 2):
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

plt.scatter(range(1, 20, 2), means)


# In[ ]:
