# Author: Miriam Heller <mheller8@gatech.edu>
# CS 7641, Spring 2016 GA Tech OMSCS

# Application to the sk-learns's built in 0-9 digit dataset
from sklearn.neighbors import KNeighborsClassifier

# Standard scientific Python imports
import numpy as np
import matplotlib.pyplot as plt
get_ipython().magic(u'matplotlib inline')
from sklearn import cross_validation
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

# Import digits dataset
from sklearn.datasets import load_digits


# Explorating k-NN
#
# Goal: Apply the k-NN classifier to two datasets. Evaluate the models while varying hyperparameter values.
# Learn how to select the best model for each algorithm and dataset. Compare the performance of the k-NN
# algorithm on the different datasets and determine whether one type of data was more amenable to the
# algorithm than the other. Through analysis speculate or explain why.
#
# Dataset 1 = the sci-kit subset of the MNIST database of handwritten digits 0-9. This subset consists of
# 1797 samples with each digit depicted/store as 8x8 16 grey-scale # images. Thus each sample has 64
# features.
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
#pl.matshow(digits.images[12], cmap = pl.cm.gray)
#print digits.target[12]
#pl.show()



# In[20]:

# Separate data into randomly selected training/validation and test
# sets keeping the test set for final model testing. * Since the iris
# data is so small will do a K-fold cv but will still do final
# test for some percent reserved for hold_out.*
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


# In[21]:

# Fit the model on the training set, varying k
# Simultaneously calculate the mean accuracy of each model defined by k on the validation set data

# Test models for k = 1,max_k_train = 10% of training set size

accuracy_scores_CV = []
ks = []

for i in range(10):

    ks.append(i+1)
    neigh = KNeighborsClassifier(n_neighbors=i+1)
    neigh.fit(X_train,y_train)
    
# Predict class for each example in validation set (for fixed k) and calculate corresponding accuracy
    y_CV_pred = neigh.predict(X_CV)
    accuracy_scores_CV.append(accuracy_score(y_CV, y_CV_pred))


# In[44]:

train_pct = str(int((1.-hold_out)*100))
test_pct = str(int((hold_out)*100/2))


# In[53]:

filename = "kNN_" + data_set_name + "_train_" + train_pct + "_" + test_pct + "_" + test_pct + ".csv"


# In[54]:

# Choose the ks associated with the optimal accuracy_score over all k

k_max = np.argmax(accuracy_scores_CV) + 1


# In[55]:

# Predict y_test using training set according to find optimal k from validation set

neigh = KNeighborsClassifier(n_neighbors=k_max)
neigh.fit(X_train,y_train)


# In[56]:

# Compare accuracy of optimal model defined by optimal k and training set using the test set

test_set_accuracy = accuracy_score(y_test, neigh.predict(X_test))

print filename

print ks


# print 'Accuracy scores based on training', accuracy_scores_train
print 'Accuracy scores based on validation', accuracy_scores_CV

print 'Optimal model: k=', k_max, "Maximum accuracy = ", test_set_accuracy

#np.savetxt(filename,'Optimal model: k=', k_max, ' and maximal accuracy = ', test_set_accuracy)
np.savetxt(filename,zip(ks,accuracy_scores_CV),delimiter=',')


# fini


# In[ ]:




# In[ ]:



