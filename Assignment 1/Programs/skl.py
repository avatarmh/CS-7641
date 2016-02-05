
# coding: utf-8

# In[89]:

from sklearn.datasets import load_digits
from sklearn.cross_validation import KFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, recall_score
import numpy as np
from sklearn.svm import SVC
from sklearn import tree
import matplotlib.pyplot as plt
import matplotlib
# get_ipython().magic(u'matplotlib inline')


# In[73]:

digits = load_digits(n_class=10)


# In[98]:

kf = KFold(len(digits.target), n_folds=5)
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
    s = "{0:.2f}".format(y*100)

    if matplotlib.rcParams['text.usetex']:
        return s + r'$\%$'
    else:
        return s + "%"


#means has 10 values
fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(range(1,11), means, '-ro')
# plt.plot(range(1,11), test_means, '-go')
# plt.show()
# plt.scatter(range(1,11), means)
ax.set_title("k-NN Complexity vs Accuracy")
ax.set_xlabel("k-nearest neighbors")
ax.set_ylabel("Accuracy")
ax.get_yaxis().set_major_formatter(matplotlib.ticker.FuncFormatter(percentFormatter))
plt.savefig("knn.jpg")

# In[113]:

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



