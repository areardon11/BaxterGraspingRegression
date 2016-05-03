#! /usr/bin/env python

import numpy as np
from scipy.stats import expon, uniform
from sklearn.linear_model import LinearRegression, SGDRegressor
from sklearn.grid_search import GridSearchCV, RandomizedSearchCV
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.tree import DecisionTreeClassifier
from sklearn.cross_validation import train_test_split
from IPython import embed

def load(n='00', force_closure=True, mounted=False):
    """
    Loads data from given file index `n`
    """
    prepend = ''
    if mounted:
        prepend = '/Volumes/Gdata/patches/'
    
    w1 = np.load(prepend+'w1_projection_window_' + n + '.npz')['arr_0']
    w2 = np.load(prepend+'w2_projection_window_' + n + '.npz')['arr_0']
    moments = np.load(prepend+'moment_arms_' + n + '.npz')['arr_0']
    if force_closure:
        fc = np.load(prepend+'force_closure_' + n + '.npz')['arr_0']
    else:
        fc = np.load(prepend+'ferrari_canny_L1_' + n + '.npz')['arr_0']
    data = np.hstack((w1,w2,moments))
    return data, fc


def split(data, labels):
    """
    Splits `data` and `labels` into a 
    training set and a test set
    """
    n = data.shape[0]
    test_idx = np.random.randint(0,n,int(n/10))
    train_idx = np.delete(np.arange(0, n),test_idx)

    train_data, train_labels = data[train_idx,:], labels[train_idx]
    test_data, test_labels = data[test_idx,:], labels[test_idx]
    return train_data, train_labels, test_data, test_labels

def sampler(names, count, force_closure=True, mounted=False):
    """
    Samples `count` number of data points from
    a list of files given by `names`
    """
    samples_per_file = int(count/len(names))
    sample_data = []
    sample_labels = []
    for name in names:
        print("Opening file: " + name)
        data, labels = load(name, force_closure, mounted)
        idx = np.random.choice(data.shape[0], samples_per_file, replace=False) #samples without replacement
        sample_data.append(data[idx, :])
        sample_labels.append(labels[idx])
    sample_data = np.vstack(sample_data)
    sample_labels = np.hstack(sample_labels)
    return sample_data, sample_labels

def create_svm(pd, pl, qd, ql):
    lsvc = LinearSVC()
    params = {'C': expon(scale=100)}
    svm = RandomizedSearchCV(lsvc, params, n_jobs=4, n_iter=10, verbose=10)
    print("Training Linear SVM Randomly")
    svm.fit(pd, pl)
    print("SVM Score: " + str(svm.score(qd, ql)))
    return svm

def create_lr(pd, pl, qd, ql):
    lr = LinearRegression()
    print("Doing Linear Regression")
    lr.fit(pd, pl)
    return lr

def sgd(pd, pl, qd, ql):
    params = {'loss':['squared_loss', 'huber', 'epsilon_insensitive',
                     'squared_epsilon_insensitive'],
                'alpha':expon(scale=1),
                'epsilon':expon(scale=1),
                'l1_ratio':uniform(),
                'penalty':[ 'l2', 'l1', 'elasticnet']}
    clf = SGDRegressor()
    #clf = RandomizedSearchCV(clf, params, n_jobs=2, n_iter=10, verbose=10)
    print("Training Linear SVM Randomly")
    clf.fit(pd, pl)
    print("Score: " + str(clf.score(qd, ql)))
    return clf

def create_lr(pd, pl, qd, ql):
    lr = LinearRegression()
    print("Doing Linear Regression")
    lr.fit(pd, pl)
    return lr

def grow_tree(pd, pl, qd, ql):
    print("Growing Tree!")
    dt = DecisionTreeClassifier()
    dt.fit(pd, pl)
    print("Decision Tree Score: " + str(dt.score(qd,ql)))
    return dt

def grow_forest(pd, pl, qd, ql, classifier=True, trees=10):
    if classifier:
        rf = RandomForestClassifier(n_estimators=trees, n_jobs=4, verbose=5)
    else:
        rf = RandomForestRegressor(n_estimators=trees, n_jobs=4, verbose=5)
    print("Growing Trees!")
    rf.fit(pd, pl)
    print("Testing Forest!")
    print(rf.score(qd, ql))
    return rf

    

def add_noise(data, labels, num=3):
    """
    Augments dataset with some gaussian noise
    """
    augmented_data = []
    augmented_labels = []
    for i in range(data.shape[0]):
        #augmented_labels.append(labels[i])
        #augmented_data.append(data[i])
        print("Adding: " + str(i))
        for __ in range(num):
            augmented_data.append(data[i] + 
                np.random.normal(0.0, 0.01, data[i].shape[0]))
            augmented_labels.append(labels[i])
    return np.vstack(augmented_data), np.hstack(augmented_labels)

if __name__ == "__main__":

    #Creates datasets to test on by sampling Jeff's data
    #Only run this when you want a fresh dataset
    names = ["0" + str(i) for i in range(10)]
    names += [str(i) for i in range(10,105)]
    data, labels = sampler(names, 300000, force_closure=True, mounted=True)
    X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.33)
    np.savez('training', data=X_train, labels=y_train)
    np.savez('test', data=X_test, labels=y_test)

    #Loads the dataset
    data = np.load('training.npz')
    pd, pl = data['data'], data['labels']
    data = np.load('test.npz')
    qd, ql = data['data'], data['labels']
    pd, pl = add_noise(pd, pl, 3)

    #Examples on how to create a new model
    #svm = create_svm(pd, pl, qd, ql)
    #dt = grow_tree(pd, pl, qd, ql)
    rf = grow_forest(pd, pl, qd, ql)
         
