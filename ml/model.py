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
import neural_netA as nna
import matplotlib.pyplot as plt

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

def grow_neural_net(pd, pl, qd, ql, use_weights=None, neural_net_model=nna, classifier=True, epsilon=.01, nin=452, n_hid=200, nout=2, batch=300):
    #featurizes the data in the proper format for the neural net
    def convert_to_feature_label(y_val):
        return np.bincount(np.array([y_val]), minlength=nout)

    pdf = np.column_stack((pd,np.ones(pd.shape[0])))
    qdf = np.column_stack((qd,np.ones(qd.shape[0])))
    if classifier:
        plf = np.zeros((pl.shape[0], nout))
        for x in range(pl.shape[0]):
            plf[x] = convert_to_feature_label(pl[x])
    else:
        nout = 1

    #prepares the neural net
    neural_net = neural_net_model.Neural_Network(n_in=nin, n_hidden=n_hid, n_out=nout)
    if use_weights == None:
        print "Growing Neural Net!"
        if classifier:
            neural_net.train(pdf, plf, neural_net.crossEntropyError, neural_net.crossEntropyPrime, batch_size=batch)
        else:
            neural_net.train(pdf, pl, neural_net.meanSquaredError, neural_net.meanSquaredPrime, batch_size=batch, epsilon=epsilon)
    else:
        neural_net.load_weights(arg_list=use_weights)

    print "Testing Neural Net!"
    if classifier:
        preds = neural_net.predict(qdf)
        error_rate, indices = nna.benchmark(preds, ql)
        print "Test accuracy = ", 1-error_rate
    else:
        preds = neural_net.forward(qdf).reshape(qdf.shape[0])
        # print ql.shape
        # print preds.shape
        # print len(np.unique(preds))
        mse = sum((preds-ql)**2)/len(ql)
        me = abs(sum(preds-ql)/len(ql))
        print "mean squared error:", mse
        print "mean error:", me
    return neural_net

def predict_neural_net(qd, use_weights, n_hid=200, nin=452, classifier=True, nout=2, neural_net_model=nna):
    #preprocess data
    qdf = np.column_stack((qd,np.ones(qd.shape[0])))
    if not classifier:
        nout = 1

    #prepare neural net and load weights
    neural_net = neural_net_model.Neural_Network(n_in=nin, n_hidden=n_hid, n_out=nout)
    neural_net.load_weights(arg_list=use_weights)
    if classifier:
        return neural_net.predict(qdf)
    return neural_net.forward(qdf).reshape(qdf.shape[0])

def add_flipped_windows(data, labels):
    """
    Augments dataset with window1, window2, and their respective moment arms swapped
    """
    data_flipped = np.column_stack((data[:,225:450], data[:,:225], data[:,453:], data[:,450:453]))
    return np.concatenate((data, data_flipped), axis=0), np.concatenate((labels, np.copy(labels)), axis=0)

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
    # names = ["0" + str(i) for i in range(10)]
    # names += [str(i) for i in range(10,105)]
    # data, labels = sampler(names, 300000, force_closure=True, mounted=True)
    # X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.33)
    # np.savez('training', data=X_train, labels=y_train)
    # np.savez('test', data=X_test, labels=y_test)
    # data, labels = sampler(names, 300000, force_closure=False, mounted=True)
    # X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.33)
    # np.savez('training_ferrari_canny', data=X_train, labels=y_train)
    # np.savez('test_ferrari_canny', data=X_test, labels=y_test)

    #Loads the force closure dataset
    # data = np.load('training.npz')
    # pd, pl = data['data'], data['labels']
    # data = np.load('test.npz')
    # qd, ql = data['data'], data['labels']

    #Loads the ferrari_canny dataset
    data = np.load('training_ferrari_canny.npz')
    pd, pl = data['data'], data['labels']
    data = np.load('test_ferrari_canny.npz')
    qd, ql = data['data'], data['labels']

    #adds flipped windows and noise
    #WARNING: makes the dataset take up a ton of space!
    # pd, pl = add_noise(pd, pl, 2)
    # pd, pl = add_flipped_windows(pd, pl)

    #substitutes magnitude of moment arm for the arm
    pd = np.column_stack((pd[:,:450], np.linalg.norm(pd[:,450:453], axis=-1), np.linalg.norm(pd[:,453:], axis=-1)))
    qd = np.column_stack((qd[:,:450], np.linalg.norm(qd[:,450:453], axis=-1), np.linalg.norm(qd[:,453:], axis=-1)))

    #Examples on how to create a new model
    #svm = create_svm(pd, pl, qd, ql)
    #dt = grow_tree(pd, pl, qd, ql)
    # rf = grow_forest(pd, pl, qd, ql)
    # nn = grow_neural_net(pd, pl, qd, ql, use_weights=["nn_weights/V1.npy", "nn_weights/W1.npy"])
    # nn = grow_neural_net(pd, pl, qd, ql, classifier=False, epsilon=.001) #this trains a regressor on ferarri_canny (convergence=.0000001)
    nn = grow_neural_net(pd, pl, qd, ql, classifier=False, use_weights=["nn_weights/V_ferrari_canny2.npy", "nn_weights/W_ferrari_canny2.npy"])
         
