"""
.. module:: RFExplain

RFExplain
*************

:Description: RFExplain

    

:Authors: bejar
    

:Version: 

:Created on: 18/05/2018 9:50 

"""

from numpy import loadtxt
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import numpy as np
import eli5
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.svm import LinearSVC, SVC
import seaborn as sns
import matplotlib.pyplot as plt

__author__ = 'bejar'

def classifierMAD(num=0):
    if num==0:
        return RandomForestClassifier(n_estimators=1000,
                                     criterion='gini',
                                     max_depth=4,
                                     min_samples_split=2,
                                     min_samples_leaf=1,
                                     min_weight_fraction_leaf=0.0,
                                     max_features=None,
                                     max_leaf_nodes=None,
                                     bootstrap=True,
                                     oob_score=False,
                                     n_jobs=1,
                                     class_weight='balanced')
    elif num == 1:
        return LogisticRegression(C=1000, penalty='l2',fit_intercept=False)
    elif num == 2:
        return LinearSVC(C=65,fit_intercept=False)
    elif num ==3:
        return RidgeClassifier(alpha=1.5, fit_intercept=False)
    elif num == 4:
        return SVC(C=10000, gamma=0.01, kernel='rbf',degree=3)
        # return SVC(C=1, gamma=0.01, kernel='poly',degree=2)



        # return SVC(C=1, gamma=0.01, kernel='poly',degree=2)

def classifier(num=0, rfnest=1000, rfdep=10, SVMC=10):
    if num==0:
        return RandomForestClassifier(n_estimators=rfnest,
                                     criterion='gini',
                                     max_depth=rfdep,
                                     min_samples_split=2,
                                     min_samples_leaf=1,
                                     min_weight_fraction_leaf=0.0,
                                     max_features=None,
                                     max_leaf_nodes=None,
                                     bootstrap=True,
                                     oob_score=False,
                                     n_jobs=1,
                                     class_weight='balanced')
    elif num == 1:
        return LogisticRegression(C=10000, penalty='l2',fit_intercept=False)
    elif num == 2:
        return LinearSVC(C=SVMC,fit_intercept=False)
    elif num ==3:
        return RidgeClassifier(alpha=2, fit_intercept=False)
    elif num == 4:
        return SVC(C=10000, gamma=0.01, kernel='rbf',degree=3)

if __name__ == "__main__":
    # data = loadtxt('idfmadrf2S.csv', delimiter=';', skiprows=1)
    #
    # X = data[:, 1:-1]
    # y = data[:, -1]

    data = loadtxt('CVIrf.csv', delimiter=';', skiprows=1)
    ncl = 6
    X = data[:, 1:-ncl]
    yv = data[:, -ncl:]
    print(yv.shape)
    nlb = 5
    y = yv[:,nlb].reshape(yv.shape[0])



    sc = MinMaxScaler()
    X= sc.fit_transform(X)
    # print(X,y)

    print(X.shape)
    explain = True
    nclassif = 1

    # corr = np.corrcoef(X.T)
    #
    # mask = np.zeros_like(corr, dtype=np.bool)
    # mask[np.triu_indices_from(mask)] = True
    #
    # # Set up the matplotlib figure
    # f, ax = plt.subplots(figsize=(11, 9))
    #
    # # Generate a custom diverging colormap
    # cmap = sns.diverging_palette(220, 10, as_cmap=True)
    #
    # # Draw the heatmap with the mask and correct aspect ratio
    # sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,
    #         square=True, linewidths=.5, cbar_kws={"shrink": .5})
    # plt.show()

    n_splits = 10
    skf = StratifiedKFold(n_splits=n_splits)
    skf.get_n_splits(X, y)

    lacc = []
    for train_index, test_index in skf.split(X, y):
        clf= classifier(nclassif)

        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        clf.fit(X_train, y_train)
        prediction = clf.predict(X_test)
        acc = accuracy_score(y_test, prediction)
        print(acc)
        lacc.append(acc)

    print('ACC mean=', np.mean(lacc))

    if explain:
        clf= classifier(nclassif)


        clf.fit(X, y)

        y_hat = clf.predict(X)
        print (classification_report(y_hat, y))
        print (confusion_matrix(y_hat, y))
        print()

        expl = eli5.explain_weights(clf)

        print(eli5.format_as_text(expl, show_feature_values=True))

        # expl = eli5.explain_prediction(clf, X[1])
        #
        # print(eli5.format_as_text(expl))
