#!/usr/bin/env python
# coding: utf-8

#Basic functions
def sigma(theList):
    sum = 0
    n = len(theList)
    for i in range(n):
        xi = theList[i]
        sum += xi
    return sum

def press (l1,l2):
    sum = 0
    n1 = len(l1)
    n2 = len(l2)
    if n1 == n2:
        for i in range (n1):
            sum += ((l1[i] - l2[i])**2)
        return sum
    else:
        print ("list error --> ", n1, " <> ", n2)

def press_root (l1,l2):
    from numpy import abs
    sum = 0
    n1 = len(l1)
    n2 = len(l2)
    if n1 == n2:
        for i in range (n1):
            sum += abs(l1[i] - l2[i])
        return sum
    else:
        print ("list error --> ", n1, " <> ", n2)

def press_m (l1,N):
    from numpy import mean
    sum = 0
    n1 = len(l1)
    for i in range (n1):
        sum += ((l1[i] - mean(N))**2)
    return sum

def q2r2 (obs, pred):
    from numpy import mean
    n1=press(obs, pred)
    n2 = press_m(obs, mean(obs))
    q2 = 1 - (n1/n2)
    return q2
def r2test (obs, pred, train):
    n1 = press(obs, pred)
    n2 = press_m(obs, train)
    if n2 != 0:
        r2test = 1 - (n1 / n2)
    else:
        r2test = 0
    return r2test

def RMSEP_CV_C (obs,pred, k=0):
    #root mean square of error prediction (external test set)
    #root mean square of error of cross validation (training set) or SEDP (standard deviation of error of prediction)
    from numpy import square
    n1 = press (obs, pred)
    n2 = len (obs)
    if k == 0:
        RMSE = square((n1 / n2))
        MSE = ((n1/n2))
    else:
        # root mean square of error of calibration (training set R2) k--> number of descriptors (or components) or SEE (Standard error of estimate)
        RMSE = square((n1/(n2-k-1)))
        MSE = ((n1/(n2-k-1)))
    return RMSE

def F (obs, pred, k):
    # for train set
    from numpy import mean
    Press = press_m (pred, mean(obs))
    n2 = RMSEP_CV_C (obs,pred, k=k)
    n1 = Press/k
    f = n1/n2
    return f

def analyse(ytrain, y_pred_train, ytest, y_pred_test, ycv1, ycv2, k):#k = number of predictors The CV needs another formula for kfold and just works for LOO
    r2= q2r2(ytrain, y_pred_train)
    print (len(ytrain))
    print(k)
    e1 = ((len(ytrain))-1)/0.9999999999
    e2 = ((len(ytrain)) - k - 1)
    r2A= 1-((e1/e2)*(1-r2))
    q2 = q2r2(ycv1, ycv2)
    R2test = r2test(ytest, y_pred_test, ytrain)
    f = F (ytrain, y_pred_train, k)
    RMSE_train = RMSEP_CV_C(ytrain, y_pred_train)
    MAE_train = press_root(ytrain, y_pred_train)/len(ytrain)
    RMSE_CV = RMSEP_CV_C(ycv1, ycv2)
    MAE_CV = press_root(ycv1, ycv2) / len(ycv2)
    RMSE_test = RMSEP_CV_C(ytest, y_pred_test)
    MAE_test = press_root(ytest, y_pred_test) / len(ytest)
    result = {"R2":r2, "R2_Adj": r2A, "R2_test":R2test, "F": f, "q2": q2, "RMSE_train": RMSE_train, "MAE_train": MAE_train, "RMSE_test": RMSE_test, "MAE_test": MAE_test, "RMSE_CV": RMSE_CV, "MAE_CV": MAE_CV}
    return result

def makecolumn(twodlist2, c):
    import numpy
    column = twodlist(c, 0)
    twodlist2 = numpy.array(twodlist2)
    for j in range(0, c):
        column[j] += list(twodlist2[:, j])
    return column

def twodlist(m, n):
    twod_list = []
    for i in range(0, m):
        new = []
        for j in range(0, n):
            new.append("none")
        twod_list.append(new)
    return twod_list

def log(s,DEBUG):
    if DEBUG:
        print (s)
  

