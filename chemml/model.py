#coding=utf-8
from modules.basic import q2r2, RMSEP_CV_C, r2test, F
from sklearn.datasets import make_regression
from sklearn.metrics import mean_squared_error
import math
from sklearn import metrics
from sklearn.model_selection import train_test_split, cross_val_score, LeaveOneOut, KFold, RepeatedKFold, ShuffleSplit
from sklearn.metrics import accuracy_score
import numpy

import itertools

import matplotlib.pyplot as plt
import matplotlib

from sklearn.metrics import confusion_matrix
import pandas

from modules.basic import analyse
#############regressors
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet
from sklearn.linear_model import Lars
from sklearn.linear_model import LassoLars
from sklearn.linear_model import OrthogonalMatchingPursuit
from sklearn.linear_model import BayesianRidge
from sklearn.linear_model import ARDRegression
from sklearn.linear_model import RANSACRegressor
from sklearn.linear_model import TheilSenRegressor
from sklearn.linear_model import HuberRegressor

from sklearn.svm import SVR
from sklearn.svm import LinearSVR
from sklearn.linear_model import SGDRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.cross_decomposition import PLSRegression
from sklearn.tree import DecisionTreeRegressor
# ensemble methods
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import BaggingRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import AdaBoostRegressor
# neural network
from sklearn.neural_network import MLPRegressor

reg = []

##########binary classifiers
from sklearn.linear_model import RidgeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import Perceptron
from sklearn.linear_model import PassiveAggressiveClassifier

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.kernel_ridge import KernelRidge
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.linear_model import SGDClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import RadiusNeighborsClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.naive_bayes import GaussianNB  # Gaussian Naive Bayes
from sklearn.naive_bayes import ComplementNB #for class imbalance
from sklearn.tree import DecisionTreeClassifier
# ensemble methods
from sklearn.ensemble import RandomForestClassifier  # Forests of randomized trees
from sklearn.ensemble import ExtraTreesClassifier  # Extremely Randomized Trees
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import GradientBoostingClassifier  # Gradient Tree Boosting or Gradient Boosted Regression Trees (GBRT)
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import VotingClassifier
# neural network
from sklearn.neural_network import MLPClassifier

clas = [RidgeClassifier, LogisticRegression, Perceptron, PassiveAggressiveClassifier, LinearDiscriminantAnalysis,
        QuadraticDiscriminantAnalysis, KernelRidge, SVC, SGDClassifier, KNeighborsClassifier,
        GaussianProcessClassifier, GaussianNB, ComplementNB, DecisionTreeClassifier, RandomForestClassifier, ExtraTreesClassifier,
        BaggingClassifier, GradientBoostingClassifier, AdaBoostClassifier, VotingClassifier, MLPClassifier]

from sklearn.multioutput import MultiOutputClassifier

from scipy import stats

#Unsupervised clustering
def clus_uns (x, y, path,xtest="", ytest="", rs=None, M="pca",n=2, v_names="", v1="",v2="", n2=2):
    # Clustering
    from sklearn.cluster import KMeans
    from sklearn.cluster import MiniBatchKMeans
    from sklearn.cluster import AffinityPropagation
    from sklearn.cluster import MeanShift
    from sklearn.cluster import SpectralClustering
    from sklearn.cluster import AgglomerativeClustering
    from sklearn.cluster import DBSCAN
    from sklearn.cluster import Birch
    from sklearn import decomposition

    import numpy as np
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    from sklearn import datasets

    if "cat" == "cat11":
        y1 = []
        for i in y:
            if i < 6:
                y1 += ["b"]
            elif i >= 6 and i <= 8:
                y1 += ["grey"]
            else:
                y1 += ["r"]
        #print y1
    print (y)
    if "change" == "change11":
        y1 = []
        for i in y:
            if i == 2:
                y1 += [1]
            elif i == 0:
                y1 += [0]

    #print y1


    y1 = y########################


    #print (y1)
    colorset=["lime", "blue", "yellow", "red"]
    marker2=[10,11]
    marker3 = ["^", "o"]
    colors2 = colorset[0:n]
    colors3 = colorset[0:n]
    #print colors2
    colors3.reverse()


    if M=="pca" or M=="pc-km":
        np.random.seed(1)
        centers = [[1, 1], [-1, -1], [1, -1]]
        X = x
        
        # Specify number of the components
        pca = decomposition.PCA(n_components=3)
        pca.fit(X)
        X = pca.transform(X)

        print (dir (pca))

        # Dump components relations with features:
        w=pandas.DataFrame(pca.components_, columns=x.columns, index=['PC-1', 'PC-2', "PC-3"])
        print (w)
        w=w.transpose()
        w.to_csv(path+"pc-weights.csv")

        plt.title("PCA",fontsize=16)
        plt.xlabel("PC1")
        plt.ylabel("PC2")
        plt.scatter(X[:, 0], X[:, 1], c=y1, cmap=matplotlib.colors.ListedColormap(colors2))#marker="^"
        plt.show()
        plt.title("PCA", fontsize=16)
        plt.xlabel("PC1")
        plt.ylabel("PC3")
        plt.scatter(X[:, 0], X[:, 2], c=y1,cmap=matplotlib.colors.ListedColormap(colors2))
        plt.show()
        plt.title("PCA", fontsize=16)
        plt.xlabel("PC2")
        plt.ylabel("PC3")
        plt.scatter(X[:, 1], X[:, 2], c=y1, cmap=matplotlib.colors.ListedColormap(colors2))
        plt.show()

        fig = plt.figure(1, figsize=(4, 3))
        plt.clf()
        ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=48, azim=134)


        plt.cla()

        ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=y1, cmap=matplotlib.colors.ListedColormap(colors2))#, cmap=plt.cm.nipy_spectral, edgecolor='k')

        ax.w_xaxis.set_ticklabels([])
        ax.w_yaxis.set_ticklabels([])
        ax.w_zaxis.set_ticklabels([])

        #plt.title("PCA", fontsize=16)
        plt.xlabel("PC1")
        plt.ylabel("PC3")
        #plt.zlabel("PC2")

        plt.show()

        return [X[:, 0], X[:, 1], X[:, 2]]

    if M=="kmeans" or M== "pc-km":
        #print x
        if M == "pc-km":
            kmeans = KMeans(n_clusters=n2, random_state=1).fit(X)
        elif M == "kmeans":
            kmeans = KMeans(n_clusters=n2, random_state=1).fit(x)
        #print x
        labels = kmeans.labels_
        #print labels
        #print y1

        centers = kmeans.cluster_centers_
        #print centers

        if "labelchange" == "labelchange":
            labelslist= labels.tolist()
            if labelslist.count(1) > len(labels)/2:
                for i in range (len(labels)):
                    if labels[i] == 1:
                        labels[i]=1#2
                    elif labels[i] == 0:
                        labels[i] =0 #1
            else:
                for i in range (len(labels)):
                    if labels[i] == 1:
                        labels[i]=0#1
                    elif labels[i] == 0:
                        labels[i] =1 #2

        d = kmeans.cluster_centers_
        fig = plt.figure(1, figsize=(4, 3))
        X_array = numpy.array(x)



        if M== "kmeans" and "plot" == "plot":
            if v1 != "" and v2 != "" and v_names!= "":
                va=v_names.index(v1)
                vb=v_names.index(v2)

            else:
                va = 0
                vb = 1
            plt.title("K-Means clustering (predicted classes)", fontsize=14)
            plt.xlabel(str(v1), fontsize=10)
            plt.ylabel(str(v2), fontsize=10)
            plt.scatter(X_array[:,va],X_array[:,vb], c=labels, cmap=matplotlib.colors.ListedColormap(colors2), alpha=0.6)
            plt.scatter(centers[:, va], centers[:, vb], c=["darkblue", "darkgreen"], s=300, alpha=1, marker="P")
            plt.show()
            plt.title("K-Means clustering (actual classes)",fontsize=14)
            plt.xlabel(str(v1), fontsize=10)
            plt.ylabel(str(v2), fontsize=10)

            plt.scatter(X_array[:, va], X_array[:, vb], c=y1, cmap=matplotlib.colors.ListedColormap(colors2), alpha=0.6)
            plt.scatter(centers[:, va], centers[:, vb], c=["darkblue", "darkgreen"], s=300, alpha=1, marker="P")
            plt.show()

        elif M =="pc-km":
            plt.title("K-Means clustering on PC (predicted labels)",fontsize=14)
            plt.xlabel("Variable 1")
            plt.ylabel("Variable 2")
            plt.scatter(centers[:, 0], centers[:, 1], c=colors3, s=300, alpha=0.5,marker="P")
            plt.scatter(X[:, 0], X[:, 1], c=labels, cmap=matplotlib.colors.ListedColormap(colors2),
                        alpha=0.6)
            plt.show()
            plt.title("K-Means clustering on PC (actual labels)",fontsize=14)
            plt.scatter(centers[:, 0], centers[:, 1], c=colors3, s=300, alpha=0.5, marker="P")
            plt.scatter(X[:, 0], X[:, 1], c=y1, cmap=matplotlib.colors.ListedColormap(colors2), alpha=0.6)
            plt.show()

        #print (labels)
        #print (y1)
        if "binary"=="binary":
            cnf = confusion_matrix(y1, labels)
            #print cnf
            tp, fp, fn, tn = cnf.ravel()
            #print ("tn,fp,fn,tp", tn, fp, fn, tp)
            cnf[0][0] = tp
            cnf[1][1] = tn
            cnf[0][1] = fn
            cnf[1][0] = fp
            #print (cnf)###################################

        if M == "kmeans":
            ytest_labels = kmeans.predict(xtest)
            ytest_labelslist = ytest_labels.tolist()
            if ytest_labelslist.count(1) > len(ytest_labels) / 2:
                for i in range(len(ytest_labels)):
                    if ytest_labels[i] == 1:
                        ytest_labels[i] =1#2
                    elif ytest_labels[i] == 0:
                        ytest_labels[i] = 0#1
            else:
                for i in range(len(ytest_labels)):
                    if ytest_labels[i] == 1:
                        ytest_labels[i] =0# 1
                    elif ytest_labels[i] == 0:
                        ytest_labels[i] = 1#2
            if "binary" == "binary":
                cnf3 = confusion_matrix(ytest, ytest_labels)
                tpt, fpt, fnt, tnt = cnf3.ravel()
                # print ("tn,fp,fn,tp", tn, fp, fn, tp)
                cnf3[0][0] = tpt
                cnf3[1][1] = tnt
                cnf3[0][1] = fnt
                cnf3[1][0] = fpt
                #print (cnf3)
            if "binary" == "binary":
                testmetrics = metr(tpt, tnt, fpt, fnt)
                #print ("test", testmetrics)
                total = metr(tpt + tp, tnt + tn, fpt + fp, fnt + fn)
                #print ("total", total)
                # tp, fp, fn, tn = cnf.ravel()

                trainmetrics = metr(tp, tn, fp, fn)
                #print("train", trainmetrics)
                # tpt, fpt, fnt, tnt = cnf2.ravel()
                plt.figure()
                plot_confusion_matrix(cnf, classes=["N1", "N0"], title="Confusion matrix - " + "K-means")
                plt.ylabel("Predicted")
                plt.xlabel("Actual")
                # plt.figure()
                # plot_confusion_matrix(cnf2, classes="", title='Confusion matrix test, without normalization')
                # Plot normalized confusion matrix
                # plt.figure()
                # plot_confusion_matrix(cnf, classes="", normalize=True,  title='Normalized confusion matrix train')

                plt.show()  ######################

def Model(x, y, xtest, ytest, v_names, c=10, M="mlr", rs=None, cv="loo"):
    
    #Variations of splits
    # from sklearn.model_selection import KFold
    # loo= KFold(n_splits=2)
    # from sklearn.model_selection import RepeatedKFold
    # loo = RepeatedKFold(n_splits=2, n_repeats=2, random_state=random_state)
    # from sklearn.model_selection import LeavePOut
    # loo = LeavePOut(p=2)
    # from sklearn.model_selection import ShuffleSplit
    # loo = ShuffleSplit(n_splits=3, test_size=0.25, random_state = 0)
    # from sklearn.model_selection import StratifiedKFold
    # loo = StratifiedKFold(n_splits=3)

    # gkf = GroupKFold(n_splits=3)
    # logo = LeaveOneGroupOut()
    # lpgo = LeavePGroupsOut(n_groups=2)
    # gss = GroupShuffleSplit(n_splits=4, test_size=0.5, random_state=0)
    # tscv = TimeSeriesSplit(n_splits=3)
    
    ytests = []
    ypreds = []
    X_array = numpy.array(x)
    y_array = numpy.array(y)

    if M == "pls":
        model = PLSRegression(n_components=c)
        model2 = PLSRegression(n_components=c)
    elif M == "mlr":
        model = LinearRegression()
        model2 = LinearRegression()
    elif M == "rf":
        model = RandomForestRegressor(n_estimators=c,max_depth=c-1, random_state=rs, max_features=c)
        model2= RandomForestRegressor(n_estimators=c,max_depth=c-1, random_state=rs, max_features=c)
    elif M == "svm":
        #c1=1#1
        g1="auto"
        model = SVR(kernel="rbf", gamma=g1,C=c)
        model2 = SVR(kernel="rbf", gamma=g1,C=c)
    elif M == "lsvm":
        model = LinearSVR(random_state=rs)
        model2 = LinearSVR(random_state=rs)
    elif M == "lasso":
        model = Lasso(alpha=0.1,random_state=rs)
        model2 = Lasso(alpha=0.1, random_state=rs)
    elif M == "nn":
        model = MLPRegressor(max_iter=200, hidden_layer_sizes=([20,10,10]), random_state=rs, warm_start= False, alpha=0.0001, solver="adam")
        model2 = MLPRegressor(max_iter=200, hidden_layer_sizes=(20,10,10), random_state=rs, warm_start=False, alpha=0.0001,
                             solver="adam")
    elif M == "tree":
        model = DecisionTreeRegressor(max_depth=c,max_features=c+1,random_state=rs)
        model2 = DecisionTreeRegressor(max_depth=c, max_features=c + 1, random_state=rs)
    elif M == "rg":
        model = Ridge(random_state=rs)
        model2 = Ridge(random_state=rs)
    elif M == "el":
        #All coefficiens are zero
        model = ElasticNet(random_state=rs)
        model2 = ElasticNet(random_state=rs)
    elif M == "la":
        #needs parameter change
        model = Lars()
        model2 = Lars()
    elif M == "ll":
        # All coefficiens are zero
        model = LassoLars()
        model2 = LassoLars()
    elif M == "or":
        model = OrthogonalMatchingPursuit()
        model2 = OrthogonalMatchingPursuit()
    elif M == "brg":
        model = BayesianRidge()
        model2 = BayesianRidge()
    elif M == "ardr":
        model = ARDRegression()
        model2 = ARDRegression()
    elif M == "ransa":
        #ValueError: `min_samples` may not be larger than number of samples
        model = RANSACRegressor(random_state=rs)
        model2 = RANSACRegressor(random_state=rs)
    elif M == "the":
        model = TheilSenRegressor(random_state=rs)
        model2 = TheilSenRegressor(random_state=rs)
    elif M == "hub":
        model = HuberRegressor()
        model2 = HuberRegressor()
    elif M == "sgdr":
        model = SGDRegressor(random_state=rs)
        model2 = SGDRegressor(random_state=rs)
    elif M == "kn":
        #
        model = KNeighborsRegressor()
        model2 = KNeighborsRegressor()
    elif M == "gu":
        #
        model = GaussianProcessRegressor(random_state=rs)
        model2 = GaussianProcessRegressor(random_state=rs)
    elif M == "ex":
        model = ExtraTreesRegressor(random_state=rs)
        model2 = ExtraTreesRegressor(random_state=rs)
    elif M == "bg":
        #
        model = BaggingRegressor(random_state=rs)
        model2 = BaggingRegressor(random_state=rs)
    elif M == "gb":
        model = GradientBoostingRegressor(random_state=rs)
        model2 = GradientBoostingRegressor(random_state=rs)
    elif M == "ada":
        model = AdaBoostRegressor(random_state=rs)
        model2 = AdaBoostRegressor(random_state=rs)
    else:
        print ("**************Wrong model name****************")
    model.fit(x, y)
    y_predict_train = model.predict(x)
    y_predict_test = model.predict(xtest)
    r2 = model.score(x, y)
    R2test = r2test(ytest, y_predict_test, y)
    #Pearson = stats.pearsonr(ytest, y_predict_test)
    Pearson =""
    q2f2= r2test(ytest, y_predict_test, ytest)
    model_mse_test = mean_squared_error(y_predict_test, ytest)
    math.sqrt(model_mse_test)
    #f = F(y, y_predict_train, 4)  ##################
    f=0
    if cv != "off":########################
        if cv =="loo":
            #LOO
            loo = LeaveOneOut()
        elif cv=="kf":
            loo = KFold (n_splits=5, random_state=rs,  shuffle=True)
        elif cv=="kfr":
            loo = RepeatedKFold(n_splits=2, n_repeats=2, random_state=rs)
        for train_idx, test_idx in loo.split(x):
            X_train, X_test = X_array[train_idx], X_array[test_idx]  # requires arrays
            y_train, y_test = y_array[train_idx], y_array[test_idx]
            model2.fit(X_train, y_train)
            y_pred = model2.predict(X_test)
            # there is only one y-test and y-pred per iteration over the loo.split,
            # so to get a proper graph, we append them to respective lists.
            ytests += list(y_test)
            ypreds += list(y_pred)

        rr = metrics.r2_score(ytests, ypreds)
        ms_error = metrics.mean_squared_error(ytests, ypreds)
        # print("R^2: {:.5f}%, MSE: {:.5f}".format(rr * 100, ms_error))
        q2 = q2r2(ytests, ypreds)
        rmse = RMSEP_CV_C(ytests, ypreds)
        List = {"q2": q2, "RMSECV": rmse, "Q2F2": q2f2}
    if 1 == 2:
        # CV
        scores = cross_val_score(model, x, y, cv=5)  # model3?********************
        print (scores)
        # The mean score and the 95 % confidence interval of the score estimate are hence given by:
        print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

    if M == "svm":
        sorted_VI =""
    elif M =="nn":
        VI = model.coefs_
        VI = numpy.dot(VI[0], VI[1])
    elif (M == "rf") or (M == "tree")or (M == "ex")or (M == "gb")or (M == "ada"):
        VI = model.feature_importances_
    else:
        VI = model.coef_
    if M != "svm":
        coef = VI.ravel()
        VI = pandas.DataFrame(data=VI, index=v_names)
        sorted_VI = VI.sort_values(by=[0])
        sorted_v_names = sorted_VI.index.tolist()
        sorted_coefficients = numpy.sort(coef)
    if M== "mlr":
        result = {"R2": r2, "Intercept": model.intercept_, "Mean_squared_error_test": model_mse_test,
                  "R2_test": R2test, "F": f, "Coefficients": model.coef_}#, "Variable Importance": sorted_VI}
    else:
        result = {"R2": r2, "Mean_squared_error_test": model_mse_test, "R2_test": R2test, "F": f, "Variable Importance": sorted_VI, "Pearson": Pearson}
    if "plot"=="pl":
        plt.scatter(y, y_predict_train)
        plt.show()
        plt.scatter(ytest, y_predict_test)
        plt.show()
        #plot_w(sorted_v_names,sorted_coefficients)
        #return [result, List, model]
    y1 = pandas.DataFrame(y, index=x.index, columns=["observed"])
    y2 = pandas.DataFrame(y_predict_train, index=x.index, columns=["predicted"])
    y3 = pandas.concat([y1, y2], axis=1)
    pandas.DataFrame.to_csv(y3, "train.csv")
    y4 = pandas.DataFrame(ytest, index=xtest.index, columns=["observed"])
    y5 = pandas.DataFrame(y_predict_test, index=xtest.index, columns=["predicted"])
    y6 = pandas.concat([y4, y5], axis=1)
    pandas.DataFrame.to_csv(y6, "test.csv")
    analysis= analyse(y, y_predict_train, ytest, y_predict_test, ytests, ypreds,  k=len(v_names))
    return [result, List, model, analysis]



#binary="binary"
def ModelC(x, y, xtest, ytest, v_names, c1=10, M="tree", rs=None, M2="off", cv="loo", c2=10, c3="rbf", c5=200, ep=300, dl2=[500,1000],lr1=0.01,nesterov=True,omp="adam",dp=0.2,bs=16, c4="adam", grid=False,weights='balanced'):
    # M2 should be on for MultiOutputClassifier
    # M2 should be equal to on2 to predict a new data set
    ytests = []
    ypreds = []
    X_array = numpy.array(x)
    y_array = numpy.array(y)
    
    if M == "tree":
        model = DecisionTreeClassifier(random_state=rs, max_depth=c1,max_features=c2)# ,max_leaf_nodes=6, min_samples_leaf=3)
        model2 = DecisionTreeClassifier(random_state=rs, max_depth=c1,max_features=c2)# ,max_leaf_nodes=6, min_samples_leaf=3)
    elif M == "nn":
        #Solver: c4="adam"
        #Layer_size: c1
        #max_iter: c2
        
        model = MLPClassifier(solver=c4, alpha=0.0001, hidden_layer_sizes=(c1,c1,c1, ), random_state=rs,max_iter=c5)
        model2 = MLPClassifier(solver=c4, alpha=1e-4, hidden_layer_sizes=(c1,c1,c1, ), random_state=rs,max_iter=c5)
    elif M == "rf":
        model = RandomForestClassifier( max_depth=c1, n_estimators=c1*2,random_state=rs,max_features="sqrt", max_leaf_nodes=None,class_weight=weights)
        model2 = RandomForestClassifier( max_depth=c1, n_estimators=c1*2,random_state=rs,max_features='sqrt', max_leaf_nodes=None,class_weight=weights)
        #model = RandomForestClassifier( max_depth=None, random_state=rs, max_leaf_nodes=None)
        #model2 = RandomForestClassifier( max_depth=None, random_state=rs, max_leaf_nodes=None)
    elif M == "ex":
        model = ExtraTreesClassifier(max_depth=c1, random_state=rs)
        model2 = ExtraTreesClassifier(max_depth=c1, random_state=rs)
    elif M == "lsvm":
        #default: max_iter=-1 
        model = LinearSVC(class_weight=weights, C=c1, random_state=rs)
        model2 = LinearSVC(class_weight=weights, C=c1, random_state=rs)
    elif M == "svm":
        #default: max_iter=-1 
        model = SVC(class_weight=weights ,random_state=rs, probability=True, C=c1, kernel=c3, max_iter=c5, gamma=c2 )
        model2 = SVC(class_weight=weights,random_state=rs, probability=True, C=c1, kernel=c3, max_iter=c5,gamma=c2)
    elif M == "lasso":
        #Classification metrics can't handle a mix of binary and continuous targets
        model = Lasso(alpha=0.1,random_state=rs)
        model2 = Lasso(alpha=0.1, random_state=rs)
    elif M == "lr":
        #c1=100
        #c2="sag"
        c2="liblinear"
        #model = LogisticRegression(random_state=rs, max_iter=c1, solver=c2)
        #model2 = LogisticRegression(random_state=rs, max_iter=c1, solver=c2)
        model = LogisticRegression(max_iter=c1,random_state=rs)
        model2 = LogisticRegression(max_iter=c1,random_state=rs)
    elif M == "ld":
        model = LinearDiscriminantAnalysis()
        model2 = LinearDiscriminantAnalysis()
    elif M == "rg":
        model = RidgeClassifier(random_state=rs)
        model2 = RidgeClassifier(random_state=rs)
    elif M == "per":
        model = Perceptron(random_state=rs)
        model2 = Perceptron(random_state=rs)
    elif M == "pass":
        #parameters
        model = PassiveAggressiveClassifier(random_state=rs)
        mode2 = PassiveAggressiveClassifier(random_state=rs)
    elif M == "qua":
        #
        model = QuadraticDiscriminantAnalysis()
    elif M == "kern":
        #ValueError: Classification metrics can't handle a mix of binary and continuous targets
        model = KernelRidge()
    elif M == "sgdc":
        #parameters
        model = SGDClassifier(random_state=rs)
    elif M == "kn":
        #
        model = KNeighborsClassifier(n_neighbors=c1)
        model2 = KNeighborsClassifier(n_neighbors=c1)
    elif M == "rn":
        #
        model = RadiusNeighborsClassifier()
        model2 = RadiusNeighborsClassifier()
    elif M == "gu":
        #
        model = GaussianProcessClassifier(random_state=rs)
    elif M == "gunb":
        #
        model = GaussianNB()
        model2 = GaussianNB()
    elif M == "cnb":
        #
        model = ComplementNB()
        model2 = ComplementNB()
    elif M == "bg":
        #
        model = BaggingClassifier(random_state=rs)
    elif M == "gb":
        model = GradientBoostingClassifier(random_state=rs)
        model2 = GradientBoostingClassifier(random_state=rs)
    elif M == "ada":
        model = AdaBoostClassifier(random_state=rs)
        model2 = AdaBoostClassifier(random_state=rs)
    elif M == "v":
        #error
        model = VotingClassifier()
    elif M=="dl":
        # fix random seed for reproducibility
        seed = 1
        numpy.random.seed(seed)

        if M=="dl":

            if M == "dl":
                # fix random seed for reproducibility
                seed = 1
                numpy.random.seed(seed)
                from keras.models import Sequential
                from keras.layers import Dense, Dropout
                from keras import optimizers
                from keras import backend as K
                K.clear_session()


                def get_default_args(func):
                    import inspect
                    """
                    returns a dictionary of arg_name:default_values for the input function
                    """
                    args, varargs, keywords, defaults = inspect.getargspec(func)
                    return dict(zip(args[-len(defaults):], defaults))

                #print(get_default_args (optimizers.SGD))

                sgd = optimizers.SGD(lr=lr1, momentum=0.9, nesterov=nesterov)
                adam = optimizers.Adam(lr=lr1)

                dl1=len(dl2)

                def baseline_model():
                    # create model
                    model = Sequential()
                    if dl1 == 4:
                        model.add(Dense(dl2[3], input_dim=len(v_names), kernel_initializer='uniform', activation='relu'))
                        model.add(Dropout(dp))
                        if dl1 == 3:
                            model.add(
                                Dense(dl2[2], input_dim=len(v_names), kernel_initializer='uniform', activation='relu'))
                        model.add(Dense(dl2[2], kernel_initializer='uniform', activation='relu'))
                        model.add(Dropout(dp))
                    if dl1 >= 2:
                        if dl1 == 2:
                            model.add(
                                Dense(dl2[1], input_dim=len(v_names), kernel_initializer='uniform', activation='relu'))
                        model.add(Dense(dl2[1], kernel_initializer='uniform', activation='relu'))
                        model.add(Dropout(dp))
                    if dl1 >= 1:
                        if dl1 == 1:
                            model.add(
                                Dense(dl2[0], input_dim=len(v_names), kernel_initializer='uniform', activation='relu'))
                        model.add(Dense(dl2[0], kernel_initializer='uniform', activation='relu'))
                        model.add(Dropout(dp))
                    model.add(Dense(1, kernel_initializer='normal',activation='sigmoid'))
                    # Compile models
                    if omp == "sgd":
                        model.compile(loss='binary_crossentropy', optimizer=sgd, metrics=["accuracy"])#
                    elif omp=="adam":
                        model.compile(loss='binary_crossentropy', optimizer=adam, metrics=["accuracy"])#'CategoricalCrossentropy'
                    #Cross-entropy is the default loss function to use for binary classification problems.
                    #It is intended for use with binary classification where the target values are in the set {0, 1}.
                    return model

                model = baseline_model()

                from keras.callbacks import EarlyStopping
                early_stopping_monitor = EarlyStopping(monitor='val_loss', patience=0)
                from sklearn.utils import class_weight
                #class_weights = class_weight.compute_class_weight('balanced',numpy.unique(y_array),y_array)
                class_weights = weights

    else:
        print ("**************Wrong model name****************")
    if M2 == "on":
        model = MultiOutputClassifier(model, n_jobs=1)
        #model = RandomForestClassifier(max_depth=10,random_state=rs)
    if M == "dl":
        from sklearn.model_selection import StratifiedKFold
        #model.fit(x,y,batch_size=32, epochs=ep)

        loo = StratifiedKFold(n_splits=5)#, random_state=rs)  #################n_splits=5
        n = 0
        for train_idx, test_idx in loo.split(x,y):#####################
            X_train, X_test = X_array[train_idx], X_array[test_idx]  # requires arrays
            y_train, y_test = y_array[train_idx], y_array[test_idx]

            if M=="dl":
                if n==0:
                    #print (n)
                    model2 = baseline_model()

                    ######it seems that there is a critical problem with CV, early stoping??
                    #model2.fit(X_train, y_train, batch_size=bs, epochs=ep, validation_data=[X_test,y_test], callbacks=[early_stopping_monitor],verbose=0,,class_weight=class_weights)
                    model2.fit(X_train, y_train, batch_size=bs, epochs=ep, validation_data=[X_test,y_test], verbose=0,class_weight=class_weights)

                    #for tensorflow v 2.5 or earlier
                    #y_pred = model2.predict_classes(X_test, verbose=0)

                    #for tensorflow v 2.6 or later
                    predict_temp=model.predict(X_test) 
                    #print(predict_temp,1)
                    y_pred2=numpy.argmax(predict_temp,axis=1)
                    #print(y_pred2,2)
                    y_pred=(model2.predict(X_test) > 0.5).astype("int32")
                    #print(y_pred,3)
                    #model2 = model
                    ytests += list(y_test)
                    ypreds += list(y_pred)
                    #print (ytests, 'ytest')
                    #print (ypreds, 'ypred')
            n = n + 1
        #model.fit(x,y, batch_size=32, epochs=200, validation_split=0.2, callbacks=[early_stopping_monitor])


        #print (early_stopping_monitor.stopped_epoch)

        #n_epochs = len(model2.history['loss'])
        #print ( len(model2.history))
        #print (model2.history)

        #input("pause")
        model.fit(x, y, batch_size=bs, epochs=(early_stopping_monitor.stopped_epoch),verbose=0, class_weight=class_weights)

        #input ("pause1")
        #model2.fit(x, y, batch_size=32, epochs=2000, validation_split=0.2, callbacks=[early_stopping_monitor])
        #input ("pause2")

        #for tensorflow v 2.5 or earlier
        #y_predict_train = model.predict_classes(x, verbose=0)
        #y_predict_test = model.predict_classes(xtest, verbose=0)

        #for tensorflow v 2.6 or later
        #predict_temp=model.predict(x) 
        #y_predict_train=numpy.argmax(predict_temp,axis=1)
        y_predict_train=(model.predict(x) > 0.5).astype("int32")
        
        #predict_temp=model.predict(xtest) 
        #y_predict_test=numpy.argmax(predict_temp,axis=1)
        y_predict_test=(model.predict(xtest) > 0.5).astype("int32")
        #print (y_predict_test,'ptest')
        #print (ytest,'test')
        #print (y_predict_train)
        #print (y)
        def numpy_concatenate(a):
            return list(numpy.concatenate(a))

        #for tensorflow v 2.5 or earlier
        #y_predict_train = numpy_concatenate(y_predict_train)
        #y_predict_test = numpy_concatenate(y_predict_test)

        #for tensorflow v 2.6 or later
        #pass

    else:
        model.fit(x, y)
        y_predict_train = model.predict(x)
        y_predict_test = model.predict(xtest)


    #y_predict_train =reduce(operator.concat, y_predict_train)
    #y_predict_test = reduce(operator.concat, y_predict_test)
    #scores = cross_val_score(model2, x, y)###############
    #print scores.mean()###############
    # r2 = model.score(x, y)
    # R2test = r2test(ytest, y_predict_test, y)
    # model_mse_test = mean_squared_error(y_predict_test, ytest)
    # math.sqrt(model_mse_test)
    # f = F(y, y_predict_train, 4)  ##################
    #print (y_predict_train)
    accuracy_score_train= accuracy_score(y, y_predict_train, normalize= True)
    accuracy_score_test=  accuracy_score(ytest, y_predict_test, normalize= True)
    if M2 != "on" and M2 != "on2" and M2 != "dl":
        ni = 2000
        if cv =="loo":
            #LOO
            loo = LeaveOneOut()
        elif cv=="kf":
            loo = KFold (n_splits=5, shuffle=True, random_state=rs)
        elif cv=="kfr":
            loo = RepeatedKFold(n_splits=2, n_repeats=2, random_state=rs)
        elif cv=="shuff":
            loo= ShuffleSplit(n_splits=ni, test_size=0.2, random_state=0)
        n = 0
        for train_idx, test_idx in loo.split(x):
            # requires arrays
            X_train, X_test = X_array[train_idx], X_array[test_idx]  
            y_train, y_test = y_array[train_idx], y_array[test_idx]
            if M != 'dl':
                model2.fit(X_train, y_train)

                y_pred = model2.predict(X_test)


            if n == 0:
                ytests2 = [list(y_test)]
                ypreds2 = [list(y_pred)]
            elif n!= 0:
                ytests2 = ytests2 +[list(y_test)]
                ypreds2 = ypreds2 + [list(y_pred)]
            # there is only one y-test and y-pred per iteration over the loo split,
            # so to get a proper graph, we append them to respective lists.
            if M!= "dl":
                ytests += list(y_test)
                ypreds += list(y_pred)
                n = n + 1
            accuracy_score_LOO = accuracy_score(ytests, ypreds, normalize=True)
        if cv=="shuff":
            accuracy_score_LOO_all=[None]*ni
            for i in range(0,ni):
                accuracy_score_LOO_all[i]= accuracy_score(ytests2[i], ypreds2[i], normalize=True)
            print (accuracy_score_LOO)
            print (numpy.mean (accuracy_score_LOO_all), "<---- Mean of the accuracies")
            print (numpy.std(accuracy_score_LOO_all), "<---- STD of the accuracies")
        rr = metrics.r2_score(ytests, ypreds)
        ms_error = metrics.mean_squared_error(ytests, ypreds)
        # print("Leave One Out Cross Validation")
        # print("R^2: {:.5f}%, MSE: {:.5f}".format(rr * 100, ms_error))
        # q2 = q2r2(ytests, ypreds)
        # print q2
        # rmse = RMSEP_CV_C(ytests, ypreds)
        # List = {"q2": q2, "RMSECV": rmse}
    else:
        accuracy_score_LOO=""

    if M2 != "on":
        if M == "tree":
            import os
            # It is necessary to make the Graphviz works after the installation
            os.environ["PATH"] += os.pathsep + 'D:/Program Files (x86)/Graphviz2.38/bin/'
            import graphviz
            from sklearn.tree import export_graphviz

            dot_data = export_graphviz(model, out_file=None, max_depth=None, feature_names=v_names, class_names=["1", "2","3","4"], label="root", impurity=True, proportion=False, rounded=True, precision=2)###################
            graph = graphviz.Source(dot_data)
            graph.render("tree_chart")

            VI = model.feature_importances_
        elif (M=="rf") or (M=="ex") or (M=="lasso") or (M=="gb") or (M=="ada"):
            VI = model.feature_importances_
        elif (M == "svm"):
                if c3=="linear":
                    #VI = model.dual_coef_
                    VI = model.coef_[0]
                else:
                    VI=""
        elif M == "nn":
            VI = model.coefs_
            
            VI = numpy.dot(VI[0], VI[1])
        elif (M=="qua") or (M=="kn") or (M=="gu") or (M=="bg"):
            VI=""
        elif M == "gunb":
            #neg_class_prob_sorted = model.feature_log_prob_[0, :].argsort()
            #pos_class_prob_sorted = model.feature_log_prob_[1, :].argsort()
            #print(numpy.take(count_vect.get_feature_names(), neg_class_prob_sorted[:10]))
            #print(numpy.take(count_vect.get_feature_names(), pos_class_prob_sorted[:10]))
            #VI = [neg_class_prob_sorted, pos_class_prob_sorted]
            #pred_proba = model.predict_proba(x)
            #words = numpy.take(count_vect.get_feature_names(), pred_proba.argmax(axis=1))
            #print (words)
            VI=""
        elif M =="dl":
            VI=""
        else:
            VI = model.coef_[0]



        if  (M=="qua") or (M=="kn") or (M=="gu")  or (M=="bg") or (M=="rn")or (M=="dl"):#(M== "svm") or
            sorted_VI= ""
            coef =""
        elif (M=="gunb"):
            sorted_VI = ""
        elif M=="svm":
            if c2!="linear":
                sorted_VI = ""
        else:
            coef = VI.ravel()
            VI = pandas.DataFrame(data=VI, index=v_names)
            sorted_VI = VI.sort_values(by=[0])
            sorted_v_names = sorted_VI.index.tolist()
            sorted_coefficients = numpy.sort(coef)
            
            #result = {"R2": r2, "Mean_squared_error_test": model_mse_test, "R2_test": R2test, "F": f, "Variable Importance": sorted_VI}
            # To plot the importances
            #plot_w(sorted_v_names, sorted_coefficients)  
        if (M=="tree") or (M=="ld") or (M=="lr") or (M== "nn") or (M=="rf")   or (M=="lasso") or (M=="ex") or (M=="gb") or (M=="ada") or (M== "svm") or (M=="qua") or (M=="kn") or (M=="gu") or (M=="gunb") or (M=='cnb') or(M=="bg") or (M=="rn"):
            pra_train= model.predict_proba(x)
            pra_test = model.predict_proba(xtest)
        elif M=="dl":
            pra_train = model.predict(x)
            pra_test = model.predict(xtest)
        elif (M== "lsvm") or (M=="rg")or (M== "per"):
            pra_train = ""
            pra_test = ""
        else:
            pra = ""##############
        #int "pra",  pra_test
    else:
        sorted_VI =""
    if M2 != "on" and M2 != "on2":
        # They are not correct
        cnf = confusion_matrix (y, y_predict_train)
        cnf3= confusion_matrix (ytests, ypreds)#CV
        cnf2 = confusion_matrix (ytest, y_predict_test)
        
    else:
        cnf =""
        cnf2=""
        cnf3=""
    # If the number of classes is two:
    if "binary" == "binary":
        tp, fp, fn, tn = cnf.ravel()
        #print ("tn,fp,fn,tp", tn, fp, fn, tp)
        cnf[0][0]= tp
        cnf[1][1]= tn
        cnf[0][1] = fn
        cnf[1][0] = fp
        tpt, fpt, fnt, tnt = cnf2.ravel()
        cnf2[0][0] = tpt
        cnf2[1][1] = tnt
        cnf2[0][1] = fnt
        cnf2[1][0] = fpt
        tpc, fpc, fnc, tnc = cnf3.ravel()
        cnf3[0][0] = tpc
        cnf3[1][1] = tnc
        cnf3[0][1] = fnc
        cnf3[1][0] = fpc

        
        # tp, fp, fn, tn = cnf.ravel()
        trainmetrics = metr(tp, tn, fp, fn)
        #print("train", trainmetrics)
        cvmetrics = metr(tpc, tnc, fpc, fnc)
        #print ("CV", cvmetrics)
        # tpt, fpt, fnt, tnt = cnf2.ravel()
        testmetrics = metr(tpt, tnt, fpt, fnt)
        #print ("test", testmetrics)
        total = metr(tpt + tp, tnt + tn, fpt + fp, fnt + fn)
        #print ("total", total)
    else:
        result = "empty"
    # Plot non-normalized confusion matrix
    if "plot" == "plot":
        plt.figure()
        plot_confusion_matrix(cnf, classes=["Inactive","Active"], title="Train set")
        plt.ylabel("Predicted",fontsize="12",fontweight="bold")
        plt.xlabel("Actual",fontsize="12",fontweight="bold")
        #plt.figure()
        #plot_confusion_matrix(cnf2, classes="", title='Confusion matrix test, without normalization')
        # Plot normalized confusion matrix
        #plt.figure()
        #plot_confusion_matrix(cnf, classes="", normalize=True,  title='Normalized confusion matrix train')

        #plt.show()  
        plt.figure()
        plot_confusion_matrix(cnf3, classes=["Inactive","Active"], title="Cross-Validation")
        plt.ylabel("Predicted",fontsize="12",fontweight="bold")
        plt.xlabel("Actual",fontsize="12",fontweight="bold")
        #plt.show()

        #plt.show()
        plt.figure()
        plot_confusion_matrix(cnf2, classes=["Inactive","Active"], title="Test set")
        plt.ylabel("Predicted",fontsize="12",fontweight="bold")
        plt.xlabel("Actual",fontsize="12",fontweight="bold")
        #plt.show()  

        

    #rocc()

    y1=pandas.DataFrame(y,  index=x.index, columns=["observed"])
    y2=pandas.DataFrame(y_predict_train,  index=x.index, columns=["predicted"])
    y3 = pandas.concat([y1, y2], axis=1)
    pandas.DataFrame.to_csv(y3, "train.csv")
    y4 = pandas.DataFrame(ytest, index=xtest.index, columns=["observed"])
    y5 = pandas.DataFrame(y_predict_test, index=xtest.index, columns=["predicted"])
    y6 = pandas.concat([y4, y5], axis=1)
    pandas.DataFrame.to_csv(y6, "test.csv")

    if "binary" == "binary":
        result ={"y1": y1, "y2": y2, "y3": y3, "y4": y4,"y5": y5, "y6": y6,"confusion matrix": cnf, "accuracy_score_train": accuracy_score_train,
                  "accuracy_score_test": accuracy_score_test, "accuracy_score_LOO": accuracy_score_LOO,
                  "Variable Importance": sorted_VI, "probability_train": pra_train, "probability_test": pra_test, "confusion matrix_CV":cnf3, "confusion matrix_test":cnf2,
                 "train_metrics": trainmetrics, "CV_metrics":cvmetrics,"test_metrics":testmetrics, "total_metrics":total,"VI":VI}
        result1 = {"y1": y1, "y2": y2, "confusion matrix": cnf, "accuracy_score_train": accuracy_score_train,
                  "accuracy_score_test": accuracy_score_test, "accuracy_score_LOO": accuracy_score_LOO,
                  "Variable Importance": sorted_VI, "probability_train": pra_train, "probability_test": pra_test, "confusion matrix":cnf3, "confusion matrix":cnf2,
                  "train_AC":trainmetrics["AC"], "train_SEN":trainmetrics["SEN"], "train_SPEC":trainmetrics["SPEC"], "train_PREC":trainmetrics["PREC"], "train_F":trainmetrics["F"], "train_MCC":trainmetrics["MCC"],
                  "test_AC": testmetrics["AC"], "test_SEN": testmetrics["SEN"], "test_SPEC": testmetrics["SPEC"],
                  "test_PREC": testmetrics["PREC"], "test_F": testmetrics["F"], "test_MCC": testmetrics["MCC"],
                  "total_AC": total["AC"], "total_SEN": total["SEN"], "total_SPEC": total["SPEC"],
                  "total_PREC": total["PREC"], "total_F": total["F"], "total_MCC": total["MCC"]}


    else:
        result = {"y1": y1, "y2": y2, "confusion matrix": cnf, "accuracy_score_train": accuracy_score_train,
                  "accuracy_score_test": accuracy_score_test, "accuracy_score_LOO": accuracy_score_LOO,
                  "Variable Importance": sorted_VI, "probability_train": pra_train, "probability_test": pra_test,
                   "train_MCC":trainmetrics["MCC"],"test_MCC": testmetrics["MCC"],"VI":VI}
    return [result, model]








##########################################################################################################################################
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    
    #plt.rcParams['font.size'] = 40
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, numpy.newaxis]
        #print("Normalized confusion matrix")
    else:
        pass#print('Confusion matrix, without normalization')

    #print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title,fontsize="26",fontweight="bold")
    plt.colorbar()
    tick_marks = numpy.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45,fontsize="12")
    plt.yticks(tick_marks, classes,fontsize="12")

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black",fontsize="30",fontweight="bold")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
def plot_w(x,y):
    plt.bar(x, y)
    plt.xticks(rotation=90)
    plt.show()
def rocc (x,y,ycat="",title='Receiver operating characteristic', tsize=12, pos_l=1, l="b"):
    def COUNT(col, l=l, percent=1):
        n=0
        t= int(len(col)*percent)

        #print col
        #print col.iloc[0], 44444
        for i in range (0, t):
            if col.iloc[i] == l:
                n= n+1
                #print col.iloc[i]
        return n

    #print (y)
    all=float(len(y))
    #sub1 = int (all * 0.01)
    B=(COUNT(y,l=l,percent=1))/all
    #print "B", B

    A1=(COUNT(y,l=l,percent=0.01))/(all*0.01)
    #print (COUNT(y,l="b",percent=0.01)),"A1"
    A2 = (COUNT(y, l="b", percent=0.02)) / (all * 0.02)
    A10 = (COUNT(y, l="b", percent=0.1)) / (all * 0.1)
    A20 = (COUNT(y, l="b", percent=0.2)) / (all * 0.2)
    A50 = (COUNT(y, l="b", percent=0.5)) / (all * 0.5)
    EF1= A1/B
    EF2 = A2 / B
    EF10 = A10 / B
    EF20 = A20 / B
    EF50 = A50 / B
    print (EF1, " --> EF1")
    print (EF2, " --> EF2")
    print (EF10, " --> EF10")
    print (EF20, " --> EF20")
    print (EF50, " --> EF50")

    t = 1.1
    if ycat=="cat":
        for i in range (0, len(y)):
            if y[i] <= t:
                y[i] = 1
            elif y[i] > t:
                y[i] = 0
    fpr, tpr, thresholds = metrics.roc_curve(y, x, pos_label=pos_l)
    AUC= metrics.auc (fpr,tpr)
    print (AUC)
    plt.figure()
    lw = 2
    plt.plot(fpr, tpr, color='darkorange',
             lw=lw, label='ROC curve (area = %0.2f)' % AUC)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title, fontsize=tsize)
    plt.legend(loc="lower right")
    plt.show()

def crazy_div(x, y):
    return 0 if y == 0 else x / y

def metr(tp1,tn1,fp1,fn1):
    #print ("tp,tn,fp,fn" , tp1,tn1,fp1,fn1,"def")
    tp=float(tp1)
    tn=float(tn1)
    fp=float(fp1)
    fn=float(fn1)
    accuracy_m = ifzero((tp+tn),(tp+tn+fp+fn))
    accuracy_m =float(accuracy_m)
    precision_m =ifzero((tp),(tp+fp))#fp rate
    sensitivity_m =ifzero((tp),(tp+fn))#recall
    specificity_m =ifzero((tn),(tn+fp))
    F_measure_m=ifzero((2*sensitivity_m*precision_m),(sensitivity_m+precision_m))
    MCC= ifzero(((tp*tn)-(fp*fn)),(math.sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn))))
    listcnf = {"AC":accuracy_m, "SEN": sensitivity_m, "SPEC": specificity_m, "PREC": precision_m, "F": F_measure_m, "MCC": MCC}
    return listcnf

def ifzero (x1,x2):
    try:
        return x1 / x2
    except ZeroDivisionError:
        return 0

