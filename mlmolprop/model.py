"""Fit and evaluate regression/classification models, with clustering and metrics helpers."""

from __future__ import annotations

import itertools
import math

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn import metrics
from sklearn.cross_decomposition import PLSRegression
from sklearn.discriminant_analysis import (
    LinearDiscriminantAnalysis,
    QuadraticDiscriminantAnalysis,
)
from sklearn.ensemble import (
    AdaBoostClassifier,
    AdaBoostRegressor,
    BaggingClassifier,
    BaggingRegressor,
    ExtraTreesClassifier,
    ExtraTreesRegressor,
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.gaussian_process import GaussianProcessClassifier, GaussianProcessRegressor
from sklearn.linear_model import (
    ARDRegression,
    BayesianRidge,
    ElasticNet,
    HuberRegressor,
    Lars,
    Lasso,
    LassoLars,
    LinearRegression,
    LogisticRegression,
    OrthogonalMatchingPursuit,
    PassiveAggressiveClassifier,
    Perceptron,
    RANSACRegressor,
    Ridge,
    RidgeClassifier,
    SGDClassifier,
    SGDRegressor,
    TheilSenRegressor,
)
from sklearn.metrics import accuracy_score, confusion_matrix, mean_squared_error
from sklearn.model_selection import (
    KFold,
    LeaveOneOut,
    RepeatedKFold,
    ShuffleSplit,
)
from sklearn.naive_bayes import ComplementNB, GaussianNB
from sklearn.neighbors import (
    KNeighborsClassifier,
    KNeighborsRegressor,
    RadiusNeighborsClassifier,
)
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.svm import SVC, SVR, LinearSVC, LinearSVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from .basic import RMSEP_CV_C, analyse, q2r2, r2test


def _distinct_colors(count: int) -> list:
    """Return `count` visually distinct colors from a qualitative colormap."""
    cmap = (
        matplotlib.colormaps["tab10"] if count <= 10 else matplotlib.colormaps["tab20"]
    )
    return [cmap(i) for i in range(count)]


def clus_uns(
    x,
    y,
    path,
    xtest=None,
    ytest=None,
    rs=None,
    M="pca",
    n=2,
    v_names=None,
    v1=None,
    v2=None,
    n2=2,
):
    """Unsupervised clustering / PCA visualization.

    Parameters
    ----------
    x : array-like
        Feature matrix.
    y : array-like
        Labels, used only for coloring plots.
    path : str
        Directory prefix for output files (e.g. ``"pc-weights.csv"``).
    xtest, ytest
        Held-out set, used to report cluster assignment metrics when
        ``M="kmeans"``.
    M : {"pca", "kmeans", "pc-km"}, default "pca"
        "pca" plots the first 3 principal components and returns their
        scores. "kmeans" runs KMeans directly on ``x``. "pc-km" runs
        KMeans on the PCA scores.
    n : int, default 2
        Number of colors to use from the palette (>= number of classes).
    v_names, v1, v2 : list[str], str, str
        Feature names and the two feature names to plot on the axes for
        M="kmeans" (ignored for "pca"/"pc-km").
    n2 : int, default 2
        Number of KMeans clusters. For ``M="kmeans"``, the downstream
        evaluation metrics (confusion matrix vs. ``y``/``ytest``,
        sensitivity/specificity/etc. via :func:`metr`) assume binary
        clustering and only work correctly for ``n2=2``; ``M="pc-km"``
        (which skips that evaluation step) supports any ``n2``.
    """
    from sklearn import decomposition
    from sklearn.cluster import KMeans

    print(y)
    y1 = y

    # colors2 covers the y1/predicted-label classes (sized by `n`); colors3
    # covers the KMeans cluster centers, which must be sized by `n2` (the
    # actual number of clusters) -- these can legitimately differ.
    colors2 = _distinct_colors(n)
    colors3 = _distinct_colors(n2)[::-1]

    if M == "pca" or M == "pc-km":
        np.random.seed(1)

        pca = decomposition.PCA(n_components=3)
        pca.fit(x)
        X = pca.transform(x)

        # Dump components relations with features:
        w = pd.DataFrame(
            pca.components_, columns=x.columns, index=["PC-1", "PC-2", "PC-3"]
        )
        w = w.transpose()
        w.to_csv(path + "pc-weights.csv")

        plt.figure()
        plt.title("PCA", fontsize=16)
        plt.xlabel("PC1")
        plt.ylabel("PC2")
        plt.scatter(
            X[:, 0], X[:, 1], c=y1, cmap=matplotlib.colors.ListedColormap(colors2)
        )
        plt.show()
        plt.figure()
        plt.title("PCA", fontsize=16)
        plt.xlabel("PC1")
        plt.ylabel("PC3")
        plt.scatter(
            X[:, 0], X[:, 2], c=y1, cmap=matplotlib.colors.ListedColormap(colors2)
        )
        plt.show()
        plt.figure()
        plt.title("PCA", fontsize=16)
        plt.xlabel("PC2")
        plt.ylabel("PC3")
        plt.scatter(
            X[:, 1], X[:, 2], c=y1, cmap=matplotlib.colors.ListedColormap(colors2)
        )
        plt.show()

        fig = plt.figure(1, figsize=(4, 3))
        plt.clf()
        ax = fig.add_subplot(111, projection="3d", elev=48, azim=134)
        ax.scatter(
            X[:, 0],
            X[:, 1],
            X[:, 2],
            c=y1,
            cmap=matplotlib.colors.ListedColormap(colors2),
        )
        ax.xaxis.set_ticklabels([])
        ax.yaxis.set_ticklabels([])
        ax.zaxis.set_ticklabels([])
        plt.xlabel("PC1")
        plt.ylabel("PC3")
        plt.show()

        if M == "pca":
            return [X[:, 0], X[:, 1], X[:, 2]]

    if M == "kmeans" or M == "pc-km":
        if M == "pc-km":
            kmeans = KMeans(n_clusters=n2, random_state=1).fit(X)
        elif M == "kmeans":
            kmeans = KMeans(n_clusters=n2, random_state=1).fit(x)
        labels = kmeans.labels_
        centers = kmeans.cluster_centers_

        # Canonicalize cluster labels so that "1" is always the majority class.
        labelslist = labels.tolist()
        if labelslist.count(1) <= len(labels) / 2:
            for i in range(len(labels)):
                if labels[i] == 1:
                    labels[i] = 0
                elif labels[i] == 0:
                    labels[i] = 1

        X_array = np.array(x)

        if M == "kmeans":
            if v1 and v2 and v_names:
                va = v_names.index(v1)
                vb = v_names.index(v2)
            else:
                va = 0
                vb = 1
            plt.figure()
            plt.title("K-Means clustering (predicted classes)", fontsize=14)
            plt.xlabel(str(v1), fontsize=10)
            plt.ylabel(str(v2), fontsize=10)
            plt.scatter(
                X_array[:, va],
                X_array[:, vb],
                c=labels,
                cmap=matplotlib.colors.ListedColormap(colors2),
                alpha=0.6,
            )
            plt.scatter(
                centers[:, va],
                centers[:, vb],
                c=colors3,
                s=300,
                alpha=1,
                marker="P",
            )
            plt.show()
            plt.figure()
            plt.title("K-Means clustering (actual classes)", fontsize=14)
            plt.xlabel(str(v1), fontsize=10)
            plt.ylabel(str(v2), fontsize=10)
            plt.scatter(
                X_array[:, va],
                X_array[:, vb],
                c=y1,
                cmap=matplotlib.colors.ListedColormap(colors2),
                alpha=0.6,
            )
            plt.scatter(
                centers[:, va],
                centers[:, vb],
                c=colors3,
                s=300,
                alpha=1,
                marker="P",
            )
            plt.show()
        elif M == "pc-km":
            plt.figure()
            plt.title("K-Means clustering on PC (predicted labels)", fontsize=14)
            plt.xlabel("Variable 1")
            plt.ylabel("Variable 2")
            plt.scatter(
                centers[:, 0], centers[:, 1], c=colors3, s=300, alpha=0.5, marker="P"
            )
            plt.scatter(
                X[:, 0],
                X[:, 1],
                c=labels,
                cmap=matplotlib.colors.ListedColormap(colors2),
                alpha=0.6,
            )
            plt.show()
            plt.figure()
            plt.title("K-Means clustering on PC (actual labels)", fontsize=14)
            plt.scatter(
                centers[:, 0], centers[:, 1], c=colors3, s=300, alpha=0.5, marker="P"
            )
            plt.scatter(
                X[:, 0],
                X[:, 1],
                c=y1,
                cmap=matplotlib.colors.ListedColormap(colors2),
                alpha=0.6,
            )
            plt.show()

        if M == "kmeans":
            # Only "kmeans" actually uses this confusion matrix (plotted
            # below); computing it unconditionally for "pc-km" too was dead
            # work that also assumed exactly 2 clusters (cnf.ravel() into
            # tp/fp/fn/tn), crashing for any other n2.
            cnf = confusion_matrix(y1, labels)
            # sklearn's confusion_matrix(...).ravel() for binary labels [0, 1]
            # returns [tn, fp, fn, tp] -- NOT [tp, fp, fn, tn]. Unpacking in
            # the wrong order silently swaps which count is called "tp" vs
            # "tn" (fp/fn happen to land correctly, same position either way).
            tn, fp, fn, tp = cnf.ravel()
            cnf[0][0] = tp
            cnf[1][1] = tn
            cnf[0][1] = fn
            cnf[1][0] = fp

            ytest_labels = kmeans.predict(xtest)
            ytest_labelslist = ytest_labels.tolist()
            if ytest_labelslist.count(1) <= len(ytest_labels) / 2:
                for i in range(len(ytest_labels)):
                    if ytest_labels[i] == 1:
                        ytest_labels[i] = 0
                    elif ytest_labels[i] == 0:
                        ytest_labels[i] = 1

            cnf3 = confusion_matrix(ytest, ytest_labels)
            tnt, fpt, fnt, tpt = cnf3.ravel()
            cnf3[0][0] = tpt
            cnf3[1][1] = tnt
            cnf3[0][1] = fnt
            cnf3[1][0] = fpt

            plt.figure()
            plot_confusion_matrix(
                cnf, classes=["N1", "N0"], title="Confusion matrix - K-means"
            )
            plt.ylabel("Predicted")
            plt.xlabel("Actual")
            plt.show()


def Model(x, y, xtest, ytest, v_names, c=10, M="mlr", rs=None, cv="loo"):
    """Fit a regression model and evaluate it on train/test/cross-validation.

    Parameters
    ----------
    x, y : training feature matrix and target.
    xtest, ytest : held-out feature matrix and target.
    v_names : list[str]
        Feature names (used for variable-importance reporting).
    c : int, default 10
        Model-specific hyperparameter (n_components for "pls",
        n_estimators/max_depth for "rf", C for "svm", max_depth for
        "tree", ...).
    M : str, default "mlr"
        Which regressor to fit. One of: pls, mlr, rf, svm, lsvm, lasso,
        nn, tree, rg, el, la, ll, or, brg, ardr, ransa, the, hub, sgdr,
        kn, gu, ex, bg, gb, ada.
    rs : int or None
        Random state, where the estimator supports one.
    cv : {"loo", "kf", "kfr", "off"}, default "loo"
        Cross-validation strategy for the reported CV metrics ("off"
        skips cross-validation; ``List``/``analysis`` are then ``None``).

    Returns
    -------
    list
        ``[result, List, model, analysis]`` -- a metrics dict, CV metrics
        dict (or None), the fitted model, and the full analyse() report
        (or None).
    """
    ytests = []
    ypreds = []
    X_array = np.array(x)
    y_array = np.array(y)

    if M == "pls":
        model = PLSRegression(n_components=c)
        model2 = PLSRegression(n_components=c)
    elif M == "mlr":
        model = LinearRegression()
        model2 = LinearRegression()
    elif M == "rf":
        model = RandomForestRegressor(
            n_estimators=c, max_depth=c - 1, random_state=rs, max_features=c
        )
        model2 = RandomForestRegressor(
            n_estimators=c, max_depth=c - 1, random_state=rs, max_features=c
        )
    elif M == "svm":
        model = SVR(kernel="rbf", gamma="auto", C=c)
        model2 = SVR(kernel="rbf", gamma="auto", C=c)
    elif M == "lsvm":
        model = LinearSVR(random_state=rs)
        model2 = LinearSVR(random_state=rs)
    elif M == "lasso":
        model = Lasso(alpha=0.1, random_state=rs)
        model2 = Lasso(alpha=0.1, random_state=rs)
    elif M == "nn":
        model = MLPRegressor(
            max_iter=200,
            hidden_layer_sizes=[20, 10, 10],
            random_state=rs,
            warm_start=False,
            alpha=0.0001,
            solver="adam",
        )
        model2 = MLPRegressor(
            max_iter=200,
            hidden_layer_sizes=[20, 10, 10],
            random_state=rs,
            warm_start=False,
            alpha=0.0001,
            solver="adam",
        )
    elif M == "tree":
        model = DecisionTreeRegressor(max_depth=c, max_features=c + 1, random_state=rs)
        model2 = DecisionTreeRegressor(max_depth=c, max_features=c + 1, random_state=rs)
    elif M == "rg":
        model = Ridge(random_state=rs)
        model2 = Ridge(random_state=rs)
    elif M == "el":
        model = ElasticNet(random_state=rs)
        model2 = ElasticNet(random_state=rs)
    elif M == "la":
        model = Lars()
        model2 = Lars()
    elif M == "ll":
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
        model = KNeighborsRegressor()
        model2 = KNeighborsRegressor()
    elif M == "gu":
        model = GaussianProcessRegressor(random_state=rs)
        model2 = GaussianProcessRegressor(random_state=rs)
    elif M == "ex":
        model = ExtraTreesRegressor(random_state=rs)
        model2 = ExtraTreesRegressor(random_state=rs)
    elif M == "bg":
        model = BaggingRegressor(random_state=rs)
        model2 = BaggingRegressor(random_state=rs)
    elif M == "gb":
        model = GradientBoostingRegressor(random_state=rs)
        model2 = GradientBoostingRegressor(random_state=rs)
    elif M == "ada":
        model = AdaBoostRegressor(random_state=rs)
        model2 = AdaBoostRegressor(random_state=rs)
    else:
        raise ValueError(f"unknown model name M={M!r}")

    model.fit(x, y)
    y_predict_train = model.predict(x)
    y_predict_test = model.predict(xtest)
    r2 = model.score(x, y)
    R2test = r2test(ytest, y_predict_test, y)
    Pearson = stats.pearsonr(ytest, y_predict_test)
    q2f2 = r2test(ytest, y_predict_test, ytest)
    model_mse_test = mean_squared_error(y_predict_test, ytest)
    f = 0

    List = None
    if cv != "off":
        if cv == "loo":
            loo = LeaveOneOut()
        elif cv == "kf":
            loo = KFold(n_splits=5, random_state=rs, shuffle=True)
        elif cv == "kfr":
            loo = RepeatedKFold(n_splits=2, n_repeats=2, random_state=rs)
        for train_idx, test_idx in loo.split(x):
            X_train, X_test = X_array[train_idx], X_array[test_idx]
            y_train, y_test = y_array[train_idx], y_array[test_idx]
            model2.fit(X_train, y_train)
            y_pred = model2.predict(X_test)
            # There is only one y_test/y_pred per CV iteration, so accumulate
            # them across folds to get a proper overall CV metric/graph.
            ytests += list(y_test)
            ypreds += list(y_pred)

        q2 = q2r2(ytests, ypreds)
        rmse = RMSEP_CV_C(ytests, ypreds)
        List = {"q2": q2, "RMSECV": rmse, "Q2F2": q2f2}

    if M == "svm":
        sorted_VI = ""
    else:
        try:
            if M == "nn":
                VI = model.coefs_
                VI = np.dot(VI[0], VI[1])
            elif M in ("rf", "tree", "ex", "gb", "ada"):
                VI = model.feature_importances_
            elif M == "pls":
                # PLSRegression.coef_ is (n_targets, n_features), the
                # opposite orientation from the other linear models here.
                VI = model.coef_.reshape(-1)
            else:
                VI = model.coef_
            VI = pd.DataFrame(data=VI, index=v_names)
            sorted_VI = VI.sort_values(by=[0])
        except AttributeError:
            # Some estimators (e.g. kn, gu, bg) expose neither coef_ nor
            # feature_importances_ -- no variable importance available.
            sorted_VI = ""

    if M == "mlr":
        result = {
            "R2": r2,
            "Intercept": model.intercept_,
            "Mean_squared_error_test": model_mse_test,
            "R2_test": R2test,
            "F": f,
            "Coefficients": model.coef_,
        }
    else:
        result = {
            "R2": r2,
            "Mean_squared_error_test": model_mse_test,
            "R2_test": R2test,
            "F": f,
            "Variable Importance": sorted_VI,
            "Pearson": Pearson,
        }

    y1 = pd.DataFrame(y, index=x.index, columns=["observed"])
    y2 = pd.DataFrame(y_predict_train, index=x.index, columns=["predicted"])
    y3 = pd.concat([y1, y2], axis=1)
    y3.to_csv("train.csv")
    y4 = pd.DataFrame(ytest, index=xtest.index, columns=["observed"])
    y5 = pd.DataFrame(y_predict_test, index=xtest.index, columns=["predicted"])
    y6 = pd.concat([y4, y5], axis=1)
    y6.to_csv("test.csv")

    analysis = None
    if cv != "off":
        analysis = analyse(
            y, y_predict_train, ytest, y_predict_test, ytests, ypreds, k=len(v_names)
        )
    return [result, List, model, analysis]


def _build_dl_model(n_features, dl2, dp, omp, lr1, nesterov):
    from keras import Sequential, optimizers
    from keras.layers import Dense, Dropout, Input

    model = Sequential()
    model.add(Input(shape=(n_features,)))
    for units in dl2:
        model.add(Dense(units, kernel_initializer="uniform", activation="relu"))
        model.add(Dropout(dp))
    model.add(Dense(1, kernel_initializer="normal", activation="sigmoid"))

    if omp == "sgd":
        optimizer = optimizers.SGD(learning_rate=lr1, momentum=0.9, nesterov=nesterov)
    else:
        optimizer = optimizers.Adam(learning_rate=lr1)
    model.compile(loss="binary_crossentropy", optimizer=optimizer, metrics=["accuracy"])
    return model


def ModelC(
    x,
    y,
    xtest,
    ytest,
    v_names,
    c1=10,
    M="tree",
    rs=None,
    cv="loo",
    c2=10,
    c3="rbf",
    c5=200,
    ep=300,
    dl2=(500, 1000),
    lr1=0.01,
    nesterov=True,
    omp="adam",
    dp=0.2,
    bs=16,
    c4="adam",
    grid=False,
    weights="balanced",
):
    """Fit a binary classifier and evaluate it on train/test/cross-validation.

    Parameters
    ----------
    x, y : training feature matrix and target.
    xtest, ytest : held-out feature matrix and target.
    v_names : list[str]
        Feature names (used for variable-importance reporting).
    M : str, default "tree"
        Which classifier to fit. One of: tree, nn, rf, ex, lsvm, svm,
        lr, ld, rg, per, pass, qua, sgdc, kn, rn, gu, gunb, cnb, bg, gb,
        ada, dl (a small Keras MLP). (Lasso/KernelRidge are regression-only
        models and aren't offered here -- see Model() for those.)
    cv : {"loo", "kf", "kfr", "shuff"}
        Cross-validation strategy for the CV-based accuracy report.
    c1, c2, c3, c5, weights
        Model-specific hyperparameters (see the branch for each `M`).
    ep, dl2, lr1, nesterov, omp, dp, bs
        Hyperparameters for M="dl": epochs, hidden layer sizes, learning
        rate, SGD nesterov momentum, optimizer ("adam"/"sgd"), dropout
        rate, batch size.

    Returns
    -------
    list
        ``[result, model]``.
    """
    ytests = []
    ypreds = []
    X_array = np.array(x)
    y_array = np.array(y)

    if M == "tree":
        model = DecisionTreeClassifier(random_state=rs, max_depth=c1, max_features=c2)
        model2 = DecisionTreeClassifier(random_state=rs, max_depth=c1, max_features=c2)
    elif M == "nn":
        model = MLPClassifier(
            solver=c4,
            alpha=0.0001,
            hidden_layer_sizes=(c1, c1, c1),
            random_state=rs,
            max_iter=c5,
        )
        model2 = MLPClassifier(
            solver=c4,
            alpha=1e-4,
            hidden_layer_sizes=(c1, c1, c1),
            random_state=rs,
            max_iter=c5,
        )
    elif M == "rf":
        model = RandomForestClassifier(
            max_depth=c1,
            n_estimators=c1 * 2,
            random_state=rs,
            max_features="sqrt",
            max_leaf_nodes=None,
            class_weight=weights,
        )
        model2 = RandomForestClassifier(
            max_depth=c1,
            n_estimators=c1 * 2,
            random_state=rs,
            max_features="sqrt",
            max_leaf_nodes=None,
            class_weight=weights,
        )
    elif M == "ex":
        model = ExtraTreesClassifier(max_depth=c1, random_state=rs)
        model2 = ExtraTreesClassifier(max_depth=c1, random_state=rs)
    elif M == "lsvm":
        model = LinearSVC(class_weight=weights, C=c1, random_state=rs)
        model2 = LinearSVC(class_weight=weights, C=c1, random_state=rs)
    elif M == "svm":
        model = SVC(
            class_weight=weights,
            random_state=rs,
            probability=True,
            C=c1,
            kernel=c3,
            max_iter=c5,
            gamma=c2,
        )
        model2 = SVC(
            class_weight=weights,
            random_state=rs,
            probability=True,
            C=c1,
            kernel=c3,
            max_iter=c5,
            gamma=c2,
        )
    elif M == "lr":
        model = LogisticRegression(max_iter=c1, random_state=rs, class_weight=weights)
        model2 = LogisticRegression(max_iter=c1, random_state=rs, class_weight=weights)
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
        model = PassiveAggressiveClassifier(random_state=rs)
        model2 = PassiveAggressiveClassifier(random_state=rs)
    elif M == "qua":
        model = QuadraticDiscriminantAnalysis()
        model2 = QuadraticDiscriminantAnalysis()
    elif M == "sgdc":
        model = SGDClassifier(random_state=rs)
        model2 = SGDClassifier(random_state=rs)
    elif M == "kn":
        model = KNeighborsClassifier(n_neighbors=c1)
        model2 = KNeighborsClassifier(n_neighbors=c1)
    elif M == "rn":
        model = RadiusNeighborsClassifier()
        model2 = RadiusNeighborsClassifier()
    elif M == "gu":
        model = GaussianProcessClassifier(random_state=rs)
        model2 = GaussianProcessClassifier(random_state=rs)
    elif M == "gunb":
        model = GaussianNB()
        model2 = GaussianNB()
    elif M == "cnb":
        model = ComplementNB(alpha=c1)
        model2 = ComplementNB(alpha=c1)
    elif M == "bg":
        model = BaggingClassifier(random_state=rs)
        model2 = BaggingClassifier(random_state=rs)
    elif M == "gb":
        model = GradientBoostingClassifier(random_state=rs)
        model2 = GradientBoostingClassifier(random_state=rs)
    elif M == "ada":
        model = AdaBoostClassifier(random_state=rs)
        model2 = AdaBoostClassifier(random_state=rs)
    elif M == "dl":
        from sklearn.utils.class_weight import compute_class_weight

        np.random.seed(1)
        class_weight_values = compute_class_weight(
            "balanced", classes=np.unique(y_array), y=y_array
        )
        class_weights = dict(enumerate(class_weight_values))
        model = _build_dl_model(len(v_names), dl2, dp, omp, lr1, nesterov)
    else:
        raise ValueError(f"unknown model name M={M!r}")

    if M == "dl":
        from keras.callbacks import EarlyStopping
        from sklearn.model_selection import StratifiedKFold

        loo = StratifiedKFold(n_splits=5)
        for train_idx, test_idx in loo.split(x, y):
            X_train, X_test = X_array[train_idx], X_array[test_idx]
            y_train, y_test = y_array[train_idx], y_array[test_idx]

            model2 = _build_dl_model(len(v_names), dl2, dp, omp, lr1, nesterov)
            early_stopping = EarlyStopping(
                monitor="val_loss", patience=5, restore_best_weights=True
            )
            model2.fit(
                X_train,
                y_train,
                batch_size=bs,
                epochs=ep,
                validation_data=(X_test, y_test),
                callbacks=[early_stopping],
                verbose=0,
                class_weight=class_weights,
            )
            y_pred = (model2.predict(X_test, verbose=0) > 0.5).astype("int32")
            ytests += list(y_test)
            ypreds += list(np.ravel(y_pred))

        early_stopping = EarlyStopping(
            monitor="loss", patience=5, restore_best_weights=True
        )
        model.fit(
            x,
            y,
            batch_size=bs,
            epochs=ep,
            callbacks=[early_stopping],
            verbose=0,
            class_weight=class_weights,
        )

        y_predict_train = (model.predict(x, verbose=0) > 0.5).astype("int32").ravel()
        y_predict_test = (model.predict(xtest, verbose=0) > 0.5).astype("int32").ravel()
    else:
        model.fit(x, y)
        y_predict_train = model.predict(x)
        y_predict_test = model.predict(xtest)

    accuracy_score_train = accuracy_score(y, y_predict_train, normalize=True)
    accuracy_score_test = accuracy_score(ytest, y_predict_test, normalize=True)

    if M != "dl":
        ni = 2000
        if cv == "loo":
            loo = LeaveOneOut()
        elif cv == "kf":
            loo = KFold(n_splits=5, shuffle=True, random_state=rs)
        elif cv == "kfr":
            loo = RepeatedKFold(n_splits=2, n_repeats=2, random_state=rs)
        elif cv == "shuff":
            loo = ShuffleSplit(n_splits=ni, test_size=0.2, random_state=0)
        n = 0
        ytests2 = []
        ypreds2 = []
        for train_idx, test_idx in loo.split(x):
            X_train, X_test = X_array[train_idx], X_array[test_idx]
            y_train, y_test = y_array[train_idx], y_array[test_idx]
            model2.fit(X_train, y_train)
            y_pred = model2.predict(X_test)

            if n == 0:
                ytests2 = [list(y_test)]
                ypreds2 = [list(y_pred)]
            else:
                ytests2 = ytests2 + [list(y_test)]
                ypreds2 = ypreds2 + [list(y_pred)]
            # There is only one y_test/y_pred per CV iteration, so accumulate
            # them across folds to get a proper overall CV metric/graph.
            ytests += list(y_test)
            ypreds += list(y_pred)
            n = n + 1
            accuracy_score_LOO = accuracy_score(ytests, ypreds, normalize=True)
        if cv == "shuff":
            accuracy_score_LOO_all = [None] * ni
            for i in range(ni):
                accuracy_score_LOO_all[i] = accuracy_score(
                    ytests2[i], ypreds2[i], normalize=True
                )
            print(accuracy_score_LOO)
            print(np.mean(accuracy_score_LOO_all), "<---- Mean of the accuracies")
            print(np.std(accuracy_score_LOO_all), "<---- STD of the accuracies")
    else:
        accuracy_score_LOO = ""

    if M == "tree":
        VI = model.feature_importances_
        try:
            import graphviz
            from sklearn.tree import export_graphviz

            dot_data = export_graphviz(
                model,
                out_file=None,
                max_depth=None,
                feature_names=v_names,
                class_names=["1", "2", "3", "4"],
                label="root",
                impurity=True,
                proportion=False,
                rounded=True,
                precision=2,
            )
            graphviz.Source(dot_data).render("tree_chart")
        except Exception as exc:  # noqa: BLE001
            # Optional visualization; many possible failure modes
            # (missing graphviz package, missing system binary, ...).
            print(f"could not render decision tree diagram: {exc}")
    elif M in ("rf", "ex", "gb", "ada"):
        VI = model.feature_importances_
    elif M == "svm":
        if c3 == "linear":
            VI = model.coef_[0]
        else:
            VI = ""
    elif M == "nn":
        VI = model.coefs_
        VI = np.dot(VI[0], VI[1])
    elif M in ("qua", "kn", "gu", "bg", "rn") or M == "gunb" or M == "cnb" or M == "dl":
        VI = ""
    elif M == "rg":
        # Unlike the other linear classifiers here, RidgeClassifier's
        # coef_ for binary classification is already 1D (n_features,),
        # not (1, n_features) -- indexing [0] would grab a single
        # coefficient instead of the whole vector.
        VI = model.coef_
    else:
        VI = model.coef_[0]

    if (
        M in ("qua", "kn", "gu", "bg", "rn", "dl")
        or M == "gunb"
        or M == "cnb"
        or M == "svm"
        and c3 != "linear"
    ):
        sorted_VI = ""
    else:
        VI = pd.DataFrame(data=VI, index=v_names)
        sorted_VI = VI.sort_values(by=[0])

    if M in (
        "tree",
        "ld",
        "lr",
        "nn",
        "rf",
        "ex",
        "gb",
        "ada",
        "svm",
        "qua",
        "kn",
        "gu",
        "gunb",
        "cnb",
        "bg",
        "rn",
    ):
        pra_train = model.predict_proba(x)
        pra_test = model.predict_proba(xtest)
    elif M == "dl":
        pra_train = model.predict(x, verbose=0)
        pra_test = model.predict(xtest, verbose=0)
    else:
        pra_train = ""
        pra_test = ""

    cnf = confusion_matrix(y, y_predict_train)
    cnf3 = confusion_matrix(ytests, ypreds)  # CV
    cnf2 = confusion_matrix(ytest, y_predict_test)

    # Binary classification only: reorder the 2x2 confusion matrix so
    # cnf[0][0]/[1][1] are TP/TN and cnf[0][1]/[1][0] are FN/FP.
    #
    # sklearn's confusion_matrix(...).ravel() for binary labels [0, 1] returns
    # [tn, fp, fn, tp] -- NOT [tp, fp, fn, tn]. Unpacking in the wrong order
    # silently swaps which count is called "tp" vs "tn" below (fp/fn happen
    # to land correctly, same position either way); every metr() call fed by
    # these was reporting swapped sensitivity/specificity/precision/F as a
    # result, even though accuracy and MCC are symmetric under the swap and
    # so came out right regardless.
    tn, fp, fn, tp = cnf.ravel()
    cnf[0][0] = tp
    cnf[1][1] = tn
    cnf[0][1] = fn
    cnf[1][0] = fp
    tnt, fpt, fnt, tpt = cnf2.ravel()
    cnf2[0][0] = tpt
    cnf2[1][1] = tnt
    cnf2[0][1] = fnt
    cnf2[1][0] = fpt
    tnc, fpc, fnc, tpc = cnf3.ravel()
    cnf3[0][0] = tpc
    cnf3[1][1] = tnc
    cnf3[0][1] = fnc
    cnf3[1][0] = fpc

    trainmetrics = metr(tp, tn, fp, fn)
    cvmetrics = metr(tpc, tnc, fpc, fnc)
    testmetrics = metr(tpt, tnt, fpt, fnt)
    total = metr(tpt + tp, tnt + tn, fpt + fp, fnt + fn)

    plt.figure()
    plot_confusion_matrix(cnf, classes=["Active", "Inactive"], title="Train set")
    plt.ylabel("Predicted", fontsize=12, fontweight="bold")
    plt.xlabel("Actual", fontsize=12, fontweight="bold")
    plt.figure()
    plot_confusion_matrix(
        cnf3, classes=["Active", "Inactive"], title="Cross-Validation"
    )
    plt.ylabel("Predicted", fontsize=12, fontweight="bold")
    plt.xlabel("Actual", fontsize=12, fontweight="bold")
    plt.figure()
    plot_confusion_matrix(cnf2, classes=["Active", "Inactive"], title="Test set")
    plt.ylabel("Predicted", fontsize=12, fontweight="bold")
    plt.xlabel("Actual", fontsize=12, fontweight="bold")

    y1 = pd.DataFrame(y, index=x.index, columns=["observed"])
    y2 = pd.DataFrame(y_predict_train, index=x.index, columns=["predicted"])
    y3 = pd.concat([y1, y2], axis=1)
    y3.to_csv("train.csv")
    y4 = pd.DataFrame(ytest, index=xtest.index, columns=["observed"])
    y5 = pd.DataFrame(y_predict_test, index=xtest.index, columns=["predicted"])
    y6 = pd.concat([y4, y5], axis=1)
    y6.to_csv("test.csv")

    result = {
        "y1": y1,
        "y2": y2,
        "y3": y3,
        "y4": y4,
        "y5": y5,
        "y6": y6,
        "confusion matrix": cnf,
        "accuracy_score_train": accuracy_score_train,
        "accuracy_score_test": accuracy_score_test,
        "accuracy_score_LOO": accuracy_score_LOO,
        "Variable Importance": sorted_VI,
        "probability_train": pra_train,
        "probability_test": pra_test,
        "confusion matrix_CV": cnf3,
        "confusion matrix_test": cnf2,
        "train_metrics": trainmetrics,
        "CV_metrics": cvmetrics,
        "test_metrics": testmetrics,
        "total_metrics": total,
        "VI": VI,
        "train_AC": trainmetrics["AC"],
        "train_SEN": trainmetrics["SEN"],
        "train_SPEC": trainmetrics["SPEC"],
        "train_PREC": trainmetrics["PREC"],
        "train_F": trainmetrics["F"],
        "train_MCC": trainmetrics["MCC"],
        "test_AC": testmetrics["AC"],
        "test_SEN": testmetrics["SEN"],
        "test_SPEC": testmetrics["SPEC"],
        "test_PREC": testmetrics["PREC"],
        "test_F": testmetrics["F"],
        "test_MCC": testmetrics["MCC"],
        "total_AC": total["AC"],
        "total_SEN": total["SEN"],
        "total_SPEC": total["SPEC"],
        "total_PREC": total["PREC"],
        "total_F": total["F"],
        "total_MCC": total["MCC"],
    }
    return [result, model]


def plot_confusion_matrix(
    cm, classes, normalize=False, title="Confusion matrix", cmap=plt.cm.Blues
):
    """Plot a confusion matrix (optionally row-normalized)."""
    if normalize:
        cm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]

    plt.imshow(cm, interpolation="nearest", cmap=cmap)
    plt.title(title, fontsize=26, fontweight="bold")
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45, fontsize=12)
    plt.yticks(tick_marks, classes, fontsize=12)

    fmt = ".2f" if normalize else "d"
    thresh = cm.max() / 2.0
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(
            j,
            i,
            format(cm[i, j], fmt),
            horizontalalignment="center",
            color="white" if cm[i, j] > thresh else "black",
            fontsize=30,
            fontweight="bold",
        )

    plt.tight_layout()
    plt.ylabel("True label")
    plt.xlabel("Predicted label")


def plot_w(x, y) -> None:
    plt.bar(x, y)
    plt.xticks(rotation=90)
    plt.show()


def rocc(
    x, y, ycat="", title="Receiver operating characteristic", tsize=12, pos_l=1, l="b"
):
    """Plot an ROC curve and print early-recognition enrichment factors (EF1/2/10/20/50)."""

    def COUNT(col, label, percent=1):
        n = 0
        t = int(len(col) * percent)
        for i in range(t):
            if col.iloc[i] == label:
                n = n + 1
        return n

    total = float(len(y))
    baseline = COUNT(y, l, percent=1) / total
    ef_percents = {"EF1": 0.01, "EF2": 0.02, "EF10": 0.1, "EF20": 0.2, "EF50": 0.5}
    for name, percent in ef_percents.items():
        rate = COUNT(y, l, percent=percent) / (total * percent)
        print(rate / baseline, f" --> {name}")

    if ycat == "cat":
        y = np.where(np.asarray(y) <= 1.1, 1, 0)

    fpr, tpr, _ = metrics.roc_curve(y, x, pos_label=pos_l)
    auc = metrics.auc(fpr, tpr)
    print(auc)
    plt.figure()
    lw = 2
    plt.plot(
        fpr, tpr, color="darkorange", lw=lw, label=f"ROC curve (area = {auc:0.2f})"
    )
    plt.plot([0, 1], [0, 1], color="navy", lw=lw, linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(title, fontsize=tsize)
    plt.legend(loc="lower right")
    plt.show()


def safe_divide(numerator, denominator):
    try:
        return numerator / denominator
    except ZeroDivisionError:
        return 0


def metr(tp1, tn1, fp1, fn1) -> dict:
    """Compute accuracy/sensitivity/specificity/precision/F-measure/MCC from confusion counts."""
    tp = float(tp1)
    tn = float(tn1)
    fp = float(fp1)
    fn = float(fn1)
    accuracy_m = float(safe_divide(tp + tn, tp + tn + fp + fn))
    precision_m = safe_divide(tp, tp + fp)
    sensitivity_m = safe_divide(tp, tp + fn)  # recall
    specificity_m = safe_divide(tn, tn + fp)
    f_measure_m = safe_divide(
        2 * sensitivity_m * precision_m, sensitivity_m + precision_m
    )
    mcc = safe_divide(
        (tp * tn) - (fp * fn), math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    )
    return {
        "AC": accuracy_m,
        "SEN": sensitivity_m,
        "SPEC": specificity_m,
        "PREC": precision_m,
        "F": f_measure_m,
        "MCC": mcc,
    }
