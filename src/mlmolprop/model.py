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
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
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
from sklearn.metrics import accuracy_score, confusion_matrix, mean_squared_error, r2_score
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
from sklearn.utils.class_weight import compute_sample_weight

from .basic import RMSEP_CV_C, F, analyse, q2r2, r2test


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
            # `labels` is always canonicalized to {0, 1} above, but the
            # actual target y1/ytest can be any two-valued encoding (e.g.
            # a raw {1.0, 2.0} stage column) -- comparing those directly
            # against {0, 1} cluster labels makes confusion_matrix treat
            # them as 3+ distinct classes (union of both label sets)
            # instead of a 2x2 table, so cnf.ravel() below has more than
            # 4 entries. Map the two actual-target values onto {0, 1}
            # first so both sides of the comparison use the same scale.
            def _to_binary(values):
                values = np.asarray(values)
                uniq = np.unique(values)
                if len(uniq) != 2:
                    return values
                return np.where(values == uniq[1], 1, 0)

            y1_bin = _to_binary(y1)

            # Only "kmeans" actually uses this confusion matrix (plotted
            # below); computing it unconditionally for "pc-km" too was dead
            # work that also assumed exactly 2 clusters (cnf.ravel() into
            # tp/fp/fn/tn), crashing for any other n2.
            cnf = confusion_matrix(y1_bin, labels)
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

            cnf3 = confusion_matrix(_to_binary(ytest), ytest_labels)
            tnt, fpt, fnt, tpt = cnf3.ravel()
            cnf3[0][0] = tpt
            cnf3[1][1] = tnt
            cnf3[0][1] = fnt
            cnf3[1][0] = fpt

            plt.figure(figsize=(8, 7))
            plot_confusion_matrix(
                cnf, classes=["N1", "N0"], title="Confusion matrix - K-means"
            )
            plt.ylabel("Actual", fontsize=24, fontweight="bold")
            plt.xlabel("Predicted", fontsize=24, fontweight="bold")
            plt.tight_layout()
            plt.show()


def Model(x, y, xtest, ytest, v_names, params=None, M="mlr", rs=None, cv="loo", path="./"):
    """Fit a regression model and evaluate it on train/test/cross-validation.

    Parameters
    ----------
    x, y : training feature matrix and target.
    xtest, ytest : held-out feature matrix and target.
    v_names : list[str]
        Feature names (used for variable-importance reporting).
    path : str, default "./"
        Directory prefix for output files (``"train.csv"``, ``"test.csv"``) --
        same convention as :func:`clus_uns`'s ``path`` parameter.
    params : dict or None
        Model-specific hyperparameters, passed straight through as
        keyword arguments to the underlying scikit-learn estimator's
        constructor (layered on top of per-model defaults chosen to
        match this function's previous fixed behavior) -- e.g.
        ``{"n_estimators": 200, "max_depth": 8, "max_features": 5}``
        for M="rf", or ``{"n_components": 5}`` for M="pls". Any keyword
        the real estimator accepts works, which is what makes this
        usable as a generic hyperparameter-search wrapper across model
        families: build a per-model param grid using scikit-learn's own
        parameter names, no per-model translation layer needed.
    M : str, default "mlr"
        Which regressor to fit. One of: pls, mlr, rf, svm, lsvm, lasso,
        nn, tree, rg, el, la, ll, or, brg, ardr, ransa, the, hub, sgdr,
        kn, gu, ex, bg, gb, hgb, xgb, ada, dl (a small Keras MLP,
        linear output/MSE loss). M="xgb" requires the optional
        "xgboost" extra (see :func:`_import_xgboost`); M="dl" requires
        the optional "dl" extra. For M="dl" (which has no scikit-learn
        constructor), the recognized ``params`` keys are instead
        ``epochs``, ``hidden_layer_sizes``, ``learning_rate``,
        ``nesterov``, ``optimizer`` ("adam"/"sgd"), ``dropout``, ``l2``
        (L2 weight-regularization strength, default 0.0 -- no penalty), and
        ``batch_size`` -- same keys as ``ModelC()``'s M="dl". Unlike
        every other M here, M="dl" always internally runs its own
        5-fold CV to fit ``model``/populate ``List`` regardless of
        ``cv`` (only whether ``List`` is actually returned depends on
        ``cv != "off"``) -- matches ``ModelC()``'s M="dl" behavior.
    rs : int or None
        Random state, where the estimator supports one.
    cv : {"loo", "kf", "kfr", "off"}, default "loo"
        Cross-validation strategy for the reported CV metrics ("off"
        skips cross-validation; ``List``/``analysis`` are then ``None``).
        Ignored for M="dl", which always uses its own internal 5-fold CV.

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
    p = dict(params or {})

    if M == "pls":
        mp = {"n_components": 10, **p}
        model = PLSRegression(**mp)
        model2 = PLSRegression(**mp)
    elif M == "mlr":
        model = LinearRegression(**p)
        model2 = LinearRegression(**p)
    elif M == "rf":
        mp = {"n_estimators": 10, "max_depth": 9, "max_features": 10, **p}
        model = RandomForestRegressor(random_state=rs, **mp)
        model2 = RandomForestRegressor(random_state=rs, **mp)
    elif M == "svm":
        mp = {"C": 10, "kernel": "rbf", "gamma": "auto", **p}
        model = SVR(**mp)
        model2 = SVR(**mp)
    elif M == "lsvm":
        model = LinearSVR(random_state=rs, **p)
        model2 = LinearSVR(random_state=rs, **p)
    elif M == "lasso":
        mp = {"alpha": 0.1, **p}
        model = Lasso(random_state=rs, **mp)
        model2 = Lasso(random_state=rs, **mp)
    elif M == "nn":
        mp = {
            "max_iter": 200,
            "hidden_layer_sizes": [20, 10, 10],
            "alpha": 0.0001,
            "solver": "adam",
            "warm_start": False,
            **p,
        }
        model = MLPRegressor(random_state=rs, **mp)
        model2 = MLPRegressor(random_state=rs, **mp)
    elif M == "tree":
        mp = {"max_depth": 10, "max_features": 11, **p}
        model = DecisionTreeRegressor(random_state=rs, **mp)
        model2 = DecisionTreeRegressor(random_state=rs, **mp)
    elif M == "rg":
        model = Ridge(random_state=rs, **p)
        model2 = Ridge(random_state=rs, **p)
    elif M == "el":
        model = ElasticNet(random_state=rs, **p)
        model2 = ElasticNet(random_state=rs, **p)
    elif M == "la":
        model = Lars(**p)
        model2 = Lars(**p)
    elif M == "ll":
        model = LassoLars(**p)
        model2 = LassoLars(**p)
    elif M == "or":
        model = OrthogonalMatchingPursuit(**p)
        model2 = OrthogonalMatchingPursuit(**p)
    elif M == "brg":
        model = BayesianRidge(**p)
        model2 = BayesianRidge(**p)
    elif M == "ardr":
        model = ARDRegression(**p)
        model2 = ARDRegression(**p)
    elif M == "ransa":
        model = RANSACRegressor(random_state=rs, **p)
        model2 = RANSACRegressor(random_state=rs, **p)
    elif M == "the":
        model = TheilSenRegressor(random_state=rs, **p)
        model2 = TheilSenRegressor(random_state=rs, **p)
    elif M == "hub":
        model = HuberRegressor(**p)
        model2 = HuberRegressor(**p)
    elif M == "sgdr":
        model = SGDRegressor(random_state=rs, **p)
        model2 = SGDRegressor(random_state=rs, **p)
    elif M == "kn":
        model = KNeighborsRegressor(**p)
        model2 = KNeighborsRegressor(**p)
    elif M == "gu":
        model = GaussianProcessRegressor(random_state=rs, **p)
        model2 = GaussianProcessRegressor(random_state=rs, **p)
    elif M == "ex":
        model = ExtraTreesRegressor(random_state=rs, **p)
        model2 = ExtraTreesRegressor(random_state=rs, **p)
    elif M == "bg":
        model = BaggingRegressor(random_state=rs, **p)
        model2 = BaggingRegressor(random_state=rs, **p)
    elif M == "gb":
        model = GradientBoostingRegressor(random_state=rs, **p)
        model2 = GradientBoostingRegressor(random_state=rs, **p)
    elif M == "hgb":
        model = HistGradientBoostingRegressor(random_state=rs, **p)
        model2 = HistGradientBoostingRegressor(random_state=rs, **p)
    elif M == "xgb":
        xgboost = _import_xgboost()
        model = xgboost.XGBRegressor(random_state=rs, **p)
        model2 = xgboost.XGBRegressor(random_state=rs, **p)
    elif M == "ada":
        model = AdaBoostRegressor(random_state=rs, **p)
        model2 = AdaBoostRegressor(random_state=rs, **p)
    elif M == "dl":
        dl_params = {
            "epochs": 300,
            "hidden_layer_sizes": (500, 1000),
            "learning_rate": 0.01,
            "nesterov": True,
            "optimizer": "adam",
            "dropout": 0.2,
            "batch_size": 16,
            "l2": 0.0,
            **p,
        }
        model = _build_dl_model(
            len(v_names),
            dl_params["hidden_layer_sizes"],
            dl_params["dropout"],
            dl_params["optimizer"],
            dl_params["learning_rate"],
            dl_params["nesterov"],
            task="regression",
            l2=dl_params["l2"],
            rs=rs,
        )
    else:
        raise ValueError(f"unknown model name M={M!r}")

    if M == "dl":
        from keras.callbacks import EarlyStopping

        # Unlike the generic `cv != "off"` block below, this internal 5-fold CV
        # always runs, regardless of `cv` -- same (already-shipped) behavior as
        # ModelC()'s M="dl" path: whether the resulting CV metric is actually
        # returned is gated by `cv` later, but the extra fits are always paid for.
        loo = KFold(n_splits=5, shuffle=True, random_state=rs)
        for train_idx, test_idx in loo.split(x):
            X_train, X_test = X_array[train_idx], X_array[test_idx]
            y_train, y_test = y_array[train_idx], y_array[test_idx]

            model2 = _build_dl_model(
                len(v_names),
                dl_params["hidden_layer_sizes"],
                dl_params["dropout"],
                dl_params["optimizer"],
                dl_params["learning_rate"],
                dl_params["nesterov"],
                task="regression",
                l2=dl_params["l2"],
                rs=rs,
            )
            early_stopping = EarlyStopping(
                monitor="val_loss", patience=5, restore_best_weights=True
            )
            model2.fit(
                X_train,
                y_train,
                batch_size=dl_params["batch_size"],
                epochs=dl_params["epochs"],
                validation_data=(X_test, y_test),
                callbacks=[early_stopping],
                verbose=0,
            )
            y_pred = model2.predict(X_test, verbose=0).ravel()
            ytests += list(y_test)
            ypreds += list(y_pred)

        early_stopping = EarlyStopping(
            monitor="loss", patience=5, restore_best_weights=True
        )
        model.fit(
            x,
            y,
            batch_size=dl_params["batch_size"],
            epochs=dl_params["epochs"],
            callbacks=[early_stopping],
            verbose=0,
        )
        y_predict_train = model.predict(x, verbose=0).ravel()
        y_predict_test = model.predict(xtest, verbose=0).ravel()
        r2 = r2_score(y, y_predict_train)
    else:
        model.fit(x, y)
        y_predict_train = model.predict(x)
        y_predict_test = model.predict(xtest)
        r2 = model.score(x, y)
    R2test = r2test(ytest, y_predict_test, y)
    Pearson = stats.pearsonr(ytest, y_predict_test)
    q2f2 = r2test(ytest, y_predict_test, ytest)
    model_mse_test = mean_squared_error(y_predict_test, ytest)
    try:
        f = F(y, y_predict_train, k=len(v_names))
    except ValueError:
        # F() encodes classical OLS degrees-of-freedom (n > k+1, k literally-fit
        # linear parameters) -- a real constraint for M in {mlr, pls, lasso, ...},
        # but not one that means anything for RF/SVM/tree-style regressors, whose
        # capacity isn't controlled by feature count the same way. Reporting NaN
        # here (rather than blocking the fit) matches ModelC(), which has no such
        # constraint at all, and lets those model types run on any feature count.
        f = float("nan")

    List = None
    if M != "dl" and cv != "off":
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
    elif M == "dl" and cv != "off":
        # ytests/ypreds were already populated by M="dl"'s own internal 5-fold
        # CV above (unconditional, unlike the generic loop) -- just turn them
        # into the same List shape the generic path returns.
        q2 = q2r2(ytests, ypreds)
        rmse = RMSEP_CV_C(ytests, ypreds)
        List = {"q2": q2, "RMSECV": rmse, "Q2F2": q2f2}

    if M in ("svm", "dl"):
        sorted_VI = ""
    else:
        try:
            if M == "nn":
                # "Connection weights" importance (Olden & Jackson, 2002,
                # Ecological Modelling 154:135-150): the product of all
                # weight matrices, input -> ... -> output, sums the product
                # of weights along every path through the network. This is
                # only the network's *true* input-output sensitivity if
                # every activation is the identity function; sklearn's
                # MLPRegressor/MLPClassifier default to "relu" here (not
                # overridden above), so this ignores every ReLU threshold
                # and is a rough linear surrogate, not an exact value --
                # one that gets rougher the more hidden layers there are.
                # Treat it as directional (which features rank higher),
                # not as a precise coefficient.
                VI = np.linalg.multi_dot(model.coefs_)
            elif M in ("rf", "tree", "ex", "gb", "xgb", "ada"):
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
    y3.to_csv(path + "train.csv")
    y4 = pd.DataFrame(ytest, index=xtest.index, columns=["observed"])
    y5 = pd.DataFrame(y_predict_test, index=xtest.index, columns=["predicted"])
    y6 = pd.concat([y4, y5], axis=1)
    y6.to_csv(path + "test.csv")

    analysis = None
    if cv != "off":
        analysis = analyse(
            y, y_predict_train, ytest, y_predict_test, ytests, ypreds, k=len(v_names)
        )
    return [result, List, model, analysis]


def _import_xgboost():
    """Import xgboost, raising an informative error if the optional extra is missing.

    Same pattern as :func:`_build_dl_model` below (lazy import, actionable
    ImportError) -- M="xgb" isn't a hard dependency, so a bare ImportError
    from deep inside xgboost's own import machinery would be confusing.
    """
    try:
        import xgboost
    except ImportError as e:
        raise ImportError(
            "M='xgb' requires the optional 'xgboost' extra. Install with: "
            "pip install 'mlmolprop[xgboost]'"
        ) from e
    return xgboost


# Torch models belong behind this same M= interface (e.g. M="dl_torch"),
# not a separate function/entry point -- ModelC() is already the single
# place callers pick a classifier by name, and a second interface just for
# the backend would make callers care about an implementation detail that
# isn't otherwise exposed. Follow the same pattern as _build_dl_model()
# below: import torch lazily inside the builder, and raise an ImportError
# pointing at `pip install mlmolprop[torch]` if it's missing.
def _build_dl_model(
    n_features,
    hidden_layer_sizes,
    dropout,
    optimizer,
    learning_rate,
    nesterov,
    task="classification",
    l2=0.0,
    rs=None,
):
    try:
        from keras import Sequential, optimizers, regularizers
        from keras import utils as keras_utils
        from keras.layers import Dense, Dropout, Input
    except ImportError as e:
        raise ImportError(
            "M='dl' requires the optional 'dl' extra (keras plus the "
            "PyTorch backend engine). Install with: pip install "
            "'mlmolprop[dl]', and set the environment variable "
            "KERAS_BACKEND=torch before running."
        ) from e

    # Without this, weight initialization, dropout masks, and (for optimizer="sgd")
    # minibatch shuffling are all drawn from whatever global RNG state keras/torch
    # happen to be in -- entirely independent of `rs` -- so two calls with identical
    # arguments (including the same `rs`) produce different trained weights and
    # different CV/test scores. Confirmed by hand: re-running an identical (task,
    # params, rs) config gave a Q2_cv ~0.025 different from the original run.
    if rs is not None:
        keras_utils.set_random_seed(rs)

    # l2=0.0 (the default) passes kernel_regularizer=None -- keras.regularizers.l2(0.0)
    # would technically also work out to no penalty, but constructing a zero-strength
    # regularizer object is needless overhead on every layer for the common case where
    # no one asked for L2 at all.
    kernel_regularizer = regularizers.l2(l2) if l2 else None

    model = Sequential()
    model.add(Input(shape=(n_features,)))
    for units in hidden_layer_sizes:
        model.add(Dense(
            units, kernel_initializer="uniform", activation="relu",
            kernel_regularizer=kernel_regularizer,
        ))
        model.add(Dropout(dropout))
    if task == "regression":
        model.add(Dense(1, kernel_initializer="normal", activation="linear"))
        loss, metrics = "mse", ["mae"]
    else:
        model.add(Dense(1, kernel_initializer="normal", activation="sigmoid"))
        loss, metrics = "binary_crossentropy", ["accuracy"]

    if optimizer == "sgd":
        opt = optimizers.SGD(
            learning_rate=learning_rate, momentum=0.9, nesterov=nesterov
        )
    else:
        opt = optimizers.Adam(learning_rate=learning_rate)
    model.compile(loss=loss, optimizer=opt, metrics=metrics)
    return model


def _interval_loss(y_true, y_pred):
    """Squared-hinge loss against a ``[conf_low, conf_high]`` interval -- zero when the
    prediction falls inside the interval, a quadratic penalty outside it.

    Covers both real credible-interval labels and right-censored labels with the same
    formula: a censored label is just an interval with no meaningful lower bound (the
    caller sets its ``conf_low`` to a sentinel far below any real value in the data, so
    the ``under`` term never contributes and only ``conf_high`` constrains training).

    ``y_true`` carries the interval as two columns (``conf_low``, ``conf_high``);
    ``y_pred`` is the usual single-column prediction. See :func:`ModelMT`'s
    ``Y_bounds`` parameter for how this gets wired into a real fit.

    Returns the **per-sample** loss (shape ``(batch, 1)``), not a pre-reduced scalar --
    Keras's own loss wrapper applies ``sample_weight`` and reduces to a scalar itself,
    *after* calling this function, so returning an already-averaged scalar here would
    make every row contribute equally to that average regardless of its
    ``sample_weight`` (a real bug caught while adding per-row weighting: rows meant to
    be fully excluded, sample_weight 0, were still pulling predictions toward their
    dummy-filled bounds -- confirmed by a training run where predictions on real,
    correctly-labeled rows landed systematically ~1.8 pIC50 units low, from unmasked
    rows dragging the shared network toward their dummy [0, 0] target).
    """
    from keras import ops

    conf_low = y_true[:, 0:1]
    conf_high = y_true[:, 1:2]
    over = ops.relu(y_pred - conf_high)
    under = ops.relu(conf_low - y_pred)
    return ops.square(over) + ops.square(under)


def _build_dl_mt_model(
    n_features,
    task_names,
    hidden_layer_sizes,
    dropout,
    optimizer,
    learning_rate,
    nesterov,
    task="regression",
    l2=0.0,
    rs=None,
    interval_tasks=None,
    task_weights=None,
):
    """Multitask counterpart of :func:`_build_dl_model`: one shared Dense/Dropout
    trunk (same layer pattern as the single-task builder) feeding one named
    ``Dense(1)`` output head per entry in ``task_names``, via the Functional
    API rather than ``Sequential`` (needed for multiple named outputs).

    Per-task loss/metrics are chosen the same way ``_build_dl_model`` already
    splits by ``task`` -- linear/``mse`` for regression, sigmoid/
    ``binary_crossentropy`` for classification -- just applied uniformly to
    every head. See :func:`ModelMT`/:func:`ModelCMT` for how missing per-task
    labels are masked out at fit time via per-output ``sample_weight``.

    ``interval_tasks`` : set[str] or None
        Regression tasks (``task="regression"`` only, ignored otherwise) that
        compile with :func:`_interval_loss` instead of plain ``"mse"`` -- see
        :func:`ModelMT`'s ``Y_bounds`` parameter.
    ``task_weights`` : dict[str, float] or None
        Optional per-task loss weight, passed straight to
        ``model.compile(loss_weights=...)``. ``None`` weights every task equally
        (Keras's own default).
    """
    try:
        from keras import Model as KerasModel
        from keras import optimizers, regularizers
        from keras import utils as keras_utils
        from keras.layers import Dense, Dropout, Input
    except ImportError as e:
        raise ImportError(
            "ModelMT()/ModelCMT() require the optional 'dl' extra (keras plus "
            "the PyTorch backend engine). Install with: pip install "
            "'mlmolprop[dl]', and set the environment variable "
            "KERAS_BACKEND=torch before running."
        ) from e

    if rs is not None:
        keras_utils.set_random_seed(rs)

    kernel_regularizer = regularizers.l2(l2) if l2 else None
    interval_tasks = interval_tasks or set()

    inputs = Input(shape=(n_features,))
    h = inputs
    for units in hidden_layer_sizes:
        h = Dense(
            units, kernel_initializer="uniform", activation="relu",
            kernel_regularizer=kernel_regularizer,
        )(h)
        h = Dropout(dropout)(h)

    if task == "regression":
        outputs = {
            name: Dense(1, kernel_initializer="normal", activation="linear", name=name)(h)
            for name in task_names
        }
        loss = {
            name: (_interval_loss if name in interval_tasks else "mse")
            for name in task_names
        }
        metrics = {name: ["mae"] for name in task_names}
    else:
        outputs = {
            name: Dense(1, kernel_initializer="normal", activation="sigmoid", name=name)(h)
            for name in task_names
        }
        loss = {name: "binary_crossentropy" for name in task_names}
        metrics = {name: ["accuracy"] for name in task_names}

    model = KerasModel(inputs=inputs, outputs=outputs)

    if optimizer == "sgd":
        opt = optimizers.SGD(learning_rate=learning_rate, momentum=0.9, nesterov=nesterov)
    else:
        opt = optimizers.Adam(learning_rate=learning_rate)
    model.compile(loss=loss, optimizer=opt, metrics=metrics, loss_weights=task_weights)
    return model


def ModelMT(
    x, Y, xtest, Ytest, v_names, params=None, rs=None, cv="kf", path="./",
    Y_bounds=None, task_weights=None,
):
    """Multitask counterpart of :func:`Model`: fit one shared-trunk Keras MLP
    (see :func:`_build_dl_mt_model`), with one linear output head per task,
    jointly across every column of ``Y`` -- rather than one independent
    ``Model(M="dl")`` call per task.

    Parameters
    ----------
    x, xtest : shared feature matrix (train/test). Every task in ``Y`` is
        trained against the *same* input representation -- a shared trunk
        needs one common input space, so (unlike the per-task feature
        selection ``Model()`` is usually paired with) this expects a
        task-agnostic feature set as input.
    Y, Ytest : DataFrame, one column per task, ``NaN`` where that task's
        label is missing for a given row. Masking missing labels is not
        optional here -- most rows are missing most tasks.
    v_names : list[str]
        Feature names (for the per-task ``analyse()``/``F()`` calls).
    params : dict or None
        Same recognized keys as ``Model()``'s ``M="dl"`` (``epochs``,
        ``hidden_layer_sizes``, ``learning_rate``, ``nesterov``,
        ``optimizer``, ``dropout``, ``l2``, ``batch_size``) -- one shared
        config for every task's output head, since this is a shared-bottom
        model, not per-task tuning.
    cv : {"kf", "off"}
        Whether to report the internal 5-fold CV metrics. Like ``Model()``'s
        ``M="dl"`` path, the internal CV fit always happens regardless of
        ``cv``; ``cv="off"`` only skips reporting it.
    Y_bounds : dict[str, tuple] or None
        Optional ``{task: (conf_low, conf_high)}`` or ``{task: (conf_low,
        conf_high, row_weight)}`` for tasks that should train against
        :func:`_interval_loss` instead of plain MSE -- covers real
        credible-interval labels and right-censored labels with one mechanism
        (a censored row's ``conf_low`` is a sentinel far below any real value,
        set by the caller, so only its upper bound constrains training). Every
        row a task's ``Y`` column has a real point value for must also have
        bounds here -- once a task is in ``Y_bounds``, *all* of its training
        signal (real and censored) routes through the interval loss, not a
        mix of two mechanisms. Rows with bounds but no ``Y`` point value
        (censored rows) train the model but are excluded from
        ``R2``/``Pearson``/etc. reporting below, which has no true value to
        score them against. Tasks not present in ``Y_bounds`` are completely
        unaffected -- today's plain-MSE behavior, unchanged.

        The optional third element, ``row_weight``, multiplies each row's
        presence-based weight -- use it to downweight rows whose bounds are
        looser/less trustworthy than a real measurement (right-censored
        labels, harvested labels mapped from another assay). This matters in
        practice: a "free below the bound" censored label costs nothing to
        satisfy once a prediction clears it, so when censored rows are a
        large fraction of a task's training mass, satisfying them cheaply can
        systematically bias predictions low even on real rows, via the
        network they share -- downweighting keeps that from dominating the
        real measurements' gradient contribution. Omit it (a 2-tuple) for
        rows that should all count equally once present, same as before.
    task_weights : dict[str, float] or None
        Optional per-task loss weight (``model.compile(loss_weights=...)``).
        ``None`` means uniform weighting -- today's behavior.

    Returns
    -------
    tuple
        ``(results_by_task, List_by_task, model, analysis_by_task)`` --
        three dicts keyed by task name (each task's entry shaped like
        ``Model()``'s own ``result``/``List``/``analysis``), plus the single
        fitted multitask model shared by every task.
    """
    try:
        from keras.callbacks import EarlyStopping
    except ImportError as e:
        raise ImportError(
            "ModelMT()/ModelCMT() require the optional 'dl' extra (keras plus "
            "the PyTorch backend engine). Install with: pip install "
            "'mlmolprop[dl]', and set the environment variable "
            "KERAS_BACKEND=torch before running."
        ) from e

    task_names = list(Y.columns)
    p = dict(params or {})
    dl_params = {
        "epochs": 300,
        "hidden_layer_sizes": (500, 1000),
        "learning_rate": 0.01,
        "nesterov": True,
        "optimizer": "adam",
        "dropout": 0.2,
        "batch_size": 16,
        "l2": 0.0,
        **p,
    }

    Y_bounds = Y_bounds or {}
    interval_tasks = set(Y_bounds)

    X_array = np.array(x)
    mask = Y.notna()
    Y_filled = Y.fillna(0.0)
    mask_test = Ytest.notna()

    # Per-task training target/mask: interval-loss tasks train on (conf_low, conf_high)
    # pairs and are "present" wherever bounds exist (real labels AND censored rows, via
    # bounds_mask); plain-MSE tasks keep the original point-value/mask pair unchanged.
    # `mask`/`Y_filled` above stay exactly as before and still drive evaluation (real
    # point values only) further down -- fit_mask/fit_targets are fit()-only.
    #
    # Each Y_bounds[t] entry is (conf_low, conf_high) or (conf_low, conf_high,
    # row_weight) -- the optional third element down-weights specific rows (e.g. a
    # right-censored label relative to a real measurement) beyond plain presence/absence.
    # This matters in practice, not just in principle: a "free below the bound" censored
    # label costs nothing to satisfy once a prediction clears it, so when censored rows
    # are a large fraction of a task's training mass, an early, cheap gradient signal to
    # predict low across the board can dominate before the network learns to fit the
    # real, narrower intervals -- systematically biasing predictions low even on real
    # rows, via the trunk they share (confirmed empirically while building this: a
    # trained model reached a healthy, converged loss yet still landed R2 around -5 on
    # real rows, from a ~1.8-unit systematic downward offset despite a real 0.49
    # prediction-truth correlation -- correlated but badly biased, exactly what an
    # unweighted "free below" majority produces). Downweighting those rows' contribution
    # relative to real measurements is the fix, not a training bug.
    fit_targets = {}
    fit_mask = dict(mask)
    for t in task_names:
        if t in interval_tasks:
            bounds_entry = Y_bounds[t]
            conf_low, conf_high = bounds_entry[0], bounds_entry[1]
            bounds_mask = conf_low.notna() & conf_high.notna()
            if len(bounds_entry) == 3:
                row_weight = bounds_entry[2]
                fit_mask[t] = bounds_mask.astype(float) * row_weight.fillna(0.0)
            else:
                fit_mask[t] = bounds_mask
            fit_targets[t] = np.column_stack([
                conf_low.fillna(0.0).to_numpy(), conf_high.fillna(0.0).to_numpy(),
            ])
        else:
            fit_targets[t] = Y_filled[t].to_numpy()

    def _build():
        return _build_dl_mt_model(
            len(v_names), task_names,
            dl_params["hidden_layer_sizes"], dl_params["dropout"],
            dl_params["optimizer"], dl_params["learning_rate"], dl_params["nesterov"],
            task="regression", l2=dl_params["l2"], rs=rs,
            interval_tasks=interval_tasks, task_weights=task_weights,
        )

    ytests_by_task = {t: [] for t in task_names}
    ypreds_by_task = {t: [] for t in task_names}

    loo = KFold(n_splits=5, shuffle=True, random_state=rs)
    for train_idx, test_idx in loo.split(X_array):
        X_train, X_test = X_array[train_idx], X_array[test_idx]
        model2 = _build()
        model2.fit(
            X_train,
            {t: fit_targets[t][train_idx] for t in task_names},
            sample_weight={t: fit_mask[t].to_numpy(dtype=float)[train_idx] for t in task_names},
            batch_size=dl_params["batch_size"],
            epochs=dl_params["epochs"],
            validation_data=(
                X_test,
                {t: fit_targets[t][test_idx] for t in task_names},
                {t: fit_mask[t].to_numpy(dtype=float)[test_idx] for t in task_names},
            ),
            callbacks=[EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)],
            verbose=0,
        )
        preds = model2.predict(X_test, verbose=0)
        for t in task_names:
            pred_t = np.ravel(preds[t])
            # Evaluation/CV metrics always use real point values only (`mask`, not
            # `fit_mask`) -- censored rows have no true value to score predictions
            # against, even though they did contribute to training this fold's model.
            keep = mask[t].to_numpy()[test_idx]
            ytests_by_task[t] += list(Y_filled[t].to_numpy()[test_idx][keep])
            ypreds_by_task[t] += list(pred_t[keep])

    model = _build()
    model.fit(
        X_array,
        {t: fit_targets[t] for t in task_names},
        sample_weight={t: fit_mask[t].to_numpy(dtype=float) for t in task_names},
        batch_size=dl_params["batch_size"],
        epochs=dl_params["epochs"],
        callbacks=[EarlyStopping(monitor="loss", patience=5, restore_best_weights=True)],
        verbose=0,
    )

    y_predict_train_by_task = model.predict(X_array, verbose=0)
    y_predict_test_by_task = model.predict(np.array(xtest), verbose=0)

    results_by_task = {}
    List_by_task = {} if cv != "off" else None
    analysis_by_task = {} if cv != "off" else None

    for t in task_names:
        train_keep = mask[t].to_numpy()
        test_keep = mask_test[t].to_numpy()

        y_train_t = Y[t].to_numpy()[train_keep]
        y_pred_train_t = np.ravel(y_predict_train_by_task[t])[train_keep]
        y_test_t = Ytest[t].to_numpy()[test_keep]
        y_pred_test_t = np.ravel(y_predict_test_by_task[t])[test_keep]

        R2 = q2r2(y_train_t, y_pred_train_t)
        R2test = r2test(y_test_t, y_pred_test_t, y_train_t)
        Pearson = stats.pearsonr(y_test_t, y_pred_test_t)
        model_mse_test = mean_squared_error(y_pred_test_t, y_test_t)
        try:
            f = F(y_train_t, y_pred_train_t, k=len(v_names))
        except ValueError:
            f = float("nan")

        results_by_task[t] = {
            "R2": R2,
            "Mean_squared_error_test": model_mse_test,
            "R2_test": R2test,
            "F": f,
            "Variable Importance": "",
            "Pearson": Pearson,
        }

        # A task's own accumulated CV rows can, in principle, come up empty
        # (every fold's held-out slice happened to miss that task's sparse
        # labels) -- guard rather than let RMSEP_CV_C's n=0 raise.
        if cv != "off" and ytests_by_task[t]:
            q2 = q2r2(ytests_by_task[t], ypreds_by_task[t])
            rmse = RMSEP_CV_C(ytests_by_task[t], ypreds_by_task[t])
            q2f2 = r2test(y_test_t, y_pred_test_t, y_test_t)
            List_by_task[t] = {"q2": q2, "RMSECV": rmse, "Q2F2": q2f2}
            analysis_by_task[t] = analyse(
                y_train_t, y_pred_train_t, y_test_t, y_pred_test_t,
                ytests_by_task[t], ypreds_by_task[t], k=len(v_names),
            )
        elif cv != "off":
            List_by_task[t] = None
            analysis_by_task[t] = None

        y1 = pd.DataFrame(y_train_t, index=x.index[train_keep], columns=["observed"])
        y2 = pd.DataFrame(y_pred_train_t, index=x.index[train_keep], columns=["predicted"])
        pd.concat([y1, y2], axis=1).to_csv(f"{path}{t}_train.csv")
        y4 = pd.DataFrame(y_test_t, index=xtest.index[test_keep], columns=["observed"])
        y5 = pd.DataFrame(y_pred_test_t, index=xtest.index[test_keep], columns=["predicted"])
        pd.concat([y4, y5], axis=1).to_csv(f"{path}{t}_test.csv")

    return results_by_task, List_by_task, model, analysis_by_task


def _build_dl_mmoe_model(
    n_features,
    task_names,
    hidden_layer_sizes,
    n_experts,
    dropout,
    optimizer,
    learning_rate,
    nesterov,
    task="regression",
    l2=0.0,
    rs=None,
):
    """MMoE (Ma et al. 2018) counterpart of :func:`_build_dl_mt_model`: instead of
    one hard-shared trunk every task is forced through, ``n_experts`` independent
    expert subnetworks (each its own ``Input -> [Dense(relu)+Dropout]*`` stack per
    ``hidden_layer_sizes`` -- now each expert's own architecture, not one shared
    trunk's) feed a per-task learned gate (``Dense(n_experts,
    activation="softmax")`` on the raw input) that weights how much that task's
    output draws from each expert -- letting a task down-weight experts shaped
    mostly by other tasks' gradients, instead of sharing one representation
    outright.

    Combine step (the standard Keras MMoE pattern): stack the experts' outputs
    into one ``(batch, n_experts, expert_dim)`` tensor via
    ``keras.ops.stack`` (backend-agnostic -- verified working with this project's
    torch backend), then contract each task's gate against it with
    ``keras.layers.Dot(axes=(1, 1))`` to get that task's gated ``(batch,
    expert_dim)`` representation. Same linear/``mse`` vs sigmoid/
    ``binary_crossentropy`` split ``_build_dl_mt_model`` already makes for the
    final per-task ``Dense(1)`` head.
    """
    try:
        from keras import Model as KerasModel
        from keras import ops, optimizers, regularizers
        from keras import utils as keras_utils
        from keras.layers import Dense, Dot, Dropout, Input, Lambda
    except ImportError as e:
        raise ImportError(
            "ModelMMoE()/ModelCMMoE() require the optional 'dl' extra (keras plus "
            "the PyTorch backend engine). Install with: pip install "
            "'mlmolprop[dl]', and set the environment variable "
            "KERAS_BACKEND=torch before running."
        ) from e

    if rs is not None:
        keras_utils.set_random_seed(rs)

    kernel_regularizer = regularizers.l2(l2) if l2 else None

    inputs = Input(shape=(n_features,))

    experts = []
    for _ in range(n_experts):
        h = inputs
        for units in hidden_layer_sizes:
            h = Dense(
                units, kernel_initializer="uniform", activation="relu",
                kernel_regularizer=kernel_regularizer,
            )(h)
            h = Dropout(dropout)(h)
        experts.append(h)
    expert_stack = Lambda(lambda x: ops.stack(x, axis=1))(experts)

    if task == "regression":
        activation, loss_name, metrics_list = "linear", "mse", ["mae"]
    else:
        activation, loss_name, metrics_list = "sigmoid", "binary_crossentropy", ["accuracy"]

    outputs = {}
    for name in task_names:
        gate = Dense(n_experts, activation="softmax", name=f"{name}_gate")(inputs)
        gated = Dot(axes=(1, 1))([gate, expert_stack])
        outputs[name] = Dense(1, kernel_initializer="normal", activation=activation, name=name)(gated)
    loss = {name: loss_name for name in task_names}
    metrics = {name: metrics_list for name in task_names}

    model = KerasModel(inputs=inputs, outputs=outputs)

    if optimizer == "sgd":
        opt = optimizers.SGD(learning_rate=learning_rate, momentum=0.9, nesterov=nesterov)
    else:
        opt = optimizers.Adam(learning_rate=learning_rate)
    model.compile(loss=loss, optimizer=opt, metrics=metrics)
    return model


def ModelMMoE(x, Y, xtest, Ytest, v_names, params=None, rs=None, cv="kf", path="./"):
    """MMoE counterpart of :func:`ModelMT`: fit one shared-experts Keras MLP (see
    :func:`_build_dl_mmoe_model`), with ``n_experts`` shared expert subnetworks and
    a per-task learned gate, instead of ``ModelMT()``'s single hard-shared trunk
    every task is forced through equally.

    Same shared-``x``/masked-``Y`` contract, internal-CV/masking/metrics logic, and
    return shape as :func:`ModelMT` -- see its docstring for everything that
    doesn't differ. ``params`` recognizes one extra key: ``n_experts`` (default 4).

    Returns
    -------
    tuple
        ``(results_by_task, List_by_task, model, analysis_by_task)`` -- same
        shape as :func:`ModelMT`'s return.
    """
    try:
        from keras.callbacks import EarlyStopping
    except ImportError as e:
        raise ImportError(
            "ModelMMoE()/ModelCMMoE() require the optional 'dl' extra (keras plus "
            "the PyTorch backend engine). Install with: pip install "
            "'mlmolprop[dl]', and set the environment variable "
            "KERAS_BACKEND=torch before running."
        ) from e

    task_names = list(Y.columns)
    p = dict(params or {})
    dl_params = {
        "epochs": 300,
        "hidden_layer_sizes": (500, 1000),
        "n_experts": 4,
        "learning_rate": 0.01,
        "nesterov": True,
        "optimizer": "adam",
        "dropout": 0.2,
        "batch_size": 16,
        "l2": 0.0,
        **p,
    }

    X_array = np.array(x)
    mask = Y.notna()
    Y_filled = Y.fillna(0.0)
    mask_test = Ytest.notna()

    def _build():
        return _build_dl_mmoe_model(
            len(v_names), task_names,
            dl_params["hidden_layer_sizes"], dl_params["n_experts"], dl_params["dropout"],
            dl_params["optimizer"], dl_params["learning_rate"], dl_params["nesterov"],
            task="regression", l2=dl_params["l2"], rs=rs,
        )

    ytests_by_task = {t: [] for t in task_names}
    ypreds_by_task = {t: [] for t in task_names}

    loo = KFold(n_splits=5, shuffle=True, random_state=rs)
    for train_idx, test_idx in loo.split(X_array):
        X_train, X_test = X_array[train_idx], X_array[test_idx]
        model2 = _build()
        model2.fit(
            X_train,
            {t: Y_filled[t].to_numpy()[train_idx] for t in task_names},
            sample_weight={t: mask[t].to_numpy(dtype=float)[train_idx] for t in task_names},
            batch_size=dl_params["batch_size"],
            epochs=dl_params["epochs"],
            validation_data=(
                X_test,
                {t: Y_filled[t].to_numpy()[test_idx] for t in task_names},
                {t: mask[t].to_numpy(dtype=float)[test_idx] for t in task_names},
            ),
            callbacks=[EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)],
            verbose=0,
        )
        preds = model2.predict(X_test, verbose=0)
        for t in task_names:
            pred_t = np.ravel(preds[t])
            keep = mask[t].to_numpy()[test_idx]
            ytests_by_task[t] += list(Y_filled[t].to_numpy()[test_idx][keep])
            ypreds_by_task[t] += list(pred_t[keep])

    model = _build()
    model.fit(
        X_array,
        {t: Y_filled[t].to_numpy() for t in task_names},
        sample_weight={t: mask[t].to_numpy(dtype=float) for t in task_names},
        batch_size=dl_params["batch_size"],
        epochs=dl_params["epochs"],
        callbacks=[EarlyStopping(monitor="loss", patience=5, restore_best_weights=True)],
        verbose=0,
    )

    y_predict_train_by_task = model.predict(X_array, verbose=0)
    y_predict_test_by_task = model.predict(np.array(xtest), verbose=0)

    results_by_task = {}
    List_by_task = {} if cv != "off" else None
    analysis_by_task = {} if cv != "off" else None

    for t in task_names:
        train_keep = mask[t].to_numpy()
        test_keep = mask_test[t].to_numpy()

        y_train_t = Y[t].to_numpy()[train_keep]
        y_pred_train_t = np.ravel(y_predict_train_by_task[t])[train_keep]
        y_test_t = Ytest[t].to_numpy()[test_keep]
        y_pred_test_t = np.ravel(y_predict_test_by_task[t])[test_keep]

        R2 = q2r2(y_train_t, y_pred_train_t)
        R2test = r2test(y_test_t, y_pred_test_t, y_train_t)
        Pearson = stats.pearsonr(y_test_t, y_pred_test_t)
        model_mse_test = mean_squared_error(y_pred_test_t, y_test_t)
        try:
            f = F(y_train_t, y_pred_train_t, k=len(v_names))
        except ValueError:
            f = float("nan")

        results_by_task[t] = {
            "R2": R2,
            "Mean_squared_error_test": model_mse_test,
            "R2_test": R2test,
            "F": f,
            "Variable Importance": "",
            "Pearson": Pearson,
        }

        # Same empty-fold guard as ModelMT() -- a sparsely-labeled task can, in
        # principle, end up with zero accumulated CV rows.
        if cv != "off" and ytests_by_task[t]:
            q2 = q2r2(ytests_by_task[t], ypreds_by_task[t])
            rmse = RMSEP_CV_C(ytests_by_task[t], ypreds_by_task[t])
            q2f2 = r2test(y_test_t, y_pred_test_t, y_test_t)
            List_by_task[t] = {"q2": q2, "RMSECV": rmse, "Q2F2": q2f2}
            analysis_by_task[t] = analyse(
                y_train_t, y_pred_train_t, y_test_t, y_pred_test_t,
                ytests_by_task[t], ypreds_by_task[t], k=len(v_names),
            )
        elif cv != "off":
            List_by_task[t] = None
            analysis_by_task[t] = None

        y1 = pd.DataFrame(y_train_t, index=x.index[train_keep], columns=["observed"])
        y2 = pd.DataFrame(y_pred_train_t, index=x.index[train_keep], columns=["predicted"])
        pd.concat([y1, y2], axis=1).to_csv(f"{path}{t}_train.csv")
        y4 = pd.DataFrame(y_test_t, index=xtest.index[test_keep], columns=["observed"])
        y5 = pd.DataFrame(y_pred_test_t, index=xtest.index[test_keep], columns=["predicted"])
        pd.concat([y4, y5], axis=1).to_csv(f"{path}{t}_test.csv")

    return results_by_task, List_by_task, model, analysis_by_task


def ModelC(
    x,
    y,
    xtest,
    ytest,
    v_names,
    params=None,
    M="tree",
    rs=None,
    cv="loo",
    path="./",
):
    """Fit a binary classifier and evaluate it on train/test/cross-validation.

    Parameters
    ----------
    x, y : training feature matrix and target.
    xtest, ytest : held-out feature matrix and target.
    v_names : list[str]
        Feature names (used for variable-importance reporting).
    path : str, default "./"
        Directory prefix for output files (``"train.csv"``, ``"test.csv"``) --
        same convention as :func:`clus_uns`'s ``path`` parameter.
    M : str, default "tree"
        Which classifier to fit. One of: tree, nn, rf, ex, lsvm, svm,
        lr, ld, rg, per, pass, qua, sgdc, kn, rn, gu, gunb, cnb, bg, gb,
        hgb, xgb, ada, dl (a small Keras MLP). (Lasso/KernelRidge are
        regression-only models and aren't offered here -- see Model() for
        those.) M="xgb" requires the optional "xgboost" extra (see
        :func:`_import_xgboost`); M="dl" requires the optional "dl" extra.
    cv : {"loo", "kf", "kfr", "shuff"}
        Cross-validation strategy for the CV-based accuracy report.
    params : dict or None
        Model-specific hyperparameters, passed straight through as
        keyword arguments to the underlying scikit-learn estimator's
        constructor (layered on top of per-model defaults chosen to
        match this function's previous fixed behavior) -- e.g.
        ``{"max_depth": 8, "n_estimators": 200}`` for M="rf", or
        ``{"C": 5, "kernel": "linear"}`` for M="svm". ``class_weight``
        is a recognized key for every model that supports it (rf,
        lsvm, svm, lr, hgb natively; gb and xgb emulate it via a
        per-fit ``sample_weight`` computed from whatever ``y`` is
        actually passed to ``.fit()``, since their underlying
        estimators have no ``class_weight`` constructor argument).
        For M="dl" (the Keras MLP, which has no
        scikit-learn constructor), the recognized keys are instead
        ``epochs``, ``hidden_layer_sizes``, ``learning_rate``,
        ``nesterov``, ``optimizer`` ("adam"/"sgd"), ``dropout``, ``l2``
        (L2 weight-regularization strength, default 0.0 -- no penalty),
        and ``batch_size``.

    Returns
    -------
    list
        ``[result, model]``.
    """
    ytests = []
    ypreds = []
    X_array = np.array(x)
    y_array = np.array(y)
    p = dict(params or {})
    svm_kernel = p.get("kernel", "rbf")
    # Set only for models whose underlying estimator has no native
    # class_weight constructor argument (gb, xgb) -- "balanced" here means
    # a per-fit sample_weight (computed from whatever y that fit actually
    # sees) is used to emulate it, same intent as the other models' native
    # class_weight="balanced" default.
    sample_weight_mode = None

    if M == "tree":
        mp = {"max_depth": 10, "max_features": 10, **p}
        model = DecisionTreeClassifier(random_state=rs, **mp)
        model2 = DecisionTreeClassifier(random_state=rs, **mp)
    elif M == "nn":
        mp = {
            "solver": "adam",
            "alpha": 0.0001,
            "hidden_layer_sizes": (10, 10, 10),
            "max_iter": 200,
            **p,
        }
        model = MLPClassifier(random_state=rs, **mp)
        model2 = MLPClassifier(random_state=rs, **mp)
    elif M == "rf":
        mp = {
            "max_depth": 10,
            "n_estimators": 20,
            "max_features": "sqrt",
            "max_leaf_nodes": None,
            "class_weight": "balanced",
            **p,
        }
        model = RandomForestClassifier(random_state=rs, **mp)
        model2 = RandomForestClassifier(random_state=rs, **mp)
    elif M == "ex":
        mp = {"max_depth": 10, **p}
        model = ExtraTreesClassifier(random_state=rs, **mp)
        model2 = ExtraTreesClassifier(random_state=rs, **mp)
    elif M == "lsvm":
        mp = {"C": 10, "class_weight": "balanced", **p}
        model = LinearSVC(random_state=rs, **mp)
        model2 = LinearSVC(random_state=rs, **mp)
    elif M == "svm":
        # probability=True is required for the predict_proba() calls
        # this function makes later on -- not exposed as a tunable key.
        mp = {
            "C": 10,
            "kernel": "rbf",
            "max_iter": 200,
            "gamma": 10,
            "class_weight": "balanced",
            **p,
        }
        model = SVC(random_state=rs, probability=True, **mp)
        model2 = SVC(random_state=rs, probability=True, **mp)
    elif M == "lr":
        mp = {"max_iter": 10, "class_weight": "balanced", **p}
        model = LogisticRegression(random_state=rs, **mp)
        model2 = LogisticRegression(random_state=rs, **mp)
    elif M == "ld":
        model = LinearDiscriminantAnalysis(**p)
        model2 = LinearDiscriminantAnalysis(**p)
    elif M == "rg":
        model = RidgeClassifier(random_state=rs, **p)
        model2 = RidgeClassifier(random_state=rs, **p)
    elif M == "per":
        model = Perceptron(random_state=rs, **p)
        model2 = Perceptron(random_state=rs, **p)
    elif M == "pass":
        model = PassiveAggressiveClassifier(random_state=rs, **p)
        model2 = PassiveAggressiveClassifier(random_state=rs, **p)
    elif M == "qua":
        model = QuadraticDiscriminantAnalysis(**p)
        model2 = QuadraticDiscriminantAnalysis(**p)
    elif M == "sgdc":
        model = SGDClassifier(random_state=rs, **p)
        model2 = SGDClassifier(random_state=rs, **p)
    elif M == "kn":
        mp = {"n_neighbors": 10, **p}
        model = KNeighborsClassifier(**mp)
        model2 = KNeighborsClassifier(**mp)
    elif M == "rn":
        model = RadiusNeighborsClassifier(**p)
        model2 = RadiusNeighborsClassifier(**p)
    elif M == "gu":
        model = GaussianProcessClassifier(random_state=rs, **p)
        model2 = GaussianProcessClassifier(random_state=rs, **p)
    elif M == "gunb":
        model = GaussianNB(**p)
        model2 = GaussianNB(**p)
    elif M == "cnb":
        mp = {"alpha": 10, **p}
        model = ComplementNB(**mp)
        model2 = ComplementNB(**mp)
    elif M == "bg":
        model = BaggingClassifier(random_state=rs, **p)
        model2 = BaggingClassifier(random_state=rs, **p)
    elif M == "gb":
        mp = {"class_weight": "balanced", **p}
        sample_weight_mode = mp.pop("class_weight")
        model = GradientBoostingClassifier(random_state=rs, **mp)
        model2 = GradientBoostingClassifier(random_state=rs, **mp)
    elif M == "hgb":
        mp = {"class_weight": "balanced", **p}
        model = HistGradientBoostingClassifier(random_state=rs, **mp)
        model2 = HistGradientBoostingClassifier(random_state=rs, **mp)
    elif M == "xgb":
        xgboost = _import_xgboost()
        mp = {"class_weight": "balanced", **p}
        sample_weight_mode = mp.pop("class_weight")
        model = xgboost.XGBClassifier(random_state=rs, **mp)
        model2 = xgboost.XGBClassifier(random_state=rs, **mp)
    elif M == "ada":
        model = AdaBoostClassifier(random_state=rs, **p)
        model2 = AdaBoostClassifier(random_state=rs, **p)
    elif M == "dl":
        from sklearn.utils.class_weight import compute_class_weight

        dl_params = {
            "epochs": 300,
            "hidden_layer_sizes": (500, 1000),
            "learning_rate": 0.01,
            "nesterov": True,
            "optimizer": "adam",
            "dropout": 0.2,
            "batch_size": 16,
            "l2": 0.0,
            **p,
        }
        np.random.seed(1)
        class_weight_values = compute_class_weight(
            "balanced", classes=np.unique(y_array), y=y_array
        )
        class_weights = dict(enumerate(class_weight_values))
        model = _build_dl_model(
            len(v_names),
            dl_params["hidden_layer_sizes"],
            dl_params["dropout"],
            dl_params["optimizer"],
            dl_params["learning_rate"],
            dl_params["nesterov"],
            l2=dl_params["l2"],
            rs=rs,
        )
    else:
        raise ValueError(f"unknown model name M={M!r}")

    if M == "dl":
        from keras.callbacks import EarlyStopping
        from sklearn.model_selection import StratifiedKFold

        loo = StratifiedKFold(n_splits=5)
        for train_idx, test_idx in loo.split(x, y):
            X_train, X_test = X_array[train_idx], X_array[test_idx]
            y_train, y_test = y_array[train_idx], y_array[test_idx]

            model2 = _build_dl_model(
                len(v_names),
                dl_params["hidden_layer_sizes"],
                dl_params["dropout"],
                dl_params["optimizer"],
                dl_params["learning_rate"],
                dl_params["nesterov"],
                l2=dl_params["l2"],
                rs=rs,
            )
            early_stopping = EarlyStopping(
                monitor="val_loss", patience=5, restore_best_weights=True
            )
            model2.fit(
                X_train,
                y_train,
                batch_size=dl_params["batch_size"],
                epochs=dl_params["epochs"],
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
            batch_size=dl_params["batch_size"],
            epochs=dl_params["epochs"],
            callbacks=[early_stopping],
            verbose=0,
            class_weight=class_weights,
        )

        y_predict_train = (model.predict(x, verbose=0) > 0.5).astype("int32").ravel()
        y_predict_test = (model.predict(xtest, verbose=0) > 0.5).astype("int32").ravel()
    else:
        if sample_weight_mode == "balanced":
            model.fit(x, y, sample_weight=compute_sample_weight("balanced", y_array))
        else:
            model.fit(x, y)
        y_predict_train = model.predict(x)
        y_predict_test = model.predict(xtest)

    accuracy_score_train = accuracy_score(y, y_predict_train, normalize=True)
    accuracy_score_test = accuracy_score(ytest, y_predict_test, normalize=True)

    if M != "dl" and cv != "off":
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
            if sample_weight_mode == "balanced":
                model2.fit(X_train, y_train, sample_weight=compute_sample_weight("balanced", y_train))
            else:
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
        # M="dl" never ran CV here to begin with; cv="off" skips it the same
        # way Model() does (that function already supported "off" -- this
        # one didn't: none of the loo/kf/kfr/shuff branches above matched
        # "off", so `loo` was never assigned and loo.split(x) raised
        # UnboundLocalError as soon as cv="off" was passed).
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
    elif M in ("rf", "ex", "gb", "xgb", "ada"):
        VI = model.feature_importances_
    elif M == "svm":
        if svm_kernel == "linear":
            VI = model.coef_[0]
        else:
            VI = ""
    elif M == "nn":
        # "Connection weights" importance -- see the fuller note in Model()
        # above. Rough linear surrogate (ignores ReLU), directional only.
        VI = np.linalg.multi_dot(model.coefs_)
    elif (
        M in ("qua", "kn", "gu", "bg", "rn", "hgb")
        or M == "gunb"
        or M == "cnb"
        or M == "dl"
    ):
        # HistGradientBoosting* (unlike RandomForest/GradientBoosting/
        # ExtraTrees) exposes neither feature_importances_ nor coef_ --
        # sklearn's own recommendation is permutation_importance instead,
        # which needs a scorer/held-out data, not just the fitted model.
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
        M in ("qua", "kn", "gu", "bg", "rn", "hgb", "dl")
        or M == "gunb"
        or M == "cnb"
        or M == "svm"
        and svm_kernel != "linear"
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
        "hgb",
        "xgb",
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
    # cv="off": ytests/ypreds stay empty (no CV loop ran above), so there's
    # no CV confusion matrix to build -- matches Model()'s cv="off" leaving
    # its own CV-derived `analysis` as None.
    cnf3 = confusion_matrix(ytests, ypreds) if cv != "off" else None
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
    if cnf3 is not None:
        tnc, fpc, fnc, tpc = cnf3.ravel()
        cnf3[0][0] = tpc
        cnf3[1][1] = tnc
        cnf3[0][1] = fnc
        cnf3[1][0] = fpc
        cvmetrics = metr(tpc, tnc, fpc, fnc)
    else:
        cvmetrics = None

    trainmetrics = metr(tp, tn, fp, fn)
    testmetrics = metr(tpt, tnt, fpt, fnt)
    total = metr(tpt + tp, tnt + tn, fpt + fp, fnt + fn)

    plt.figure(figsize=(8, 7))
    plot_confusion_matrix(cnf, classes=["Active", "Inactive"], title="Train set")
    plt.ylabel("Actual", fontsize=24, fontweight="bold")
    plt.xlabel("Predicted", fontsize=24, fontweight="bold")
    plt.tight_layout()
    if cnf3 is not None:
        plt.figure(figsize=(8, 7))
        plot_confusion_matrix(
            cnf3, classes=["Active", "Inactive"], title="Cross-Validation"
        )
        plt.ylabel("Actual", fontsize=24, fontweight="bold")
        plt.xlabel("Predicted", fontsize=24, fontweight="bold")
        plt.tight_layout()
    plt.figure(figsize=(8, 7))
    plot_confusion_matrix(cnf2, classes=["Active", "Inactive"], title="Test set")
    plt.ylabel("Actual", fontsize=24, fontweight="bold")
    plt.xlabel("Predicted", fontsize=24, fontweight="bold")
    plt.tight_layout()

    y1 = pd.DataFrame(y, index=x.index, columns=["observed"])
    y2 = pd.DataFrame(y_predict_train, index=x.index, columns=["predicted"])
    y3 = pd.concat([y1, y2], axis=1)
    y3.to_csv(path + "train.csv")
    y4 = pd.DataFrame(ytest, index=xtest.index, columns=["observed"])
    y5 = pd.DataFrame(y_predict_test, index=xtest.index, columns=["predicted"])
    y6 = pd.concat([y4, y5], axis=1)
    y6.to_csv(path + "test.csv")

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


def ModelCMT(
    x, Y, xtest, Ytest, v_names, params=None, rs=None, cv="kf", path="./",
    plot=False, task_weights=None,
):
    """Multitask counterpart of :func:`ModelC`: fit one shared-trunk Keras MLP
    (see :func:`_build_dl_mt_model`), with one sigmoid output head per task,
    jointly across every column of ``Y``.

    Same shared-``x``/masked-``Y`` contract as :func:`ModelMT` -- see its
    docstring. Each task's ``sample_weight`` folds in both label-presence
    masking and "balanced" class weighting computed from that task's own
    non-missing labels, since Keras's per-``fit()`` ``class_weight`` argument
    (used by ``ModelC()``'s single-task ``M="dl"`` path) has no per-output
    equivalent for a multi-output model -- same "class_weight-as-
    sample_weight" pattern this module already uses for ``M="gb"``/``M="xgb"``
    in ``ModelC()``.

    Internal CV uses a plain ``KFold`` split rather than ``ModelC()``'s
    ``StratifiedKFold`` -- joint stratification across ``len(task_names)``
    independently-missing binary tasks isn't well-defined for one shared
    split.

    ``plot`` : bool, default False
        If True, draw train/CV/test confusion-matrix figures per task (3 x
        ``len(task_names)`` figures) via :func:`plot_confusion_matrix`, same
        layout and TP/TN/FN/FP display convention as ``ModelC()``'s own
        plots -- just titled per task and gated behind this flag rather than
        always-on, since a multitask call has multiple tasks to plot instead
        of ``ModelC()``'s one. Off by default so existing callers/tests are
        unaffected.
    ``task_weights`` : dict[str, float] or None
        Optional per-task loss weight (``model.compile(loss_weights=...)``),
        same as :func:`ModelMT`'s parameter of the same name -- e.g. downweight
        an auxiliary classification head relative to a track's scored main
        head(s). ``None`` means uniform weighting -- today's behavior.

    Returns
    -------
    tuple
        ``(results_by_task, model)`` -- a dict keyed by task name (each
        task's entry restricted to the same ``train_*``/``test_*``/
        ``CV_metrics`` keys ``ModelC()``'s own ``result`` already has), plus
        the single fitted multitask model.
    """
    try:
        from keras.callbacks import EarlyStopping
    except ImportError as e:
        raise ImportError(
            "ModelMT()/ModelCMT() require the optional 'dl' extra (keras plus "
            "the PyTorch backend engine). Install with: pip install "
            "'mlmolprop[dl]', and set the environment variable "
            "KERAS_BACKEND=torch before running."
        ) from e
    from sklearn.utils.class_weight import compute_class_weight

    task_names = list(Y.columns)
    p = dict(params or {})
    dl_params = {
        "epochs": 300,
        "hidden_layer_sizes": (500, 1000),
        "learning_rate": 0.01,
        "nesterov": True,
        "optimizer": "adam",
        "dropout": 0.2,
        "batch_size": 16,
        "l2": 0.0,
        **p,
    }

    X_array = np.array(x)
    mask = Y.notna()
    Y_filled = Y.fillna(0).astype(int)
    mask_test = Ytest.notna()
    Ytest_filled = Ytest.fillna(0).astype(int)

    # Computed once from the full training Y, then sliced per CV fold below --
    # rows where a task's label is missing get weight 0 regardless of their
    # (dummy) filled label, so per-fold class balance doesn't need
    # recomputing from each fold's own (already-masked) subset.
    sample_weight_full = {}
    for t in task_names:
        labels_present = Y.loc[mask[t], t].to_numpy()
        cw = dict(enumerate(compute_class_weight("balanced", classes=np.array([0, 1]), y=labels_present)))
        labels = Y_filled[t].to_numpy()
        weights = np.where(labels == 1, cw[1], cw[0]).astype(float)
        sample_weight_full[t] = weights * mask[t].to_numpy(dtype=float)

    def _build():
        return _build_dl_mt_model(
            len(v_names), task_names,
            dl_params["hidden_layer_sizes"], dl_params["dropout"],
            dl_params["optimizer"], dl_params["learning_rate"], dl_params["nesterov"],
            task="classification", l2=dl_params["l2"], rs=rs,
            task_weights=task_weights,
        )

    ytests_by_task = {t: [] for t in task_names}
    ypreds_by_task = {t: [] for t in task_names}

    loo = KFold(n_splits=5, shuffle=True, random_state=rs)
    for train_idx, test_idx in loo.split(X_array):
        X_train, X_test = X_array[train_idx], X_array[test_idx]
        model2 = _build()
        model2.fit(
            X_train,
            {t: Y_filled[t].to_numpy()[train_idx] for t in task_names},
            sample_weight={t: sample_weight_full[t][train_idx] for t in task_names},
            batch_size=dl_params["batch_size"],
            epochs=dl_params["epochs"],
            validation_data=(
                X_test,
                {t: Y_filled[t].to_numpy()[test_idx] for t in task_names},
                {t: sample_weight_full[t][test_idx] for t in task_names},
            ),
            callbacks=[EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)],
            verbose=0,
        )
        preds = model2.predict(X_test, verbose=0)
        for t in task_names:
            pred_t = (np.ravel(preds[t]) > 0.5).astype(int)
            keep = mask[t].to_numpy()[test_idx]
            ytests_by_task[t] += list(Y_filled[t].to_numpy()[test_idx][keep])
            ypreds_by_task[t] += list(pred_t[keep])

    model = _build()
    model.fit(
        X_array,
        {t: Y_filled[t].to_numpy() for t in task_names},
        sample_weight=sample_weight_full,
        batch_size=dl_params["batch_size"],
        epochs=dl_params["epochs"],
        callbacks=[EarlyStopping(monitor="loss", patience=5, restore_best_weights=True)],
        verbose=0,
    )

    y_predict_train_by_task = model.predict(X_array, verbose=0)
    y_predict_test_by_task = model.predict(np.array(xtest), verbose=0)

    results_by_task = {}
    for t in task_names:
        train_keep = mask[t].to_numpy()
        test_keep = mask_test[t].to_numpy()

        y_train_t = Y_filled[t].to_numpy()[train_keep]
        pred_train_t = (np.ravel(y_predict_train_by_task[t]) > 0.5).astype(int)[train_keep]
        y_test_t = Ytest_filled[t].to_numpy()[test_keep]
        pred_test_t = (np.ravel(y_predict_test_by_task[t]) > 0.5).astype(int)[test_keep]

        cnf = confusion_matrix(y_train_t, pred_train_t, labels=[0, 1])
        cnf2 = confusion_matrix(y_test_t, pred_test_t, labels=[0, 1])
        tn, fp, fn, tp = cnf.ravel()
        tnt, fpt, fnt, tpt = cnf2.ravel()
        trainmetrics = metr(tp, tn, fp, fn)
        testmetrics = metr(tpt, tnt, fpt, fnt)

        # Same empty-fold guard as ModelMT() -- a sparsely-labeled task can,
        # in principle, end up with zero accumulated CV rows.
        cv_panel = None
        if cv != "off" and ytests_by_task[t]:
            cnf3 = confusion_matrix(ytests_by_task[t], ypreds_by_task[t], labels=[0, 1])
            tnc, fpc, fnc, tpc = cnf3.ravel()
            cvmetrics = metr(tpc, tnc, fpc, fnc)
            cv_panel = (cnf3.copy(), tpc, tnc, fnc, fpc, f"{t} -- Cross-Validation")
        else:
            cvmetrics = None

        if plot:
            # Same TP/TN/FN/FP display reorder ModelC() applies before plotting (its own
            # tn/fp/fn/tp unpack above is metr()'s input order, not the plotted layout) --
            # [0][0]/[1][1] are TP/TN, [0][1]/[1][0] are FN/FP, so "Active" (label 1) shows
            # top-left instead of sklearn's default TN-top-left layout.
            panels = [
                (cnf.copy(), tp, tn, fn, fp, f"{t} -- Train set"),
                (cnf2.copy(), tpt, tnt, fnt, fpt, f"{t} -- Test set"),
            ]
            if cv_panel is not None:
                panels.append(cv_panel)
            for panel_cnf, panel_tp, panel_tn, panel_fn, panel_fp, title in panels:
                panel_cnf[0][0], panel_cnf[1][1] = panel_tp, panel_tn
                panel_cnf[0][1], panel_cnf[1][0] = panel_fn, panel_fp
                plt.figure(figsize=(8, 7))
                plot_confusion_matrix(panel_cnf, classes=["Active", "Inactive"], title=title)
                plt.ylabel("Actual", fontsize=24, fontweight="bold")
                plt.xlabel("Predicted", fontsize=24, fontweight="bold")
                plt.tight_layout()

        results_by_task[t] = {
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
            "CV_metrics": cvmetrics,
        }

    return results_by_task, model


def ModelCMMoE(x, Y, xtest, Ytest, v_names, params=None, rs=None, cv="kf", path="./", plot=False):
    """MMoE counterpart of :func:`ModelCMT`: fit one shared-experts Keras MLP (see
    :func:`_build_dl_mmoe_model`), with ``n_experts`` shared expert subnetworks and
    a per-task learned gate, instead of ``ModelCMT()``'s single hard-shared trunk.

    Same shared-``x``/masked-``Y`` contract, "balanced" class-weight-as-
    ``sample_weight`` handling, plain-``KFold`` internal CV, and ``plot=True``
    per-task confusion-matrix plotting as :func:`ModelCMT` -- see its docstring
    for everything that doesn't differ. ``params`` recognizes one extra key:
    ``n_experts`` (default 4).

    Returns
    -------
    tuple
        ``(results_by_task, model)`` -- same shape as :func:`ModelCMT`'s return.
    """
    try:
        from keras.callbacks import EarlyStopping
    except ImportError as e:
        raise ImportError(
            "ModelMMoE()/ModelCMMoE() require the optional 'dl' extra (keras plus "
            "the PyTorch backend engine). Install with: pip install "
            "'mlmolprop[dl]', and set the environment variable "
            "KERAS_BACKEND=torch before running."
        ) from e
    from sklearn.utils.class_weight import compute_class_weight

    task_names = list(Y.columns)
    p = dict(params or {})
    dl_params = {
        "epochs": 300,
        "hidden_layer_sizes": (500, 1000),
        "n_experts": 4,
        "learning_rate": 0.01,
        "nesterov": True,
        "optimizer": "adam",
        "dropout": 0.2,
        "batch_size": 16,
        "l2": 0.0,
        **p,
    }

    X_array = np.array(x)
    mask = Y.notna()
    Y_filled = Y.fillna(0).astype(int)
    mask_test = Ytest.notna()
    Ytest_filled = Ytest.fillna(0).astype(int)

    # Computed once from the full training Y, then sliced per CV fold below -- rows
    # where a task's label is missing get weight 0 regardless of their (dummy)
    # filled label, so per-fold class balance doesn't need recomputing from each
    # fold's own (already-masked) subset.
    sample_weight_full = {}
    for t in task_names:
        labels_present = Y.loc[mask[t], t].to_numpy()
        cw = dict(enumerate(compute_class_weight("balanced", classes=np.array([0, 1]), y=labels_present)))
        labels = Y_filled[t].to_numpy()
        weights = np.where(labels == 1, cw[1], cw[0]).astype(float)
        sample_weight_full[t] = weights * mask[t].to_numpy(dtype=float)

    def _build():
        return _build_dl_mmoe_model(
            len(v_names), task_names,
            dl_params["hidden_layer_sizes"], dl_params["n_experts"], dl_params["dropout"],
            dl_params["optimizer"], dl_params["learning_rate"], dl_params["nesterov"],
            task="classification", l2=dl_params["l2"], rs=rs,
        )

    ytests_by_task = {t: [] for t in task_names}
    ypreds_by_task = {t: [] for t in task_names}

    loo = KFold(n_splits=5, shuffle=True, random_state=rs)
    for train_idx, test_idx in loo.split(X_array):
        X_train, X_test = X_array[train_idx], X_array[test_idx]
        model2 = _build()
        model2.fit(
            X_train,
            {t: Y_filled[t].to_numpy()[train_idx] for t in task_names},
            sample_weight={t: sample_weight_full[t][train_idx] for t in task_names},
            batch_size=dl_params["batch_size"],
            epochs=dl_params["epochs"],
            validation_data=(
                X_test,
                {t: Y_filled[t].to_numpy()[test_idx] for t in task_names},
                {t: sample_weight_full[t][test_idx] for t in task_names},
            ),
            callbacks=[EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)],
            verbose=0,
        )
        preds = model2.predict(X_test, verbose=0)
        for t in task_names:
            pred_t = (np.ravel(preds[t]) > 0.5).astype(int)
            keep = mask[t].to_numpy()[test_idx]
            ytests_by_task[t] += list(Y_filled[t].to_numpy()[test_idx][keep])
            ypreds_by_task[t] += list(pred_t[keep])

    model = _build()
    model.fit(
        X_array,
        {t: Y_filled[t].to_numpy() for t in task_names},
        sample_weight=sample_weight_full,
        batch_size=dl_params["batch_size"],
        epochs=dl_params["epochs"],
        callbacks=[EarlyStopping(monitor="loss", patience=5, restore_best_weights=True)],
        verbose=0,
    )

    y_predict_train_by_task = model.predict(X_array, verbose=0)
    y_predict_test_by_task = model.predict(np.array(xtest), verbose=0)

    results_by_task = {}
    for t in task_names:
        train_keep = mask[t].to_numpy()
        test_keep = mask_test[t].to_numpy()

        y_train_t = Y_filled[t].to_numpy()[train_keep]
        pred_train_t = (np.ravel(y_predict_train_by_task[t]) > 0.5).astype(int)[train_keep]
        y_test_t = Ytest_filled[t].to_numpy()[test_keep]
        pred_test_t = (np.ravel(y_predict_test_by_task[t]) > 0.5).astype(int)[test_keep]

        cnf = confusion_matrix(y_train_t, pred_train_t, labels=[0, 1])
        cnf2 = confusion_matrix(y_test_t, pred_test_t, labels=[0, 1])
        tn, fp, fn, tp = cnf.ravel()
        tnt, fpt, fnt, tpt = cnf2.ravel()
        trainmetrics = metr(tp, tn, fp, fn)
        testmetrics = metr(tpt, tnt, fpt, fnt)

        # Same empty-fold guard as ModelMMoE() -- a sparsely-labeled task can, in
        # principle, end up with zero accumulated CV rows.
        cv_panel = None
        if cv != "off" and ytests_by_task[t]:
            cnf3 = confusion_matrix(ytests_by_task[t], ypreds_by_task[t], labels=[0, 1])
            tnc, fpc, fnc, tpc = cnf3.ravel()
            cvmetrics = metr(tpc, tnc, fpc, fnc)
            cv_panel = (cnf3.copy(), tpc, tnc, fnc, fpc, f"{t} -- Cross-Validation")
        else:
            cvmetrics = None

        if plot:
            # Same TP/TN/FN/FP display reorder ModelC()/ModelCMT() apply before
            # plotting -- [0][0]/[1][1] are TP/TN, [0][1]/[1][0] are FN/FP, so
            # "Active" (label 1) shows top-left instead of sklearn's default
            # TN-top-left layout.
            panels = [
                (cnf.copy(), tp, tn, fn, fp, f"{t} -- Train set"),
                (cnf2.copy(), tpt, tnt, fnt, fpt, f"{t} -- Test set"),
            ]
            if cv_panel is not None:
                panels.append(cv_panel)
            for panel_cnf, panel_tp, panel_tn, panel_fn, panel_fp, title in panels:
                panel_cnf[0][0], panel_cnf[1][1] = panel_tp, panel_tn
                panel_cnf[0][1], panel_cnf[1][0] = panel_fn, panel_fp
                plt.figure(figsize=(8, 7))
                plot_confusion_matrix(panel_cnf, classes=["Active", "Inactive"], title=title)
                plt.ylabel("Actual", fontsize=24, fontweight="bold")
                plt.xlabel("Predicted", fontsize=24, fontweight="bold")
                plt.tight_layout()

        results_by_task[t] = {
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
            "CV_metrics": cvmetrics,
        }

    return results_by_task, model


def plot_confusion_matrix(
    cm, classes, normalize=False, title="Confusion matrix", cmap=plt.cm.Blues
):
    """Plot a confusion matrix (optionally row-normalized)."""
    if normalize:
        cm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]

    plt.imshow(cm, interpolation="nearest", cmap=cmap)
    plt.title(title, fontsize=40, fontweight="bold")
    cbar = plt.colorbar()
    cbar.ax.tick_params(labelsize=20)
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45, fontsize=24)
    plt.yticks(tick_marks, classes, fontsize=24)

    fmt = ".2f" if normalize else "d"
    thresh = cm.max() / 2.0
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(
            j,
            i,
            format(cm[i, j], fmt),
            horizontalalignment="center",
            color="white" if cm[i, j] > thresh else "black",
            fontsize=44,
            fontweight="bold",
        )

    plt.tight_layout()
    plt.ylabel("True label", fontsize=24, fontweight="bold")
    plt.xlabel("Predicted label", fontsize=24, fontweight="bold")


def plot_w(x, y) -> None:
    plt.bar(x, y)
    plt.xticks(rotation=90)
    plt.show()


def rocc(
    x, y, ycat="", title="Receiver operating characteristic", tsize=12, pos_l=1, l="b"
):
    """Plot an ROC curve and print early-recognition enrichment factors (EF1/2/10/20/50)."""

    # Enrichment factor is defined over the top-scoring fraction, so sort by
    # score x (descending) before taking the first N rows -- COUNT() below
    # just takes rows in whatever order it's handed.
    order = np.argsort(-np.asarray(x))
    y_by_score = pd.Series(np.asarray(y)[order])

    def COUNT(col, label, percent=1):
        n = 0
        t = int(len(col) * percent)
        for i in range(t):
            if col.iloc[i] == label:
                n = n + 1
        return n

    total = float(len(y))
    baseline = COUNT(y_by_score, l, percent=1) / total
    ef_percents = {"EF1": 0.01, "EF2": 0.02, "EF10": 0.1, "EF20": 0.2, "EF50": 0.5}
    for name, percent in ef_percents.items():
        rate = COUNT(y_by_score, l, percent=percent) / (total * percent)
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
