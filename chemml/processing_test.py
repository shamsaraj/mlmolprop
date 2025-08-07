import csv
import numpy
import pandas
from modules.basic import *
import pickle
#import cPickle
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.feature_selection import VarianceThreshold
from modules.correlation import *
#http://benalexkeen.com/linear-regression-in-python-using-scikit-learn/
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression ,chi2, f_classif
#from sklearn.utils.validation import column_or_1d
#from descriptors import CI

from sklearn.feature_selection import RFE
from sklearn.ensemble import ExtraTreesClassifier
#Imputer
#OneHotEncoder

def data_prep(datafile, Scaled ="off", Normal = "off", FS = "off", Sparse = "off", cat = "", Cor = "off", ndes= 6, rs=None, newset="off", org_v_names="off", Xnormalized = "", Xscaled = "", v_names2="", imputation="off", vt=0,ref_X_train=0, output="./test.data", NAME="name", TARGET="IC50", REMOVE="", rowremoval="", ratio=0.2, selection="",my_list=""):
    ref_X_train=ref_X_train
    #Xnormalized = ""
    #Xscaled= ""
    #f = open (file, "rt")
    #reader = csv.reader(f)
    #for column in reader:
        #print column
    #print reader
    #dataset = numpy.loadtxt(reader, delimiter=",")
    #print dataset
    #print datafile, "ppppppppppppppppp"
    df=pandas.read_csv(datafile, low_memory=False)


    #print (df,2222)


    if imputation == "on": #noch hate problem

        #import sklearn.impute.SimpleImputer
        #I = sklearn.impute.SimpleImputer(missing_values="NA", strategy="mean", axis=1, verbose=0, copy=True).fit(df)
        #df2 = I.transform(df)
        #I = sklearn.impute.SimpleImputer(missing_values="NaN", strategy="mean", axis=1, verbose=0, copy=True).fit(df2)
        #df3 = I.transform(df2)
        #df = pandas.DataFrame(df3, columns=list(df.columns.values), index=df.loc[:, NAME])
        
        #new updated
        from sklearn.impute import SimpleImputer
        #example
	#imp = SimpleImputer(missing_values=np.nan, copy=False, strategy="mean", )


        #print df
        df = df.dropna(axis=1, how='all')#It will drop columns with all NaN or NA

        df1=df.drop(NAME, axis=1)# to overcome impuation error on the names
        
	#old
        #I = preprocessing.Imputer(missing_values="NaN", strategy="mean", axis=0, verbose=1, copy=True).fit(df1)
        #new version
        I = SimpleImputer (missing_values="NaN", strategy="mean", copy=True).fit(df1)
        # print dir(I)
        df2 = I.transform(df1)
        df = pandas.DataFrame(df2, columns=list(df.columns.values[1:]), index=df.loc[:, NAME])

    if rowremoval != "": #to remove rows with certain value
        #df = df.replace(rowremoval, pd.np.nan).dropna(axis=0, how='any').fillna(0).astype(int)
        #df = df.drop(rowremoval, axis=0)
        #df = df[df.TARGET != rowremoval]
        df.drop(df.loc[df[TARGET] == rowremoval].index, inplace=True)
        #df = df.drop(df[df[TARGET] == rowremoval].index)
    #print(df.head())
    if newset == "on":
        y =""
        #print df,"jjjjjjjjjjjjjjjj"
        df.columns.values[0] = NAME
        df=df.set_index(NAME)
        X=df
        #df1=df.drop(df.columns[0], axis=1)
        #df1 = df.drop(NAME, axis=1)
        if TARGET in df.columns:
            X = X.drop(TARGET, axis=1)
            y = X[[TARGET]]
            y = numpy.ravel(y)  # y = column_or_1d(y, warn=True)
        X = X[v_names2]
        if Xnormalized != "":
            X = Xnormalized.transform(X)
            X = pandas.DataFrame(X, columns=v_names2)#index=?#############
        if Xscaled != "":
            X = Xscaled.transform(X)
            X = pandas.DataFrame(X, columns=v_names2)
        #print df1, "111111111111111111111111"
        #print len (org_v_names), 444444444444
        #print len(v_names2)
        #X = df1.drop (org_v_names, axis=1)
        #print org_v_names , 9999999999999999999999
        #print (X,1111)
        X = X[org_v_names]
        
        #print (ref_X_train, 'ref_X_train')
        #print( X, 'X')
        #, org_v_names, ' org_v_names',ref_X_train.index, 'ref_X_train.index', X.index,'X.index')
        
        CI_table2 = CI(ref_X_train, X, org_v_names, ref_X_train.index, X.index, cutoff=1.5)#####################################    ref_X should be defined
        #pandas.DataFrame.to_csv(CI_table2[0], "CI_train2.csv")
        pandas.DataFrame.to_csv(CI_table2[1], "CI_exttest.csv")
        #pandas.DataFrame.to_csv(CI_table2[2], "CI_Out_train2.csv")
        pandas.DataFrame.to_csv(CI_table2[3], "CI_Out_exttest.csv")
        #print CI_table2
        #print X, "2222222222222222222222222"
        set = [X, y, org_v_names]
    else:
        #print "iiiiiiiiiiiiiiii",df, "iiiiiiiiiiiiiiiii"
        if REMOVE != "":
            #for i in REMOVE:
            #print i
            #print i[0]
            #df1 = df.drop(i, axis=1)
            df = df.drop(REMOVE, axis=1)
        if imputation != "on":
            df = df.drop(NAME, axis=1)
        if selection != "":
            selection = [TARGET] + list (selection)
            df = df[selection]
        #avr=average_bygroup(df, TARGET, "D:/pych/ml/average2.csv")###########
        X = df.drop(TARGET, axis=1)
        #X = X.loc[(X!= 0).any(1)]# to remove zero columns pandas
        #to remove zero columns
        X = X.replace(0, np.nan)
        X = X.dropna(how="all", axis=1)
        X = X.replace(np.nan, 0)
        #print X
        #store variable names
        ##v_names= list(X.columns.values)
        #print v_names
        # to remove zero variances pandas
        #selector = VarianceThreshold(threshold=0.0).fit(X)
        #X= selector.transform(X)
        #print selector.fit_transform(X)
        # store new variable names
        X = VarianceThreshold_selector(X, thereshould2= vt)
        #print (X)
        v_names = list(X.columns.values)
        v_names2 = v_names
        index_names= df.index#loc[:, NAME]
        X = pandas.DataFrame(data=X)
        X=X.set_index(index_names)
        #print X, 777777777777777
        y = df[[TARGET]]
        y=y.set_index(index_names)
        #print y ,6666666666666
        if cat == "on":
            enc = preprocessing.OneHotEncoder(X,categorical_features=cat)
            X = enc
        if Normal == "on":
            #Xnormalized = preprocessing.normalize(X)
            #X = Xnormalized
            Xnormalized = preprocessing.Normalizer().fit(X)
            X=Xnormalized.transform(X)
            X = pandas.DataFrame(X,columns=v_names,index=index_names)
        if Scaled == "on":
            #Xscaled = preprocessing.scale(X)
            #X = Xscaled
            Xscaled = preprocessing.StandardScaler().fit(X)
            X = Xscaled.transform(X)
            X = pandas.DataFrame(X, columns=v_names, index=index_names)
        if Sparse == "on":
            Scaled = preprocessing.MaxAbsScaler(X)
            X = Scaled
            X = pandas.DataFrame(X, columns=v_names, index=index_names)
        if Cor == "on":
            cor_mat= X.corr(method='pearson', min_periods=1)
            pandas.DataFrame.to_csv(cor_mat, "cor_mat.csv")
            #print cor_mat
            col_name = find_correlation(X, 0.9)
            #print col_name
            #print len(col_name)
            X = pandas.DataFrame (X)
            X = X.drop (col_name, axis=1)
            v_names = list(X.columns.values)
        y1 = y################
        if (FS == "clas") or (FS == "reg"):

            y = numpy.ravel(y)  # y = column_or_1d(y, warn=True)
            if FS == "clas":
                #s1 = SelectKBest(chi2, k=ndes)
                s1= SelectKBest(f_classif, k=ndes).fit(X,y)
                p_values=s1.pvalues_
                Bonferroni= p_values*len(v_names2)#############
                Bonferroni_d=pd.DataFrame(Bonferroni,index=v_names, columns=["p value"])
                pandas.DataFrame.to_csv(Bonferroni_d,"Bonferroni.csv")
                #result = pd.concat([Bonferroni_d, avr], axis=1, join_axes=[Bonferroni_d.index])###################
                #pandas.DataFrame.to_csv(result, "Bonferroni2.csv")##################
                #F_values= s1.scores_
                s2 = s1.fit_transform(X,y)

            elif FS == "reg":
                s1 = SelectKBest(score_func=f_regression, k=ndes).fit(X,y)
                s2= s1.transform(X)
            mask = s1.get_support()
            v_names = X.columns[mask]
            #s = SelectKBest_selector (X,y,ndes, FS)
            X = pandas.DataFrame(s2, columns=v_names, index=index_names)
            #s=SelectKBest(FS,ndes).fit(X,y)
            #X=s.transform(X,y)
            #print X,1111111111111111
            v_names = list(X.columns.values)
            #print X, 3333333333333
            #X = pandas.DataFrame(data=X, columns=v_names, index=index_names)
            #X = pandas.DataFrame(data=X, index=index_names)

        ###########################################################################################print v_names
        dfout = pd.concat([y1, X], axis=1)
        dfout.to_csv("dfout.csv")
         #y = numpy.ravel(y)  # y = column_or_1d(y, warn=True)
        if 1== 1:
            #import random
            #random.sample(range(1, 100), 26)
            
            y = numpy.ravel(y)  # y = column_or_1d(y, warn=True)
            y=pandas.DataFrame(y)
            #my_list=[123, 164, 137, 149, 168, 186, 176, 118, 159, 166, 160, 144, 178, 165, 141, 169, 147, 175]#list(range(0,129))CHEMBL2065
            
            #my_list=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116]#2065
            #my_list=[117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 161, 162, 163, 164, 165, 166, 167, 168, 169, 170, 171, 172, 173, 174, 175, 176, 177, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 191, 192, 193, 194, 195, 196, 197, 198, 199, 200, 201, 202, 203, 204, 205, 206, 207, 208, 209, 210, 211, 212, 213, 214, 215, 216, 217, 218, 219, 220, 221, 222, 223, 224]#2725
            #my_list=[225, 226, 227, 228, 229, 230, 231, 232, 233, 234, 235, 236, 237, 238, 239, 240, 241, 242, 243, 244, 245, 246, 247, 248, 249, 250, 251, 252, 253, 254, 255, 256, 257]#4114
            #my_list=[258, 259, 260, 261, 262, 263, 264, 265, 266, 267, 268, 269,269]#4314
            #my_list=[270,271, 272, 273, 274, 275, 276, 277, 278, 279, 280]#5956
            
            #reg
            #my_list=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130]#2065
            #my_list=[131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 161, 162, 163, 164, 165, 166, 167, 168, 169, 170, 171, 172, 173, 174, 175, 176, 177, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 191, 192, 193, 194, 195, 196, 197, 198, 199, 200, 201, 202, 203, 204, 205, 206, 207, 208, 209, 210, 211, 212, 213, 214, 215, 216, 217, 218, 219, 220, 221, 222, 223, 224, 225, 226, 227, 228, 229, 230, 231, 232, 233, 234, 235, 236, 237, 238, 239, 240, 241, 242, 243, 244, 245, 246, 247, 248, 249, 250, 251, 252]#2725
            #my_list=[253, 254, 255, 256, 257, 258, 259, 260, 261, 262, 263, 264, 265, 266, 267, 268, 269, 270, 271, 272, 273, 274, 275, 276, 277, 278, 279, 280, 281, 282, 283, 284, 285, 286, 287, 288, 289]#4114
            #my_list=[291, 292, 293, 294, 295, 296, 297, 298, 299, 300, 301, 302]#4314
            #my_list=[303, 303, 304, 305, 306, 307, 308, 309, 310, 311, 312, 313, 314,315]#5956 reg



            #my_list=[80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 161, 162, 163, 164, 165, 166, 167, 168, 169, 170, 171, 172, 173, 174, 175, 176, 177, 178, 179, 180, 181, 182, 183, 184, 185, 186]


            #my_list=[53,60, 68,76,111, 83, 60, 105, 63, 77, 61, 75, 84, 106, 96, 110, 70, 59,97]
            #my_list=[9, 58, 30, 7, 52, 10, 24, 8, 22, 31, 53, 43, 57, 17, 6]
            #my_list=[13, 12, 19, 22, 3, 37, 39, 15]#list(range(0,129))CHEMBL4114
            #my_list=[92, 79, 102, 101, 33, 58, 7, 51, 69, 49, 40, 61, 93, 23, 83, 116, 3, 110, 108, 45, 65, 100, 27, 56]
            #my_list=[12, 3, 1,5,6,7]#CHEMBL4314
            #my_list=[3, 1, 10]
            #my_list=[3, 1, 10]

            new_list = [x - 1 for x in my_list]
            X_test=X.iloc[new_list]
            X_train=X.drop(X.index[new_list])
            y_test= y.iloc[new_list]
            y_train= y.drop(y.index[new_list])
            y_train = numpy.ravel(y_train)  # y = column_or_1d(y, warn=True)
            y_test = numpy.ravel(y_test)
            #print (X_test,111111)
            #print(y_test,22222)
        elif 1==2:
            X_test = X.loc[["n1", "n2"]]
            X_train = X.drop(["n1", "n2"])
            y_test = y.loc[["n1", "n2"]]
            y_train = y.drop(["n1", "n2"])
            y_train = numpy.ravel(y_train)  # y = column_or_1d(y, warn=True)
            y_test = numpy.ravel(y_test)
        else:
            y = numpy.ravel(y)  # y = column_or_1d(y, warn=True)
            #print X,55555555555555555555555555555
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=ratio, random_state=rs, stratify=y)
            #print X_test, 6666666666
            #print X_test.index, 88888888888
        CI_table = CI (X_train, X_test, v_names, X_train.index,X_test.index,cutoff=3)
        #print CI_table [2]##################
        pandas.DataFrame.to_csv(CI_table[0], "CI_train.csv")
        pandas.DataFrame.to_csv(CI_table[1], "CI_test.csv")
        pandas.DataFrame.to_csv(CI_table[2], "CI_Out_train.csv")
        pandas.DataFrame.to_csv(CI_table[3], "CI_Out_test.csv")
        set = [X_train, y_train,X_test, y_test, v_names, Xnormalized, Xscaled, v_names2]

        object2file(set,output)
    return set





#http://benalexkeen.com/linear-regression-in-python-using-scikit-learn/

def multiC (datafile, Scaled ="off", Normal = "off", FS = "off", Sparse = "off", cat = "", Cor = "off", ndes= 6, rs=None, newset="off", org_v_names="off", Xnormalized = "", Xscaled = "", v_names2="", NAME="name"):
    df=pandas.read_csv(datafile)

    I = preprocessing.Imputer(missing_values="NaN", strategy="median", axis=1, verbose=0, copy=True).fit(df)
    df2 = I.transform(df)
    df2 = pandas.DataFrame(df2, columns=list(df.columns.values),index= df.loc[:, NAME])

    df1 = df2.drop(NAME, axis=1)


    X = df1.drop(df1.columns[0:34], axis=1)#31
    #to remove zero columns
    X = X.replace(0, np.nan)
    X = X.dropna(how="all", axis=1)
    X = X.replace(np.nan, 0)
    #store variable names
    #I= preprocessing.Imputer(missing_values="NaN", strategy="mean", axis=0, verbose=0, copy=True).fit(X)
    #X=I.transform(X)
    #X=pandas.DataFrame(X)
    #X = VarianceThreshold_selector (X)###############################################################################
    #print X
    v_names = list(X.columns.values)
    v_names2 = v_names
    index_names= df.loc[:, NAME]
    X = pandas.DataFrame(data=X)
    X=X.set_index(index_names)
    y = df1[df1.columns[0:34]]
    y=pandas.DataFrame(y)
    y=y.set_index(index_names)
    if cat == "on":
        enc = preprocessing.OneHotEncoder(X,categorical_features=cat)
        X = enc
    if Normal == "on":
        #Xnormalized = preprocessing.normalize(X)
        #X = Xnormalized
        Xnormalized = preprocessing.Normalizer().fit(X)
        X=Xnormalized.transform(X)
        X = pandas.DataFrame(X,columns=v_names,index=index_names)
    if Scaled == "on":
        #Xscaled = preprocessing.scale(X)
        #X = Xscaled
        Xscaled = preprocessing.StandardScaler().fit(X)
        X = Xscaled.transform(X)
        X = pandas.DataFrame(X, columns=v_names, index=index_names)
    if Sparse == "on":
        Scaled = preprocessing.MaxAbsScaler(X)
        X = Scaled
        X = pandas.DataFrame(X, columns=v_names, index=index_names)
    if Cor == "on":
        cor_mat= X.corr(method='pearson', min_periods=1)
        #print cor_mat##########
        col_name = find_correlation(X, 0.9)
        #print col_name
        #print len(col_name)
        X = pandas.DataFrame (X)
        X = X.drop (col_name, axis=1)
        v_names = list(X.columns.values)
    if (FS == "clas") or (FS == "reg"):
        y = numpy.ravel(y)  # y = column_or_1d(y, warn=True)
        if FS == "clas":
            #s1 = SelectKBest(chi2, k=ndes)
            #s1= SelectKBest(f_classif, k=ndes)
            #s2 = s1.fit(X, y)
            #s = SelectKBest(f_classif, k=ndes).fit_transform(X, y)
            estimator = ExtraTreesClassifier()
            selector = RFE(estimator, 5, step=1)
            s2 = selector.fit(X, y)
        elif FS == "reg":
            s1 = SelectKBest(score_func=f_regression, k=ndes)
            s2= s1.fit(X,y)
        mask = s2.get_support()
        v_names = X.columns[mask]
        #s = SelectKBest_selector (X,y,ndes, FS)
        X = pandas.DataFrame(X, columns=v_names, index=index_names)
        #s=SelectKBest(FS,ndes).fit(X,y)
        #X=s.transform(X,y)
        #print X,1111111111111111
        v_names = list(X.columns.values)
        #print X, 3333333333333
        #X = pandas.DataFrame(data=X, columns=v_names, index=index_names)
        #X = pandas.DataFrame(data=X, index=index_names)

    ###########################################################################################print v_names
    #y = numpy.ravel(y)  # y = column_or_1d(y, warn=True)
    if 1== 3:
        my_list=[1,2]
        new_list = [x - 1 for x in my_list]
        X_test=X.iloc[new_list]
        X_train=X.drop(X.index[new_list])
        y_test= y.iloc[new_list]
        y_train= y.drop(y.index[new_list])
        y_train = numpy.ravel(y_train)  # y = column_or_1d(y, warn=True)
        y_test = numpy.ravel(y_test)
    elif 1==2:
        X_test = X.loc[["n1", "n2"]]
        X_train = X.drop(["n1", "n2"])
        y_test = y.loc[["n1", "n2"]]
        y_train = y.drop(["n1", "n2"])
        y_train = numpy.ravel(y_train)  # y = column_or_1d(y, warn=True)
        y_test = numpy.ravel(y_test)
    else:
        #y = numpy.ravel(y)  # y = column_or_1d(y, warn=True)
        #print X,55555555555555555555555555555
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=rs)
        #print X_test, 6666666666
        #print X_test.index, 88888888888
    #CI_table = CI (X_train, X_test, v_names, X_train.index,X_test.index,1)
    #print CI_table [2]##################
    set = [X_train, y_train,X_test, y_test, v_names, Xnormalized, Xscaled, v_names2,X_train.index,X_test.index]
    return set
    pass


def VarianceThreshold_selector(data,thereshould2= 0):
    columns = data.columns
    selector = VarianceThreshold(thereshould2)
    selector.fit_transform(data)
    labels = [columns[x] for x in selector.get_support(indices=True) if x in selector.get_support(indices=True)]
    return pd.DataFrame(selector.fit_transform(data), columns=labels)

def VarianceThreshold_selector2(data):

    #Select Model
    selector = VarianceThreshold(0) #Defaults to 0.0, e.g. only remove features with the same value in all samples

    #Fit the Model
    selector.fit(data)
    features = selector.get_support(indices = True) #returns an array of integers corresponding to nonremoved features
    features = [column for column in data[features]] #Array of all nonremoved features' names

    #Format and Return
    selector = pd.DataFrame(selector.transform(data))
    selector.columns = features
    return selector

def SelectKBest_selector(data, data2, ndes, type):
    columns = data.columns
    if type == "clas":
        selector = SelectKBest(f_classif, k=ndes).fit (data, data2)
    elif type == "reg":
        selector = SelectKBest(f_regression, k=ndes).fit(data, data2)
    #selector.fit_transform(data)
    labels = [columns[x] for x in selector.get_support(indices=True) if x in selector.get_support(indices=True)]
    return pd.DataFrame(selector.fit_transform(data, data2), columns=labels)

def list2file(list1,file1):
    # define list of places
    places_list = list1

    with open(file1, 'w') as filehandle:
        filehandle.writelines("%s\n" % place for place in places_list)

def file2list(file1):
    # define empty list
    places = []
    # open file and read the content in a list
    with open(file1, 'r') as filehandle:
        places = [current_place.rstrip() for current_place in filehandle.readlines()]
    return places

def object2file (object1,file1):
    # load additional module
    import pickle
    # define a list of places
    placesList = object1
    with open(file1, 'wb') as filehandle:
        # store the data as binary data stream
        pickle.dump(placesList, filehandle)

def file2object (file1):
    # load additional module
    import pickle
    with open(file1, 'rb') as filehandle:
        # read the data as binary data stream
        placesList = pickle.load(filehandle)
    return placesList


def average_bygroup(df, groups, out):
    #df = pd.read_csv(csv)
    avr=df.groupby([groups]).mean() # .groupby(['cluster']).mean()
    avr=avr.transpose()
    #print avr
    avr.to_csv(out)
    return avr

def CI(X_train, X_test, v_names, Mtrain_names,Mtest_names, cutoff):
    import numpy
    import pandas
    temp = numpy.array(X_train)
    temp2 = numpy.array(X_test)
    #print X_train
    #print temp
    # print X_train
    r = len(temp)
    c = len(temp[0])
    table_train = twodlist(r, c)
    out_train = twodlist(r, c)
    r2 = len(temp2)
    c2 = len(temp2[0])
    table_test = twodlist(r2, c2)
    out_test= twodlist(r2, c2)
    # print X_train[0][1]
    for i in range(0, r):
        table_train[i] = temp[i]
        if i < r2:
            table_test[i] = temp2[i]
    # print table_train
    column = makecolumn(table_train, c)
    #column = makecolumn(table_test, c2)
    # print column [116]
    # print X_train
    for i in range(0, r):
        for j in range(0, c):
            # print column [j]
            if numpy.std(column[j]) != 0:
                # It is the python version of the R:CI_Test[n,i]<- abs((MATRIX_T[n,ph[i]]- MEAN[i])/SD2[i])
                table_train[i][j] = numpy.abs((table_train[i][j] - numpy.mean(column[j])) / numpy.std(column[j]))
                if i < r2:
                    table_test[i][j] = numpy.abs((table_test[i][j] - numpy.mean(column[j])) / numpy.std(column[j]))
            else:
                table_train[i][j] = 0
                if i < r2:
                    table_test[i][j] = 0
    for i in range(0, r):
        for j in range(0, c):
            if table_train[i][j] > cutoff:
                out_train [i][j]=table_train[i][j]
                if i < r2:
                    if table_test[i][j] > cutoff:
                        out_test[i][j] = table_test[i][j]
    tables = [pandas.DataFrame(table_train, columns=v_names, index=Mtrain_names), pandas.DataFrame(table_test, columns=v_names, index=Mtest_names),pandas.DataFrame(out_train, columns=v_names, index=Mtrain_names), pandas.DataFrame(out_test, columns=v_names,index=Mtest_names)]
    return tables
