from multiprocessing import freeze_support
import numpy as np
import pandas as pd
#import winsound
from modules.basic import *
from sklearn.ensemble import RandomForestClassifier  # Forests of randomized trees
from sklearn.neural_network import MLPClassifier
from sklearn import svm#, datasets
import matplotlib.pyplot as plt
import numpy as np

def lime2(model, x, feature_names, y, type1="regression", n=8, i=0, graph=False):
    import lime
    import lime.lime_tabular
    # lime
    tr = np.array(x)
    # ts=np.array(set1[2])
    explainer = lime.lime_tabular.LimeTabularExplainer(tr, feature_names=feature_names, verbose=True, mode=type1)
    list1=len(tr)*["null"]
    for i in range (i,len(tr)):
        if type1== 'classification':
            print ("lime is processing molecule", x.index[i])
            print ("...")
            #print x.iloc[i,]
            exp = explainer.explain_instance(tr[i], model.predict_proba, num_features=n)
            
            
            # print exp
            #print (dir(exp))

            print ("exp.predict_proba",exp.predict_proba)

            #print "exp.local_exp",exp.local_exp
            print ("y[i]",y[i])
            print ("exp.score",exp.score)
            print ("exp.as_list()",exp.as_list())

                       
            list1[i]=[x.index.tolist()[i],y[i],exp.predict_proba,exp.as_list(),exp]
        
        elif type1== 'regression':
            print ("lime is processing molecule", x.index[i])
            print ("...")
            #print x.iloc[i,]
            exp = explainer.explain_instance(tr[i], model.predict, num_features=n)
            #exp.show_in_notebook(show_table=True)
            # print exp
            #print (dir(exp))

            print ("exp.predicted_value",exp.predicted_value)
            

            #print "exp.local_exp",exp.local_exp
            print ("y[i]",y[i])
            print ("exp.score",exp.score)
            print ("exp.as_list()",exp.as_list())
            #exp.as_pyplot_figure()
            #plt.show()
            list1[i]=[x.index.tolist()[i],y[i],exp.predicted_value,exp.as_list(),exp]
        if graph:
            exp.show_in_notebook(show_table=True)
            exp.as_pyplot_figure()
            #plt.show()
    return list1





# skater

def main(model, x, feature_names, type1="reg"):
    from skater.core.explanations import Interpretation
    from skater.model import InMemoryModel
    from skater.core.local_interpretation.lime.lime_tabular import LimeTabularExplainer
    if __name__ == '__main__':
        freeze_support()
    interpreter = Interpretation(x, feature_names=feature_names)
    if type1 == "reg":
        pyint_model = InMemoryModel(model.predict, examples=x)
    plot = interpreter.feature_importance.plot_feature_importance(pyint_model)
    # plot.show()


# main(model[2],set1[0],set1[4])

# treeinterpreter
def tree(model, x, feature_name ,y):
    from treeinterpreter import treeinterpreter as ti
    impact = ti.predict(model, x)
    prediction, bias, contributions = impact
    list1=len(x)*["null"]
    list2 = len(x) * ["null"]
    for i in range(len(x)):
        print ("Instance", i)
        print ("rela value", y[i])
        print ("prediction", prediction[i])
        print ("Bias (trainset mean)", bias[i])
        print ("Feature contributions:")
        for CC, feature in sorted(zip(contributions[i],
                                      feature_name),
                                  key=lambda x: -abs(x[0])):
            print (feature, round(CC, 3))
        #model.predict(x)[i]
        list1[i]= (y[i],prediction[i]), sorted(zip(contributions[i],feature_name),key=lambda x: -abs(x[0]))
        list2[i]=[(y[i],prediction[i]), zip(contributions[i],feature_name)]
        print ("-" * 40)
    df=pd.DataFrame(list1)
    fe_name= len(feature_name)*[""]
    fe_val=len(feature_name)*[""]
    list3=twodlist(len(x),len(feature_name))
    #list7=twodlist(len(x),len(feature_name))
    list4=len(feature_name)*[""]
    #list5=len(feature_name)*[""]
    #print list1[10][1][4][0]
    for j in range (len(x)):
        for k in range (len(feature_name)):
            fe_name[k]= list2[j][1][k][1]
            fe_val[k]=list2[j][1][k][0]
            list3[j][k]=[fe_name[k],fe_val[k]]
            #list7[j][k] = [fe_name[k], abs(fe_val[k])]
            list4[k] = np.mean(list3[j][k][1])
            #list5[k]=[fe_name[k],list4[k]]
        list6=sorted(zip(list4,fe_name),key=lambda x: -abs(x[0]))
    #print list3[0][0][1]
    #print list3[1][0][1]
    #print list5
    #print list6 #Global interpretability

    #print df
    result =[x.index.tolist()[i],list1,df,list6]
    return result





# print set1[0].index
# np.transpose(impact)
# df5=pd.DataFrame (impact,index=set1[0].index)
# print impact, "impact"
# print len (impact)
# print len (impact [0])
# print len (impact[1])
# print len (impact[2])
# impact2=impact.loc[:set1[4]]
# impact3=impact2.loc[:model[0]["Variable Importance"][0]]
# print "accuracy_score_LOO=",model[0]["accuracy_score_LOO"],"accuracy_score_train=",model[0]["accuracy_score_train"],"accuracy_score_test=",model[0]["accuracy_score_test"]
# print "mean", mean(model[0]["R2_test"]), mean(model[1]["q2"]), mean(model[0]["R2"]), mean(model[0]["Pearson"][0]), mean(model[0]["Q2F2"]), mean(model[0]["R2"]- model[1]["q2"]),"********************"


# partial
def partial(x, y, VI, n=4,type="reg"):
    from sklearn.ensemble import GradientBoostingRegressor
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.ensemble.partial_dependence import partial_dependence, plot_partial_dependence
    import matplotlib.pyplot as plt  ################3
    VI2 = VI.sort_values(by=[0], ascending=False)
    nn = VI2.index
    nn = pd.Index.tolist(nn)
    tr_y = np.array(y)
    tr = np.array(x[nn])  ##########
    if type == "reg":
        gb = GradientBoostingRegressor(n_estimators=10).fit(tr, tr_y)
    elif type == "class":
        gb = GradientBoostingClassifier(n_estimators=10).fit(tr, tr_y)
    print (nn[0:n])
    ff = VI.iloc[0:4, 0]
    fig = plot_partial_dependence(gb, tr, features=nn[0:n + 1], feature_names=nn[0:n + 1])

    # plt.interactive(False)
    # fig.show()
    #plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=0.1)
    plt.tight_layout()
    plt.show()
    # plt.interactive(False)
    # print fig,9999999
    # raw_input("pause")


def clas_graph(x,y,feathres_name,fe=[0,1]):
    #iris = datasets.load_iris()
    #X = iris.data[:, :2]  # we only take the first two features. We could
    #print X                      # avoid this ugly slicing by using a two-dim dataset
    #y = iris.target
    #print set[4]
    #print set[0]

    fe=[feathres_name[feathres_name.index('Do you have a job stress?')],feathres_name[feathres_name.index('1-methyl-4-(1-methylethyl) benzene')]]
    #print fe
    X = x[fe]
    X = np.array(X)


    #X2=np.array(set[0])
    #X2 = X[:, :2]
    #print X


    h = .02  # step size in the mesh

    # we create an instance of SVM and fit out data. We do not scale our
    # data since we want to plot the support vectors
    C = 1.0  # SVM regularization parameter

    svc = svm.SVC(kernel='linear', C=C).fit(X, y)
    rbf_svc = svm.SVC(kernel='rbf', gamma=0.7, C=C).fit(X, y)
    poly_svc = svm.SVC(kernel='poly', degree=3, C=C).fit(X, y)
    lin_svc = svm.LinearSVC(C=C).fit(X, y)
    rf = RandomForestClassifier(max_depth=2, random_state=10).fit(X,y)
    nn= MLPClassifier().fit(X,y)

    # create a mesh to plot in
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))

    # title for the plots
    titles = ['SVC with linear kernel',
              'LinearSVC (linear kernel)',
              'SVC with RBF kernel',
              'SVC with polynomial (degree 3) kernel',"rf","nn"]

    print (6)
    for i, clf in enumerate((svc, lin_svc, rbf_svc, poly_svc,rf,nn)):#
        # Plot the decision boundary. For that, we will assign a color to each
        # point in the mesh [x_min, m_max]x[y_min, y_max].
        plt.subplot(2, 3, i + 1)
        plt.subplots_adjust(wspace=0.4, hspace=0.4)

        Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])

        # Put the result into a color plot
        Z = Z.reshape(xx.shape)
        print (11,xx[0])
        print (22,yy[0])
        print (33,Z[0])
        plt.contourf(xx, yy, Z, cmap=plt.cm.Paired, alpha=0.8)

        # Plot also the training points
        plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Paired)
        plt.xlabel('Sepal length')
        plt.ylabel('Sepal width')
        plt.xlim(xx.min(), xx.max())
        plt.ylim(yy.min(), yy.max())
        plt.xticks(())
        plt.yticks(())
        plt.title(titles[i])

    plt.show()
