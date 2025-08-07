import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


from modules.model import rocc

def myplot(df,y,type,n=8,df2="",y2="",y3="",output="plot.jpg"):
    SMALL_SIZE = 8
    MEDIUM_SIZE = 12
    BIGGER_SIZE = 16
    #print list(df.columns.values)
    plt.rc('font', size=SMALL_SIZE)  # controls default text sizes
    plt.rc('axes', titlesize=SMALL_SIZE)  # fontsize of the axes title
    plt.rc('axes', labelsize=MEDIUM_SIZE)  # fontsize of the x and y labels
    plt.rc('xtick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
    plt.rc('ytick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
    #plt.rc('xlabel', fontsize=SMALL_SIZE)  # fontsize of the tick labels
    #plt.rc('ylabel', fontsize=SMALL_SIZE)  # fontsize of the tick labels
    plt.rc('legend', fontsize=SMALL_SIZE)  # legend fontsize
    plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
    for i in range (len(df.columns)):

        plt.subplot(n, n,i+1)
        #plt.subplot(2,2,2)
        #plt.scatter(set1[0]["fr_amide"], set1[1],color="red",s=20,alpha=0.50)
        if type== "scat":
            #plt.scatter(df.iloc[:, i],y,color="blue", s=500, alpha=0.07)
            plt.title(list(df.columns.values)[i], fontsize=MEDIUM_SIZE)

            #plt.yticks(y,["N1", "N0"])
            plt.yticks([1,2])

            #if i == 0 or i == 3:
                #plt.xticks([0, 1])
            #print i
            if i==100:
                plt.scatter(df.iloc[:, i], y, color="blue", s=500, alpha=0.02)
            elif i==300:
                plt.scatter(df.iloc[:, i], y, color="blue", s=500, alpha=0.02)
            elif i==200:
                plt.scatter(df.iloc[:, i], y, color="blue", s=500, alpha=0.02)
            elif i ==300:
                plt.scatter(df.iloc[:, i], y, color="blue", s=500, alpha=0.02)
            else:
                plt.ylabel("Class", fontsize=SMALL_SIZE)
                plt.xlabel("Value", fontsize=SMALL_SIZE)
                plt.scatter(df.iloc[:, i], y, color="#e60073", s=500, alpha=0.02)
            if "test"=="test22":
                plt.scatter(df2.iloc[:, i], y2, color="#000000", s=500, alpha=0.02)





            if "yes" == "yes2":
                if i==3:
                    plt.xticks([0, 1,2,3,4,5,6,7,8,9,10])
                elif i==1:
                    plt.xticks([0, 1, 2, 3])
                elif i==4 or i==5 or i==6:
                    plt.xticks([0, 1, 2, 3, 4])
                else:
                    plt.xticks([0, 1, 2, 3, 4, 5, 6])

        elif type=="his":
            if "angio"=="angi":

                target = "Group"
                df1 = df[df[target] == 1]
                df3 = df[df[target] == 3]
                df2 = df[df[target] == 2]

                plt.hist(df1.iloc[:, i], color="#000000", alpha=1)
                plt.hist(df3.iloc[:, i], color="#49ff33", alpha=0.5)
                plt.hist(df2.iloc[:, i], color="#e67300", alpha=0.5)
            else:
                plt.hist(df.iloc[:, i], color="#e67300", alpha=1)
                if "test"=="test":
                    plt.hist(df2.iloc[:, i], color="#000000", alpha=1)

            plt.title(list(df.columns.values)[i],fontsize=MEDIUM_SIZE)
            plt.ylabel("Number", fontsize=SMALL_SIZE)
            plt.xlabel("", fontsize=MEDIUM_SIZE)

            if "yes"=="yes2":
                if i==3:
                    plt.xticks([0, 1,2,3,4,5,6,7,8,9,10])
                elif i==1:
                    plt.xticks([0, 1, 2, 3])
                elif i==4 or i==5 or i==6:
                    plt.xticks([0, 1, 2, 3, 4])
                else:
                    plt.xticks([0, 1, 2, 3, 4, 5, 6])
                #plt.hist(y, bins=10, color="#39e600")

        elif type == "his2d":
            plt.hist2d(df.iloc[:, i], y, alpha=1.0,cmap=plt.cm.Greens)
            plt.title(list(df.columns.values)[i], fontsize=BIGGER_SIZE)
            plt.ylabel("Class")
            #plt.xlim(0,1)
            #plt.ylim(0,1)
            #plt.yticks([0,1])

            if "yes" == "yes2":
                if i==3:
                    plt.xticks([0, 1,2,3,4,5,6,7,8,9,10])
                elif i==1:
                    plt.xticks([0, 1, 2, 3])
                elif i==4 or i==5 or i==6:
                    plt.xticks([0, 1, 2, 3, 4])
                else:
                    plt.xticks([0, 1, 2, 3, 4, 5, 6])

                #plt.legend()

        elif type == "rocc" :
            from sklearn import metrics
            fpr, tpr, thresholds = metrics.roc_curve(y, df.iloc[:, i], pos_label=1)
            AUC = metrics.auc(fpr, tpr)
            #plt.figure()
            lw = 2
            plt.plot(fpr, tpr, color='darkorange',
                     lw=lw, label='ROC curve (area = %0.2f)' % AUC)
            plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate',fontsize=SMALL_SIZE)
            plt.ylabel('True Positive Rate',fontsize=SMALL_SIZE)
            plt.legend(loc="lower right")
            plt.title(list(df.columns.values)[i], fontsize=MEDIUM_SIZE)

        #plt.xlabel("Number",fontsize=5)
        #plt.ylabel("Frequency",fontsize=5)
        #plt.xticks([1, 2, 3, 4, 5])
        plt.tight_layout(pad=1, w_pad=0, h_pad=0)
        #plt.tight_layout()



    plt.savefig(output, format="jpg", dpi=2400)#,figurewidth='20cm',figureheight="6cm")
    plt.show()



