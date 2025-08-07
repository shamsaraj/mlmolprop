import rdkit.ML.Descriptors
from rdkit.ML.Descriptors import MoleculeDescriptors
from modules.basic import twodlist
from modules.basic import makecolumn
import rdkit.Chem
from rdkit import Chem
from rdkit.Chem import Descriptors
#from rdkit.Chem import Descriptors
from rdkit.Chem import PandasTools
from rdkit.Chem import AllChem
from rdkit.Chem import SDMolSupplier
from rdkit.Chem.AllChem import  GetMorganFingerprintAsBitVect, GetErGFingerprint
import pandas as pd
from modules.basic import log
DEBUG=False

def desc(moleculesfile, type="molecule",delimiter=','):
    if type == "sdf":
        molecules_list = rdkit.Chem.SDMolSupplier(moleculesfile)
    elif type == "smi":
        molecules_list = rdkit.Chem.SmilesMolSupplier(moleculesfile,delimiter=delimiter,titleLine=True, smilesColumn=0,nameColumn=1)# as suppl1:
    elif type== 'molecule':
        molecules_list = moleculesfile
    if type == type :
        temp = rdkit.Chem.Descriptors
        nms = [x[0] for x in Descriptors._descList]
        names = len(molecules_list) * ["null"]
        descrs = len(molecules_list) * ["null"]
        # nms.remove('MolecularFormula')
        calc = rdkit.ML.Descriptors.MoleculeDescriptors.MolecularDescriptorCalculator(nms)
        for i in range(0, len(molecules_list)):

            descrs[i] = calc.CalcDescriptors(molecules_list[i])
            names[i] = molecules_list[i].GetProp("_Name")
            #print (i,'->',names[i])
        list = [names, nms, descrs]
        return list
def dataframe(list, input_activities, output, TARGET="IC50", type1="file", type2="des"):
    if type1!="file":#A sdf should be provided instead
        SDFFile = type1
        sdftable = rdkit.Chem.PandasTools.LoadSDF(SDFFile)
        print (sdftable.head(),222)
        df = sdftable.loc[:, ["ID", TARGET]]
        #df = df.iloc[:, [1,2]]
        df.rename(columns={"ID": "name"},inplace=True)
        pd.DataFrame.to_csv(df, input_activities, index=False)
    if type2=="finger" or type2=="image":#fingerprint
        df1 = pd.DataFrame(list)
    if type2=="des":#In case that there is a predefined CSV file for activities
        df1 = pd.DataFrame(list[2], index=list[0], columns=list[1])
        #df1.iloc[:, 0]=df1.iloc[:, 0].astype(str)
        #df1['name'] = df1['name'].astype(str)
        #print (df1.info())
        #print list[1]
    merged = df1#just to control error and to make a dataframe for new external set prediction
    if not input_activities == "":
        f = open(input_activities, "rt")
        reader = pd.read_csv(f)
        reader['name'] = reader['name'].astype(str)
        #print (reader.info())
        merged = pd.merge(reader, df1, right_index=True, left_on="name")
        pd.DataFrame.to_csv(merged, output, index=False)
    else:
        pd.DataFrame.to_csv(merged, output, index=False)
    return merged

def tanimoto(molecules_list, list, output, type="fps"):
    import pandas
    from rdkit import DataStructs
    from rdkit.Chem import MACCSkeys
    from rdkit.Chem.Fingerprints import FingerprintMols
    m = len(molecules_list)
    Ttable = twodlist(m, m)
    ms = molecules_list
    if type == "fps":
        #Daylight fingerprint
        #The default set of parameters used by the fingerprinter is: - minimum path size: 1 bond -
        # maximum path size: 7 bonds - fingerprint size: 2048 bits - number of bits set per hash: 2 -
        # minimum fingerprint size: 64 bits - target on-bit density 0.3
        fps = [FingerprintMols.FingerprintMol(x) for x in ms]
    elif type == "maccs":
        #There is a SMARTS-based implementation of the 166 public MACCS keys.
        fps = [MACCSkeys.GenMACCSKeys(x) for x in ms]
    elif type == "ecfp":
        #The default atom invariants use connectivity information similar to those used for the well known ECFP family of fingerprints.
        fps = [AllChem.GetMorganFingerprint(x,6) for x in ms]
    elif type == "fcfp":
        #Feature-based invariants, similar to those used for the FCFP fingerprints, can also be used.
        fps = [AllChem.GetMorganFingerprint(x,6,useFeatures=True) for x in ms]
    i = -1
    for x in fps:
        i = i + 1
        j = -1
        for y in fps:
            #Tanimoto, Dice, Cosine, Sokal, Russel, Kulczynski, McConnaughey, and Tversky. eg: , metric=DataStructs.DiceSimilarity)
            T = DataStructs.FingerprintSimilarity(x, y)#default is Tanimoto
            j = j + 1
            Ttable[i][j] = T
        #Ttable[i][j] += list([T])
    df1 = pandas.DataFrame(Ttable, index=list[0], columns=list[0])
    pandas.DataFrame.to_csv(df1, output)
    return df1

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
    column = makecolumn(table_train, c)
    #column = makecolumn(table_test, c2)
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

def finger(molecules_list, list1, type="fps"):#######deprecated
    import pandas
    from rdkit import DataStructs
    from rdkit.Chem import MACCSkeys
    from rdkit.Chem.Fingerprints import FingerprintMols
    m = len(molecules_list)
    #Ttable = twodlist(m, m)
    ms = molecules_list
    if type == "fps":
        #Daylight fingerprint
        #The default set of parameters used by the fingerprinter is: - minimum path size: 1 bond -
        # maximum path size: 7 bonds - fingerprint size: 2048 bits - number of bits set per hash: 2 -
        # minimum fingerprint size: 64 bits - target on-bit density 0.3
        fps = [FingerprintMols.FingerprintMol(x) for x in ms]
        print (fps)
    elif type == "maccs":
        #There is a SMARTS-based implementation of the 166 public MACCS keys.
        fps = [MACCSkeys.GenMACCSKeys(x) for x in ms]
    elif type == "ecfp":
        #The default atom invariants use connectivity information similar to those used for the well known ECFP family of fingerprints.
        #fps = [AllChem.GetMorganFingerprint(x,6) for x in ms]
        fps= [GetMorganFingerprintAsBitVect (x,6, nBits=512) for x in ms]
        #print fps[0]
        fps=fps[0].ToBitString()
        #y=rdkit.DataStructs.cDataStructs.ExplicitBitVect(fps[0])
        #y= rdkit.DataStructs.cDataStructs.IntSparseIntVect (fps[0])
        #y = fps[0].ToBitString()
        print (55)
        #y= list(fps)
        #print y
        print (66)
        #print y
    elif type == "fcfp":
        #Feature-based invariants, similar to those used for the FCFP fingerprints, can also be used.
        fps = [AllChem.GetMorganFingerprint(x,6,useFeatures=True) for x in ms]
    print (fps)
    df1 = pandas.DataFrame(fps, index=list1[0])#, columns=list[0])
    pandas.DataFrame.to_csv(df1, "D:/pych/ml/eh2/fps.csv")#, index=True)
    print (df1)
    return (df1)
