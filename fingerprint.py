from rdkit.Chem.rdMolDescriptors import *
from rdkit.Chem import Descriptors
from rdkit.Chem.rdmolops import RDKFingerprint
from rdkit.Chem.AtomPairs.Sheridan import GetBPFingerprint
from rdkit.Chem.EState.Fingerprinter import FingerprintMol
from rdkit.Avalon.pyAvalonTools import GetAvalonFP #GetAvalonCountFP  #int vector version
from rdkit.Chem.AllChem import  GetMorganFingerprintAsBitVect, GetErGFingerprint
from rdkit.DataStructs.cDataStructs import ConvertToNumpyArray
import rdkit.DataStructs.cDataStructs
from rdkit.Chem import MACCSkeys
from rdkit.Chem.Fingerprints import FingerprintMols
import numpy as np
import pandas as pd
from modules.basic import log

#to disable RDkit warnings
from rdkit import RDLogger 
RDLogger.DisableLog('rdApp.*') 

import warnings
warnings.filterwarnings("ignore")

DEBUG=False
condition=True
bit = {}
def ExplicitBitVect_to_NumpyArray(bitvector):
    bitstring = bitvector.ToBitString()
    intmap = map(int, bitstring)
    return np.array(list(intmap))

class fingerprint():
    def __init__(self, fp_fun, name):
        self.fp_fun = fp_fun
        self.name = name
        self.x = []

    def apply_fp(self, mols):
        for mol in mols:
            fp = self.fp_fun(mol)
            if isinstance(fp, tuple):
                fp = np.array(list(fp[0]))
            if isinstance(fp, rdkit.DataStructs.cDataStructs.ExplicitBitVect):
                fp = ExplicitBitVect_to_NumpyArray(fp)
            if isinstance(fp,rdkit.DataStructs.cDataStructs.IntSparseIntVect):
                fp = np.array(list(fp))

            self.x += [fp]

            #if (str(type(self.x[0])) != "<class 'numpy.ndarray'>"):
                #print("WARNING: type for ", self.name, "is ", type(self.x[0]))

def make_fingerprints(data,data_list, length = 256, path="", verbose=False, type_f="ECFP4"):
    
    
    if type_f=="all":
        fp_list = [
         #fingerprint(lambda x : GetBPFingerprint(x, fpfn=AtomPair),
         #            "Physiochemical properties (1996)"), ##NOTE: takes a long time to compute
         fingerprint(lambda x : GetHashedAtomPairFingerprintAsBitVect(x, nBits = length),
                     "Atom pair (1985)"),
         fingerprint(lambda x : GetHashedTopologicalTorsionFingerprintAsBitVect(x, nBits = length),
                     "Topological torsion (1987)"),
         fingerprint(lambda x : GetMorganFingerprintAsBitVect(x, 3, nBits = length, useFeatures=True),
                     "Morgan circular FCFP"),
         fingerprint(lambda x: GetMorganFingerprintAsBitVect(x, radius=2, nBits=length, useFeatures=False, bitInfo=bit),
                    "Morgan circular ECFP"),
         fingerprint(FingerprintMol, "Estate (1995)"),
         fingerprint(lambda x: GetAvalonFP(x, nBits=length),
                    "Avalon bit based (2006)"),
         fingerprint(lambda x: np.append(GetAvalonFP(x, nBits=length), Descriptors.MolWt(x)),
                    "Avalon+mol. weight"),
         fingerprint(lambda x: GetErGFingerprint(x), "ErG fingerprint (2006)"),
         fingerprint(lambda x : RDKFingerprint(x, fpSize=length),
                     "RDKit fingerprint"),
         fingerprint(lambda x: GetMACCSKeysFingerprint(x),
                    "MACCS"),
         fingerprint(lambda x: FingerprintMols.FingerprintMol(x),
                    "Daylight fingerprint"),
        ]
    elif type_f == "ECFP4":
        try:
            RDLogger.DisableLog('rdApp.*') 
            fp_list=[fingerprint(lambda x :  GetMorganFingerprintAsBitVect(x, radius=2, nBits = length, useFeatures=False, bitInfo=bit), "Morgan circular ECFP")]

        except:
            print ("error in fingerprnt module")
    elif type_f == "MACCs":
        try:
            RDLogger.DisableLog('rdApp.*') 
            fp_list=[fingerprint(lambda x: GetMACCSKeysFingerprint(x),
                    "MACCS")]

        except:
            print ("error in fingerprnt module")
    for fp in fp_list:
        RDLogger.DisableLog('rdApp.*')
        if (verbose): print("doing", fp.name)
        fp.apply_fp(data)
    if condition==True:
        #print fp_list,11111111111111
        #print fp_list[2].name,22222222222222
        #print fp_list[2].fp_fun,3333333333333
        #print fp_list[2].x,4444444444444444
        #print fp_list[2].x[0],555555555555555
        #print len(fp_list[2].x[0]), 66666666666
        #info={}
        #fp3= GetMorganFingerprint(data[0], 3, bitInfo=info)
        #print fp3,77777777777777
        #print len(fp3.GetNonzeroElements())
        #print dir(fp_list[2].x[0])
        #print dir(fp3)
        #print fp3.ToBinary(),888888888888888
        #print fp3.GetTotalVal(), 888888888888888
        #print fp3.GetLength(), 888888888888888
        #test={847961216: ((2, 0), (18, 0))}
        #test2=fp3[0].ToBitString()
        #print test2
        #print info
        df = pd.DataFrame(data=fp_list[0].x, index=data_list[0])## 0 is very important index here
        #pd.DataFrame.to_csv(df, path+"fps.csv")  # , index=True)## commented out for array jobs  
    return [df, bit]# bit has to be changed to a list container for dictionaries

def make_fingerprints2(data,data_list, length = 256, verbose=False,type_f="ECFP4"):
    if type_f== "ECFP4":
        fp_list = [fingerprint(lambda x: GetMorganFingerprintAsBitVect(x, 2, nBits=length, useFeatures=False),
                        "Morgan circular ECFP")]
    elif type_f=="MACCs":
        fp_list=[fingerprint(lambda x: GetMACCSKeysFingerprint(x),
                    "MACCS")]

    for fp in fp_list:
        if (verbose): print("doing", fp.name)
        fp.apply_fp(data)
    #df = pd.DataFrame(data=fp_list[0].x, index=data_list[0])##
    #pd.DataFrame.to_csv(df, path+"fps.csv")  # , index=True)## commented out for array jobs 
    return df
