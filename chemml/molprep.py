#molecules will be prapared using RDKIT
#conda install -c conda-forge cairo
#pip install ipython

from modules.env import *
env4()

import rdkit.Chem
from rdkit.Chem import rdPartialCharges
from rdkit.Chem import AllChem
from rdkit.Chem import Draw
from rdkit.Chem.SaltRemover import SaltRemover
#from rdkit.Chem.Draw import IPythonConsole
from IPython.display import SVG###################################it could not be imported in kolblab env
##################################################33
from rdkit.Chem import rdDepictor
from rdkit.Chem.Draw import rdMolDraw2D
#import cairosvg
import os

#to disable rdkit warning
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

def moltosvg(mol,molSize=(450,150),kekulize=True):
    import time
    print(time.asctime())
    mc = rdkit.Chem.Mol(mol.ToBinary())
    if kekulize:
        try:
            rdkit.Chem.Kekulize(mc)
        except:
            mc = rdkit.Chem.Mol(mol.ToBinary())
    if not mc.GetNumConformers():
        rdDepictor.Compute2DCoords(mc)
    drawer = rdMolDraw2D.MolDraw2DSVG(molSize[0],molSize[1])
    drawer.DrawMolecule(mc)
    drawer.FinishDrawing()
    svg = drawer.GetDrawingText()
    # It seems that the svg renderer used doesn't quite hit the spec.
    # Here are some fixes to make it work in the notebook, although I think
    # the underlying issue needs to be resolved at the generation step
    return svg.replace('svg:','')
def mol_prep (moleculefile):
    try:
        #print (1)
        m1 = moleculefile
        #print (m1.GetProp("_Name"), "to 3d")
        #print m1
        #try:
        #print m1.GetNumAtoms()
        #remover = SaltRemover(defnData="[Cl,Br]")
        #print (2)
        remover = SaltRemover()###################################
        #print len(remover.salts)
        #print (3)
        res = remover.StripMol(m1)##########################
        #res is not None
        #print res.GetNumAtoms()
        #print (4)
        m2 = rdkit.Chem.AddHs(res)#########################################
        #m2=m1##########################################
        #print (rdkit.Chem.MolToMolBlock(m2))
        #print (5)
        AllChem.Compute2DCoords(m2)####
        #print (6)
        rdPartialCharges.ComputeGasteigerCharges(m2)
        #print float(m2.GetAtomWithIdx(0).GetProp('_GasteigerCharge'))
        #print (7)
        AllChem.EmbedMolecule(m2)
        try:
            AllChem.UFFOptimizeMolecule(m2)
        except:
            pass
        #AllChem.EmbedMolecule(m2, AllChem.ETKDG())
        #AllChem.MMFFOptimizeMolecule(m2)
        #print (8)
        m3 = rdkit.Chem.RemoveHs(m2)
        #except:
            #print ("AllChem.UFFOptimizeMolecule(m2) ValueError: Bad Conformer Id")
            #m3 = m2
    except:
        
        m2 = rdkit.Chem.MolFromSmiles("c1ccccc1") 
        m2.SetProp("_Name","dummy")
        m2 = rdkit.Chem.AddHs(res)#########################################
        AllChem.Compute2DCoords(m2)####
        rdPartialCharges.ComputeGasteigerCharges(m2)
        #print float(m2.GetAtomWithIdx(0).GetProp('_GasteigerCharge'))
        #print (7)
        AllChem.EmbedMolecule(m2)
        m3 = rdkit.Chem.RemoveHs(m2)
        print ("error in molprop type 8")
    return m3

def mol_enumerate (moleculesfile,output1, output2, format="sdf",moleculefiles2="", image=False, imagepath="D:/pych/ml/MR1/images", delimiter=',',print_int=50):
    input_molecules=[]
    #help(rdkit.Chem.SmilesMolSupplier)
    if format == "sdf":
        with rdkit.Chem.SDMolSupplier(moleculesfile) as suppl1:
            for mol1 in suppl1:
                if mol1 is not None:
                    pass
                else:
                    mol1 = rdkit.Chem.MolFromSmiles("c1ccccc1")
                    mol1.SetProp("_Name","dummy")
                    print ("error in molprop type 9")
                input_molecules.append(mol1)
        if moleculefiles2 != "":
            input_molecules2 = rdkit.Chem.SDMolSupplier(moleculefiles2)
        
    elif format == "smi":
        if 1==1:#with
        
            suppl1= rdkit.Chem.SmilesMolSupplier(moleculesfile,delimiter=delimiter,titleLine=True, smilesColumn=0,nameColumn=1)# as suppl1:
            n=0
            for mol1 in suppl1:
                #print (mol1)
                if mol1 is not None:
                    pass
                else:
                    mol1 = rdkit.Chem.MolFromSmiles("c1ccccc1")
                    mol1.SetProp("_Name","dummy")
                    print ("Molecule number:"+str(n)+" error in molprop type 9")
                input_molecules.append(mol1)
                n=n+1
        if moleculefiles2 != "":
            input_molecules2 = rdkit.Chem.SDMolSupplier(moleculefiles2)
        if moleculefiles2 != "":
            input_molecules2 = rdkit.Chem.SmilesMolSupplier(moleculefiles2,delimiter=delimiter,titleLine=True, smilesColumn=0,nameColumn=1)
    print (" len (input_molecules) ", len (input_molecules) )
    input_molecules_list = len (input_molecules) * ["null"]
    if moleculefiles2 != "":
        input_molecules_list2 = len(input_molecules2) * ["null"]
    i = 0
    w1 = rdkit.Chem.SDWriter(output1)
    w2 = rdkit.Chem.SDWriter(output2)
    #RDimage(input_molecules)
    for mol in input_molecules:
        #print (mol)
        #print (i , mol.GetProp("_Name"), "to 3d")################
        if i%print_int==0 or i == 0:
            print (i , mol.GetProp("_Name"), "to 3d")
        preparedmolecule=mol_prep(mol)
        input_molecules_list[i] = preparedmolecule
        i = i + 1
        w1.write(preparedmolecule)
        p = (preparedmolecule)
        AllChem.Compute2DCoords(p)
        w2.write(p)
    w1.close()
    w2.close()
    if image == True:
        RDimage(input_molecules_list,imagepath)####################################################################################################
    return input_molecules_list

def RDimage (ms,dir):
    #rdkit.Chem.SmilesWriter(ms, "D:/pych/ml/dr3/newtest.txt")
    from rdkit.Chem import rdDepictor
    #Draw.MolDrawing.dotsPerAngstrom = 60
    #Draw.MolDrawing.atomLabelFontSize = 12
    #DrawingOptions.bondLineWidth = 3.0
    ms1=ms
    for m in ms1:
        AllChem.Compute2DCoords(m)
    #img = Draw.MolsToGridImage(ms1, molsPerRow=4, subImgSize=(300, 300), legends=[x.GetProp("_Name") for x in ms1])
    #img.save(dir + "molgrid.png")
    i=0
    for m in ms1:
        i = i + 1
        print (i)
        print (m.GetProp("_Name"), "to svg")
        #AllChem.Compute2DCoords(m)
        rdkit.Chem.RemoveHs(m)
        #Draw.MolToFile(m,fileName=dir + m.GetProp("_Name") + ".png", size=(300, 300), kekulize=True,
                   #wedgeBonds=False, fitImage=False, options=None, canvas=None, imageType=None,
                   #highlightAtoms=None)
        #Draw.MolToFile(m, dir + m.GetProp("_Name") + "-2.png")
        rdkit.Chem.SanitizeMol(m)
        rdkit.Chem.Kekulize(m)
        #Draw.MolToFile(m, dir + m.GetProp("_Name") + ".png")
        #Draw.MolToImage(m)
        rdDepictor.Compute2DCoords(m)
        drawer = rdMolDraw2D.MolDraw2DSVG(300, 150)#########################
        #drawer.SetFontSize(0.8)
        drawer.DrawMolecule(m)
        drawer.FinishDrawing()
        svg = drawer.GetDrawingText().replace('svg:', '')
        name=dir + m.GetProp("_Name") + ".svg"
        with open(name, 'w') as f:
            f.write(svg)#to write svg file on disk
        log("test,",DEBUG)
        SVG(svg)
        from svglib.svglib import svg2rlg
        from reportlab.graphics import renderPM
        drawing = svg2rlg(name)
        #The png and jpg are sometimes bugy
        #renderPM.drawToFile(drawing, name[:-4] + ".png", fmt='png')
        #renderPM.drawToFile(drawing, name[:-4]+".jpg", fmt='jpg')
        renderPM.drawToFile(drawing, name[:-4] + ".tif", fmt='tif')
        renderPM.drawToFile(drawing, name[:-4] + ".gif", fmt='gif')


        #SVG(moltosvg(mol))
        #cairosvg.svg2png(svg, write_to=dir + m.GetProp("_Name") + '.png') ##Compatible with python 2.7
        #os.remove()

        #cairosvg.svg2pdf(svg, write_to=dir + m.GetProp("_Name") + '.pdf')
































#Scratch

#mol = rdkit.Chem.MolFromSmiles("CN1CCN(S(=O)(C2=CC=C(OCC)C(C3=NC4=C(N(C)N=C4CCC)C(N3)=O)=C2)=O)CC1")

#rdkit.Chem.MolToSmiles()
#rdkit.Chem.SmilesWriter(ms,"test.txt")
#highlight examples:
#drawer.DrawMolecule(m,highlightAtoms=m.GetSubstructMatch(Chem.MolFromSmarts('c1ncoc1')))

#highlight=list(m.GetSubstructMatch(Chem.MolFromSmarts('c1ncoc1'))) + [10,1]

#highlight=[10,1,4,13]
#colors={1:(1,0,0), 4:(0,1,0), 13:(0,0,1), 10:(1,0,1)}
#drawer.DrawMolecule(m,highlightAtoms=highlight,highlightAtomColors=colors)


#eg
#Draw.MolToFile( mol, fname+"png" )
#Draw.MolToFile( mol, "temp.svg" )
#cairosvg.svg2png( url='./temp.svg', write_to= "svg_"+

def mol_enumerate_old (moleculesfile): ####should be updated according to the smiles def #deprecated
    input_molecules = rdkit.Chem.SDMolSupplier(moleculesfile)
    number_input_molecules= 0
    for mol in input_molecules:
        mol_prep(mol)######
        number_input_molecules= number_input_molecules +1
    input_molecules_list = number_input_molecules * ["null"]
    i = 0
    w = rdkit.Chem.SDWriter('D:/qsar/prepared-structures-3D.sdf')
    for mol in input_molecules:
        input_molecules_list[i] = mol_prep(mol)
        i = i + 1
        w.write(mol)
    w.close()
    return input_molecules_list









