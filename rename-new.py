import os
from shutil import copyfile
path="d:/3"
os.chdir(path)
for file1 in os.listdir(path):
    if file1.endswith(".mol"):

        #print file1.replace("_00","-")
        newname=file1.replace("_00","-")
        newname2=newname.replace(".mol",".pdb")
        newname3=file1.replace(".mol",".pdb")
        #os.rename(file1, newname)
        #copyfile("rec.pdb",newname2)
        copyfile("rec.pdb", newname3)