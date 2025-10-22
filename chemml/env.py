import os
import sys
#To solve DLL load failed when Python interpreter is not started from the command line, for PyCharm
# This is only for PyCharm

DEBUG=False
from basic import log
def env ():
    os.environ["PATH"] += r";D:\anaconda2\envs\py27"
    os.environ["PATH"] += r";D:\anaconda2\envs\py27\Library\mingw-w64\bin"
    os.environ["PATH"] += r";D:\anaconda2\envs\py27\Library\usr\bin"
    os.environ["PATH"] += r";D:\anaconda2\envs\py27\Library\bin"
    os.environ["PATH"] += r";D:\anaconda2\envs\py27\Scripts"
    
def env2 ():
    os.environ["PATH"] += r";D:\anaconda2\envs\my-rdkit-env"
    os.environ["PATH"] += r";D:\anaconda2\envs\my-rdkit-env\Library\mingw-w64\bin"
    os.environ["PATH"] += r";D:\anaconda2\envs\my-rdkit-env\Library\usr\bin"
    os.environ["PATH"] += r";D:\anaconda2\envs\my-rdkit-env\Library\bin"
    os.environ["PATH"] += r";D:\anaconda2\envs\my-rdkit-env\Scripts"

def env3 ():
    os.environ["PATH"] += r";D:\Anaconda2\envs\py37"
    os.environ["PATH"] += r";D:\Anaconda2\envs\py37\Library\mingw-w64\bin"
    os.environ["PATH"] += r";D:\Anaconda2\envs\py37\Library\usr\bin"
    os.environ["PATH"] += r";D:\Anaconda2\envs\py37\Library\bin"
    os.environ["PATH"] += r";D:\Anaconda2\envs\py37\Scripts"
    #os.environ["PATH"] += r";D:\Anaconda2\envs\py37\Lib\site-packages\rdkit"
    #os.environ["PATH"] += r";C:\Users\ShamsaraJ\AppData\Local\conda\conda\envs\py37"

def env4 ():
    os.environ["PATH"] += r";D:\Anaconda2\envs\py37"
    os.environ["PATH"] += r";D:\Anaconda2\envs\py37\Library\mingw-w64\bin"
    os.environ["PATH"] += r";D:\Anaconda2\envs\py37\Library\usr\bin"
    os.environ["PATH"] += r";D:\Anaconda2\envs\py37\Library\bin"
    os.environ["PATH"] += r";D:\Anaconda2\envs\py37\Scripts"
    #os.environ["PATH"] += r";D:\Anaconda2\envs\py37\Lib\site-packages\rdkit"
    #os.environ["PATH"] += r";C:\Users\ShamsaraJ\AppData\Local\conda\conda\envs\py37"

    
