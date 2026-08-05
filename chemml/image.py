import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
#from skimage import transform
from svglib.svglib import svg2rlg
from reportlab.graphics import renderPM

def topng (files):
    for file in files:
        drawing = svg2rlg('input/ice.svg')
        renderPM.drawToFile(drawing, 'output/ice.png', fmt='PNG')

def image(path,ext,output):
    imagesall = len(os.listdir(path)) * ["null"]
    imagesname = len(os.listdir(path)) * ["null"]
    a= len(os.listdir(path)) * ["null"]
    i=0
    for files in os.listdir(path):
        if files.endswith(ext):
            image=mpimg.imread(path+files)
            print (files[:-len(ext)], "image to csv")
            imagesall[i]=image
            imagesname[i] = files[:-len(ext)]
            i = i + 1
    imagesallarray=np.array(imagesall)
    #print imagesallarray [0],5555

    plt.imshow(imagesallarray [0], cmap=plt.cm.gray)
    plt.show(block=True)#to show the plot in pycharm#g=transform_f(imagesallarray,imagesname)

    if "augment"=="aug:":
        for n in range (0 , i):
            a[n] = transform_f(imagesallarray[n], imagesname[n])
            #print g[n][1], g[n][0]
            #imagesallarray[n] = g[n][0]
            #imagesname[n] = g[n][1]

        b= [a[x][1]for x in range (0, n)]
        c = [a[x][0] for x in range(0, n)]
    imagesallarray = imagesallarray.reshape((len(imagesname), -1))
    #df2 = pd.DataFrame(c, index =b)#,index=data[:,0]),
    df2 = pd.DataFrame(imagesallarray, index=imagesname)  # ,index=data[:,0]),
    pd.DataFrame.to_csv(df2, output, index=False)
    return df2

#plt.imshow(imgs3, cmap=plt.cm.gray)
#plt.show(block=True)#to show the plot in pycharm


#https://www.programcreek.com/python/example/96394/skimage.transform.rotate
def transform_f (images,labels):
    img = images
    label = labels
    print (img,11111111111111)
    #img = img[0]  # Remove channel axis for skimage manipulation
    print (img[0] ,22222222222222)

    plt.imshow(img, cmap=plt.cm.gray)
    plt.show(block=True)#to show the plot in pycharm #g=transform_f(imagesallarray,imagesname)result=[img,label]

    # Rotate
    img = transform.rotate(img, angle=np.random.uniform(-45, 45),
                           resize=True, mode='constant', cval=1)

    plt.imshow(img, cmap=plt.cm.gray)
    plt.show(block=True)

    #  Scale
    img = transform.rescale(img, scale=np.random.uniform(0.5, 1.5),
                            mode='constant', cval=1)

    plt.imshow(img, cmap=plt.cm.gray)
    plt.show(block=True)

    # Translate
    h, w = img[0].shape
    img_size =[150,300]
    if h >= img_size[0] or w >= img_size[1]:
        img1 = transform.resize(img, output_shape=img_size, mode='constant')
        img1 = img.astype(np.float32)
    else:
        img1_canvas = np.zeros(img_size, dtype=np.float32)
        ymin = np.random.randint(0, img_size[0] - h)
        xmin = np.random.randint(0, img_size[1] - w)
        img1_canvas[ymin:ymin+h, xmin:xmin+w] = img
        img1 = img1_canvas

    from skimage.transform import warp
    from skimage.transform import SimilarityTransform
    tform = SimilarityTransform(translation=(50, 50),scale=1,rotation=0)
    warped = warp(img, tform)

    plt.imshow(warped, cmap=plt.cm.gray)
    plt.show(block=True)#to show the plot in pycharm#g=transform_f(imagesallarray,imagesname)result=[img,label]

    img = img[np.newaxis, :]  # Add the bach channel back
    
    result=[img,label]
    return result

def highlight():
    import rdkit.Chem
    from rdkit.Chem import rdDepictor
    from rdkit.Chem.Draw import rdMolDraw2D

    m = rdkit.Chem.MolFromSmiles('[nH]1cnc2cncnc21')
    rdDepictor.Compute2DCoords(m)
    from IPython.display import SVG
    drawer = rdMolDraw2D.MolDraw2DSVG(400, 200)
    drawer.DrawMolecule(m)
    drawer.FinishDrawing()
    svg = drawer.GetDrawingText().replace('svg:', '')
    SVG(svg)
    highlight = list(m.GetSubstructMatch(rdkit.Chem.MolFromSmarts('c1ncoc1'))) + [0, 1]
    l = list(m.GetSubstructMatch(rdkit.Chem.MolFromSmarts('ccn')))
    # l2=len(l)
    colors1 = {}
    for i in l:
        colors1[i] = (1, 0.35, 0.35)
    print (colors1)
    drawer = rdMolDraw2D.MolDraw2DSVG(400, 200)

    opts = drawer.drawOptions()
    for i in range(m.GetNumAtoms()):
        opts.atomLabels[i] = m.GetAtomWithIdx(i).GetSymbol() + str(i)

    drawer.DrawMolecule(m, highlightAtoms=l, highlightAtomColors=colors1, highlightBonds=l, highlightBondColors=colors1)
    drawer.FinishDrawing()
    svg = drawer.GetDrawingText().replace('svg:', '')
    SVG(svg)
