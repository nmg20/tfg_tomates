import os
import argparse
import math as m
from pathlib import Path
import random
import shutil
import warnings
import re
# 40 - 20 - 40

"""
Uso:
    - python divider.py ../../datasets/T1280x720/ . 0
"""

"""
Idea:
    - se le pasa un directorio del que leer las imágenes y las anotaciones
    - se le pasa un directorio maestro en el que volcar las divisiones
    - parsear cada nombre para extraer las imágenes originales y separar los prefijos y sufijos
    - 1er paso:
        -> hacer la división de forma aleatoria
        -> no comprobar si val tiene sólo una versión de cada imágen.

"""

def dir_path(path):
    if os.path.isdir(path):
        return path
    else:
        # warnings.warn(f"El directorio {path} no existe, pero te lo creo al toque mi rey.",stacklevel=2)
        os.mkdir(path)
        return path

def validate_file(f):
    if not os.path.exists(f):
        # Argparse uses the ArgumentTypeError to give a rejection message like:
        # error: argument input: x does not exist
        raise argparse.ArgumentTypeError("{0} does not exist".format(f))
        
    return f

def wr(file, list):
    """
    Dado un archvio file, escribe los nombres de los archivos en
    list desde n1 hasta n2.
    """
    fil = Path(file)
    fil.touch(exist_ok=True)
    with open(file, "w") as f:
        for i in list:
            f.write(i.split(".")[0]+"\n")
            # f.write(i[0]+"\n")

def split(names, div=[]):
    """
    Divide una lista con las ponderaciones en div.
    """
    tr, ts, vl = div
    random.shuffle(names)
    train = names[0:m.floor(len(names)*tr)]
    test = names[m.floor(len(names)*tr):m.floor(len(names)*(tr+ts))]
    val = names[m.floor(len(names)*(1-vl)):len(names)]
    # trainval = train+val
    # return (train, test, val, trainval)
    return (train, test, val)

def split_aux(names, div):
    return [split(names, d) for d in div]

def saveAnnotations(names, src, dst):
    """
    Dada una lista de nombres "{}.jpg", buscamos sus anotaciones
    correspondientes y las guardamos en una carpeta aparte.
    """
    for name in names:
        name = str(Path(name).stem)
        og = (src+name+".xml")
        tg = (dst+name+".xml")
        shutil.copy(og, tg)

def saveImages(names, src, dst):
    """
    Dada una lista de nombres "{}.jpg", se guarda una copia en dest.
    """
    for name in names:
        og = src+name
        tg = dst+name
        shutil.copy(og, tg)

def writeImageSets(dest, train, test, val):
    wr(dest+"train.txt", train)
    wr(dest+"test.txt", test)
    wr(dest+"val.txt", val)
    wr(dest+"trainval.txt", train+val)

def make_dirs(dest, annsrc, anndst, imgsrc, imgdst, train, val, test):
    if not os.path.exists(dest+"imagesets/"):
        os.mkdir((dest+"imagesets/"))
    writeImageSets(dest+"imagesets/",train, test, val)
    # writeImageSets(dest+"imagesets/",train, val)
    if not os.path.exists(dest+"images/"):
        os.mkdir((dest+"images/"))
        os.mkdir((dest+"images/train/"))
        os.mkdir((dest+"images/test/"))
        os.mkdir((dest+"images/val/"))
    saveImages(train, imgsrc, imgdst+"train/")
    saveImages(test, imgsrc, imgdst+"test/")
    saveImages(val, imgsrc, imgdst+"val/")
    if not os.path.exists(dest+"annotations/"):
        os.mkdir((dest+"annotations/"))
        os.mkdir((dest+"annotations/train/"))
        os.mkdir((dest+"annotations/test/"))
        os.mkdir((dest+"annotations/val/"))
    saveAnnotations(train, annsrc, anndst+"train/")
    saveAnnotations(test, annsrc, anndst+"test/")
    saveAnnotations(val, annsrc, anndst+"val/")

def make_annotations(dest, name, annsrc, train, val, test):
    anndst = f"{dest}Annotations/{name}"
    if not os.path.exists(f"{dest}Annotations/{name}"):
        os.makedirs(anndst)
        os.makedirs(f"{anndst}/train/")
        os.makedirs(f"{anndst}/test/")
        os.makedirs(f"{anndst}/val/")
    saveAnnotations(train, annsrc, anndst+"/train/")
    saveAnnotations(test, annsrc, anndst+"/test/")
    saveAnnotations(val, annsrc, anndst+"/val/")

def make_imagesets(dest, name, train, val, test):
    if not os.path.exists(f"{dest}ImageSets/{name}/"):
        os.makedirs(f"{dest}ImageSets/{name}/")
    writeImageSets(f"{dest}ImageSets/{name}/",train, test, val)


def main():
    """
    Crea un directorio con anotaciones y otro con los imagesets.
        -> volcar anotaciones a imagesets convirtiendolas en csv en vez de xml
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('source_dir',help="carpeta con la imágenes a procesar.", 
        type = dir_path)
    parser.add_argument('target_dir',help="carpeta en la que volcar el procesado.", 
        type = dir_path)
    # parser.add_argument('-t', '--test', type=int)
    # parser.add_argument('-d','--div',type=str)
    # parser.add_argument('test', help="flag que indica si se establece un conjunto de test.")
    args = parser.parse_args()
    # src = ../../datasets/Tomato_300x300
    # dst = ./tests/
    src = args.source_dir
    dest = args.target_dir
        
    imgsrc = src+"JPEGImages/"
    imgdst = dest+"images/"
    annsrc = src+"Annotations/"
    anndst = dest+"annotations/"

    # if args.test=="0":
    #     div = [[0.8,0.1,0.1],[0.7,0.15,0.15],[0.6,0.2,0.2]]
    divs = [[0.8,0.1,0.1],[0.7,0.15,0.15],[0.6,0.2,0.2]]
    # else:
    #     div = [(float(x)/100) for x in re.findall("..",args.div)]
    # train, test, val, trainval = split_aux(os.listdir(src+"JPEGImages/"),div)
    splitted_dss = split_aux(os.listdir(src+"JPEGImages/"),divs)
    for ds, div in zip(splitted_dss, divs):
        train, test, val = ds
        name = "d"+''.join([str(int(x*100)) for x in div])
        # make_dirs(dest,src+"Annotations/",dest+"annotations/",src+"JPEGImages/",
        #     dest+"images/", train, val, test)
        make_annotations(dest, name, annsrc, train, val, test)
        make_imagesets(dest, name, train, val, test)

if __name__=='__main__':
    main()