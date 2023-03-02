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
Prefijos de modificadores:
    - anglerandom
    - blurrandom
    - flip
    - noiserandom
    - randomcombination3
    - scalerandom
    - translatrandom
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
    trainval = train+val
    return (train, test, val, trainval)

# def split(names, div=[]):
#     """
#     Divide una lista con las ponderaciones en div.
#     """
#     tr, vl = div
#     random.shuffle(names)
#     train = names[0:m.floor(len(names)*tr)]
#     val = names[m.floor(len(names)*tr):len(names)]
#     # trainval = train+val
#     # return (train, test, val, trainval)
#     return (train, val)
#     # return (train, test, val)

def saveAnnotations(names, src, dst):
    """
    Dada una lista de nombres "{}.jpg", buscamos sus anotaciones
    correspondientes y las guardamos en una carpeta aparte.
    """
    # src_path = str(Path(src).parent.absolute()/"Annotations")
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

# def writeImageSets(dest, train, val):
#     wr(dest+"train.txt", train)
#     # wr(dest+"test.txt", test)
#     wr(dest+"val.txt", val)
#     # wr(dest+"trainval.txt", train+val)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('source_dir',help="carpeta con la imágenes a procesar.", 
        type = dir_path)
    parser.add_argument('target_dir',help="carpeta en la que volcar el procesado.", 
        type = dir_path)
    parser.add_argument('-d','--div',type=str)
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

    # if args.test is not None:
    #     train, test, val, trainval = split(os.listdir(src+"JPEGImages/"),[0.4,0.2,0.4])
    # train, val = split(os.listdir(src+"JPEGImages/"),[0.6,0.4])
    if args.div=="0":
        div = [0.5,0.2,0.3]
    else:
        div = [(float(x)/100) for x in re.findall("..",args.div)]
    train, test, val, trainval = split(os.listdir(src+"JPEGImages/"),div)

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


if __name__=='__main__':
    main()