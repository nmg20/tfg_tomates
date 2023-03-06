import os
import glob
import argparse
import math as m
from pathlib import Path
import random
import shutil
import warnings
import re
import pandas as pd
import xml.etree.ElementTree as ET
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
        os.mkdir(path)
        return path

def validate_file(f):
    if not os.path.exists(f):
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

def split(names, div):
    """
    Divide una lista con las ponderaciones en div.
    """
    tr, ts, vl = div
    random.shuffle(names)
    train = names[0:m.floor(len(names)*tr)]
    val = names[m.floor(len(names)*tr):m.floor(len(names)*(tr+ts))]
    test = names[m.floor(len(names)*(1-vl)):len(names)]
    return (train, val, test)

def writeImageSets(setsnames, datasets):
    for i in range(len(datasets)):
        wr(setsnames[i]+".txt", datasets[i])

def xml_to_csv(ds_xmls):
    """
    Recibe una lista de anotaciones en formato .xml y las 
    comprime en un fichero .csv. La lista debe tener la ruta 
    completa a cada fichero .xml.
    """
    xml_list = []
    for xml_file in ds_xmls:
        tree = ET.parse(xml_file)
        root = tree.getroot()
        for member in root.findall('object'):
            bbx = member.find('bndbox')
            xmin = float(bbx.find('xmin').text)
            ymin = float(bbx.find('ymin').text)
            xmax = float(bbx.find('xmax').text)
            ymax = float(bbx.find('ymax').text)
            label = member.find('name').text
            value = (root.find('filename').text,
                     xmin,
                     ymin,
                     xmax,
                     ymax
                     )
            xml_list.append(value)
    column_name = ['image', 'xmin', 'ymin', 'xmax', 'ymax']
    xml_df = pd.DataFrame(xml_list, columns=column_name)
    return xml_df

def xml_to_csv(ds_xmls):
    xml_list = []
    for xml_file in ds_xmls:
        tree = ET.parse(xml_file)
        root = tree.getroot()
        for member in root.findall('object'):
            bbx = member.find('bndbox')
            xmin = float(bbx.find('xmin').text)
            ymin = float(bbx.find('ymin').text)
            xmax = float(bbx.find('xmax').text)
            ymax = float(bbx.find('ymax').text)
            label = member.find('name').text
            value = (root.find('filename').text,xmin,ymin,xmax,ymax)
            xml_list.append(value)
    column_name = ['image', 'xmin', 'ymin', 'xmax', 'ymax']
    xml_df = pd.DataFrame(xml_list, columns=column_name)
    return xml_df

def create_ds(path,name,div):
    """
    Dado un directorio(dataset original), crea otro con nombre "name" y lo
    divide según "div".
    """
    imgsrc,annsrc = path+"JPEGImages/",path+"Annotations/"
    if not os.path.exists(path+"ImageSets"):
        os.mkdir(path+"ImageSets/")
    setsdst = path+"ImageSets/"+name+"/"
    anndst = setsdst+"/annotations/"
    os.mkdir(setsdst)
    os.mkdir(anndst)
    names = [x.split(".")[0] for x in os.listdir(imgsrc)]
    datasets = split(names,div)
    setnames = ["train","val","test"]
    writeImageSets([setsdst+x for x in setnames], datasets)
    for i in range(len(datasets)):
        xml_files = [annsrc+x+".xml" for x in datasets[i]]
        xml_df = xml_to_csv(xml_files)
        xml_df.to_csv(f'{anndst}labels{setnames[i]}.csv')
    

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
    # imgdst = dest+"images/"
    annsrc = src+"Annotations/"
    anndst = dest+"annotations/"
    setsdst = dest+"imagesets/"
    if not os.path.exists(anndst):
        os.mkdir(anndst)
    if not os.path.exists(setsdst):
        os.mkdir(setsdst)

    if args.div=="0":
        div = [0.5,0.2,0.3]
    else:
        div = [(float(x)/100) for x in re.findall("..",args.div)]
    # Obtenemos sólo el nombre de cada imagen sin el .jpg
    names = [x.split(".")[0] for x in os.listdir(imgsrc)]
    # train, val, test, trainval = split(names,div)
    datasets = split(names,div)
    setnames = ["train","val","test"]
    writeImageSets([setsdst+x for x in setnames], datasets)
    for i in range(len(datasets)):
        xml_files = [annsrc+x+".xml" for x in datasets[i]]
        xml_df = xml_to_csv(xml_files)
        xml_df.to_csv(f'{anndst}labels{setnames[i]}.csv')


if __name__=='__main__':
    main()