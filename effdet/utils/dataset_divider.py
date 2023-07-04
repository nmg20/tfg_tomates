import os
import argparse
import math as m
from pathlib import Path
import random
import shutil
import warnings
import re
import glob
import pandas as pd
import xml.etree.ElementTree as ET


"""
Idea: 1 argumento (la carpeta del ds original)
    - Dividir el conjunto de imágenes para 80/10/10, 70/15/15 y 60/20/20.
        -> expandible para más/otras distribuciones
    - Registrar en .csv las anotaciones
    - Dividir el conjunto de imágenes en trainval y test
    - Conjunto de test con las mismas imágenes para cada dataset
        -> 80/10/10 menos imágenes
        -> 60/20/20 con más
            -> se usaría el conjunto de test de 80/10/10 para comparar
             todos los datasets porque es el más pequeño.

"""

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

def split(names, div, test_aux=[]):
    names = list(set(names).difference(set(test_aux)))
    tr, ts, _ = div
    random.shuffle(names)
    if test_aux!=[]:
        random.shuffle(test_aux)
        names = test_aux + names
    test_lim = m.floor(len(names)*ts)
    train_lim = test_lim+m.floor(len(names)*tr)
    test = names[0:test_lim]
    train = names[test_lim:train_lim]
    val = names[train_lim:len(names)]
    return (train, val, test)

def split_hier(names, divs=[]):
    #Ordena la lista de distribuciones en base al % de test de mayor a menor
    divs.sort(key = lambda x: x[2], reverse=True)
    test = []
    for div in divs:
        train, test, val = split(names, div, test)

def xml_to_csv(path):
    xml_list = []
    for xml_file in glob.glob(path + '/*.xml'):
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

def writeImageSets(dest, train, test, val):
    wr(dest+"train.txt", train)
    wr(dest+"test.txt", test)
    wr(dest+"val.txt", val)
    wr(dest+"trainval.txt", train+val)

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
    args = parser.parse_args()
    src = args.source_dir
    dest = args.target_dir
        
    imgsrc = src+"JPEGImages/"
    imgdst = dest+"images/"
    annsrc = src+"Annotations/"
    anndst = dest+"annotations/"

    divs = [[0.8,0.1,0.1],[0.7,0.15,0.15],[0.6,0.2,0.2]]
    splitted_dss = split_aux(os.listdir(src+"JPEGImages/"),divs)
    for ds, div in zip(splitted_dss, divs):
        train, test, val = ds
        name = "d"+''.join([str(int(x*100)) for x in div])
        make_annotations(dest, name, annsrc, train, val, test)
        make_imagesets(dest, name, train, val, test)

if __name__=='__main__':
    main()