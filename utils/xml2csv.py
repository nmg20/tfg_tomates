import os
import glob
import argparse
import pandas as pd
import xml.etree.ElementTree as ET

def dir_path(path):
    if os.path.isdir(path):
        return path
    else:
        raise argparse.ArgumentTypeError(f"readable_dir:{path} is not a valid path")

def validate_file(f):
    if not os.path.exists(f):
        raise argparse.ArgumentTypeError("{0} does not exist".format(f))
    return f

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
    # column_name = ['filename', 'width', 'height',
    #                'class', 'xmin', 'ymin', 'xmax', 'ymax']
    # column_name = ['image', 'class', 'xmin', 'ymin', 'xmax', 'ymax']
    column_name = ['image', 'xmin', 'ymin', 'xmax', 'ymax']
    xml_df = pd.DataFrame(xml_list, columns=column_name)
    return xml_df

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('source_dir',help="carpeta con la imágenes a procesar.", 
        type = dir_path)
    # parser.add_argument('target_dir',help="carpeta en la que volcar el procesado", 
    #     type = dir_path)
    args = parser.parse_args()
    src = args.source_dir
    # dest = args.target_dir
    datasets = ['train', 'test', 'val']
    for ds in datasets:
        # src = './tests/'
        # dest = './data/tomato2/'
        annotations_path = os.path.join(src, 'annotations',ds)
        xml_df = xml_to_csv(annotations_path)
        xml_df.to_csv(f'{src}/annotations/labels{ds}.csv')
        print(f'Successfully converted xml to csv.[{ds}]')

if __name__=='__main__':
    main()