import os

imagesets_dir = "../../datasets/Tomato_1280x720/ImageSets/"
models_dir = "./modelos/"

def read_imageset(path, ds="test"):
    file = f"{imagesets_dir}{path}/{ds}.txt"
    contents = open(file).read().split("\n")[:-1]
    return contents

def read_models_dir(path="./modelos/"):
    return [x[:-3] for x in os.listdir(path)]

def intersect(names):
    sets = []
    for conj in names:
        sets.append(set(conj))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('source_dir',help="carpeta con la imágenes a procesar.", 
        type = dir_path)
    parser.add_argument('target_dir',help="carpeta en la que volcar el procesado.", 
        type = dir_path)
    parser.add_argument('-d','--div',type=str)
    # parser.add_argument('test', help="flag que indica si se establece un conjunto de test.")
    args = parser.parse_args()