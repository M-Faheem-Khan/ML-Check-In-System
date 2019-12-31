from os import listdir
from os.path import isdir
from numpy import asarray
from numpy import savez_compressed
from extractFace import Extract


def load_faces(directory):
    faces = list()
    # enumerate files
    for filename in listdir(directory):
        # path
        path = directory + filename
        # get face
        face = Extract(path).arrayOfFaces[0]
        # store
        faces.append(face)
    return faces


def load_dataset(directory):
    X, y = list(), list()
    # enumerate folders, on per class
    for subdir in listdir(directory):
        # path
        path = directory + subdir + '/'
        # skip any files that might be in the dir
        if not isdir(path):
            continue
        # load all faces in the subdirectory
        faces = load_faces(path)
        # create labels
        labels = [subdir for _ in range(len(faces))]
        # summarize progress
        print('>loaded %d examples for class: %s' % (len(faces), subdir))
        # store
        X.extend(faces)
        y.extend(labels)
    return asarray(X), asarray(y)


class CreateNPZFromDirectory:
    def __init__(self, directory):
        # load train dataset
        trainX, trainy = load_dataset(directory)
        savez_compressed('5-celebrity-faces-dataset.npz', trainX, trainy)

    # load a dataset that contains one subdir for each class that in turn contains images

