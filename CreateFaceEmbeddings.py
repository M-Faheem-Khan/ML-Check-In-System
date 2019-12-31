# develop a classifier for the 5 Celebrity Faces Dataset
from numpy import load
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import Normalizer
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
import pickle # importing pickle
from extractFace import Extract
import os

class CreateFaceEmbedding(object):
    def __init__(self, NPZ_Path, Model_Path):
        # load dataset
        data = load(NPZ_Path)
        trainX, trainy = data['arr_0'], data['arr_1']

        # normalize input vectors
        in_encoder = Normalizer(norm='l2')
        trainX = in_encoder.transform(trainX)

        # label encode targets
        out_encoder = LabelEncoder()
        out_encoder.fit(trainy)
        trainy = out_encoder.transform(trainy)
        # GETTING FILE/FOLDER IN CWD
        dirs = os.listdir(os.getcwd())
        
        if (Model_Path in dirs):
            # LOADING MODEL
            with open(Model_Path, "w+") as model:
                pickle_model = pickle.load(model)

                # TRAIN MODEL
                print("MODEL LOADED - TRAINING...")
                model.fit(trainX, trainy)
                print("MODEL TRAINED - SAVING...")

                # SAVING MODEL
                pickle.dump(pickle_model, model)
        else:
            print("CreateFaceEmbedding Exception: Model not found")



        
        '''

        In a nutshell

        load a model using pickle
        https://stackabuse.com/scikit-learn-save-and-restore-models/

        retrain model with new photos, then on a seperate class, evalute the photos being passed
        '''
        
        
