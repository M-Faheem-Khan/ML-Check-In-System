from mtcnn import MTCNN

from numpy import asarray
from PIL import Image

'''
Takes a image path, and returns an array of faces
'''


class Extract:

    def __init__(self, filename):
        # load image from file
        image = Image.open(filename)
        # convert to RGB, if needed
        image = image.convert('RGB')
        # convert to array
        pixels = asarray(image)
        # create the detector, using default weights
        detector = MTCNN()
        # detect faces in the image
        results = detector.detect_faces(pixels)
        # extract the bounding box from the first face
        faceArray = []
        for result in results:
            x1, y1, width, height = result['box']
            # bug fix
            x1, y1 = abs(x1), abs(y1)
            x2, y2 = x1 + width, y1 + height
            # extract the face
            face = pixels[y1:y2, x1:x2]
            # resize pixels to the model size
            image = Image.fromarray(face)
            image = image.resize((160, 160))
            face_array = asarray(image)
            faceArray.append(face_array)
        self.arrayOfFaces = faceArray
    # load the photo and extract the face
