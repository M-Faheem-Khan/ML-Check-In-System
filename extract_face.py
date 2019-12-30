from matplotlib import pyplot as plt
from mtcnn.mtcnn import MTCNN
from numpy import asarray
from PIL import Image


class ExtractFace:
    def __init__(self, image_path, required_size=(160, 160)):
        # load image and detect faces
        image = plt.imread(image_path)
        detector = MTCNN()
        faces = detector.detect_faces(image)

        face_images = []

        for face in faces:
            # extract the bounding box from the requested face
            x1, y1, width, height = face['box']
            x2, y2 = x1 + width, y1 + height

            # extract the face
            face_boundary = image[y1:y2, x1:x2]

            # resize pixels to the model size
            face_image = Image.fromarray(face_boundary)
            face_image = face_image.resize(required_size)
            face_array = asarray(face_image)
            face_images.append(face_array)
        self.face_images = face_images
