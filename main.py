from extractFace import Extract  # Takes an image path and returns an array of faces
from matplotlib import pyplot
from os import listdir

i = 1
for filename in listdir("Eric_Marcantonio"):
    path = "Eric_Marcantonio/" + filename

    face = Extract(path).arrayOfFaces[0]
    print(i, face.shape)
    pyplot.subplot(2, 7, i)
    pyplot.axis('off')
    pyplot.imshow(face)
    i += 1
pyplot.show()
