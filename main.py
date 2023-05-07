import numpy as np
import cv2
import random
import os
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D


DIRECTORIO = "Datasets"
CATEGORIAS = ["fresas", "manzanas"]

datos = []

for categoria in CATEGORIAS:
    path = os.path.join(DIRECTORIO, categoria)
    class_num = CATEGORIAS.index(categoria)
    for img in os.listdir(path):
        img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)
        #print(img_array)
        #print(img_array.shape)
        #print("-----------")
        new_array = cv2.resize(img_array, (100, 100))
        #print(new_array)
        #print(new_array.shape)
        #print("-----------")
        print("::::::::::::::::::::::::: ", img)
        datos.append([new_array,class_num])

print(len(datos))

random.shuffle(datos)

X = []
y = []

for matriz, etiquetas in datos:
    X.append(matriz)
    y.append(etiquetas)

X = np.array(X).reshape(-1,100, 100, 1)
y = np.array(y)
print("----------")


X = X/255.0

model = Sequential()
model.add(Conv2D(64, (5,5), input_shape= X.shape[1:]))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(64, (5,5)))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Flatten())
model.add(Dense(64))

model.add(Dense(1))
model.add(Activation("sigmoid"))

model.compile(loss="binary_crossentropy", optimizer="adam", metrics = ['accuracy'])

model.fit(X,y, batch_size=32, epochs = 12, validation_split=0.2)


#-------------------------------------------------


def preparar(filepath):
    img_array = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
    new_array = cv2.resize(img_array, (100, 100))
    return new_array.reshape(-1, 100, 100, 1)
#preuba

PREDICCIONES =["Fresa", "Manzana"]
carpeta="Predecir"

path = os.path.join(carpeta)
for img in os.listdir(path):
    print(img)
    prediction = model.predict([preparar(os.path.join(path,img))])
    print(prediction)
    print(f"La imagen {img} es una", PREDICCIONES[int(prediction[0][0])])