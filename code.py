import tensorflow as tf
import tensorflow_datasets as tfds

# Descargar el set de datos de perros y gatos
datos,metadatos = tfds.load("cats_vs_dogs",as_supervised=True,with_info=True)

# analizando los metadatos
metadatos

# Ver algunas de las imagenes
tfds.as_dataframe(datos["train"].take(5),metadatos)

    # Otra manera
tfds.show_examples(datos["train"],metadatos)

# Mostrando con matplotlib
import matplotlib.pyplot as plt
import cv2 # Redimencionar imagenes

# Tamaño para redimensionar la imagen
size_img = 100

plt.figure(figsize=(20,20))
for i, (imagen,etiqueta) in enumerate(datos["train"].take(25)):
    # Resizing image to 100px X 100px
    imagen = cv2.resize(imagen.numpy(),(size_img,size_img))
    # Changing color to black and white every image
    imagen = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(imagen,cmap="gray")

train_data = []
# transformando las imagenes de la data de entremiento ->
#   resizing and change color to black and white
for i,(image, label) in enumerate(datos["train"]):
    # Resizing image to 100px X 100px
    image = cv2.resize(image.numpy(),(size_img,size_img))
    # Changing color to black and white every image
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # resdimensinando la imagen y especificando que que imagen está en blanco
    # y negro -> es decir tamaño de 100,100,1
    image = image.reshape(size_img,size_img,1)
    train_data.append([image,label])

# observando el primer dato
train_data[0]
# Tamaño del entrenamiento
len(train_data)

# Crenado variable independiente y dependiente
x = [] # Imagen dentrasa (pixeles)
y = [] # etiquetas (perro o gato)

for image, label in train_data:
    x.append(image)
    y.append(label)

# Normalizando las imagenes -> es decir pasar de los colores de rgb, es decir
# valores de 0-255 a valore sde 0-1
import numpy as np
x = np.array(x).astype(float)/255
# Pasando de tensores a valores de 0,1
y = np.array(y)


# MODELO
# Estructura del las redes neuronales
    # Modelo 1 -> red neuronal Denso
modeloDenso = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(100,100,1)),
    tf.keras.layers.Dense(150,activation="relu"),
    tf.keras.layers.Dense(150,activation="relu"),
    tf.keras.layers.Dense(1,activation="sigmoid")
])

    # Modleo 2 -> red neuronal convolucional
modeloCNN = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32,(3,3),activation="relu",input_shape=(100,100,1)),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(64,(3,3),activation="relu"),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(128,(3,3),activation="relu"),
    tf.keras.layers.MaxPooling2D(2,2),

    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(100,activation="relu"),
    tf.keras.layers.Dense(1, activation = "sigmoid")
])
    # Modleo 3
modeloCNN2 = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32,(3,3),activation="relu",input_shape=(100,100,1)),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(64,(3,3),activation="relu"),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(128,(3,3),activation="relu"),
    tf.keras.layers.MaxPooling2D(2,2),

    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(100,activation="relu"),
    tf.keras.layers.Dense(1, activation = "sigmoid")
])

# Compilando los modelos
modeloDenso.compile(optimizer="adam",
                    loss="binary_crossentropy",
                    metrics=["accuracy"])

modeloCNN.compile(optimizer="adam",
                    loss="binary_crossentropy",
                    metrics=["accuracy"])

modeloCNN2.compile(optimizer="adam",
                    loss="binary_crossentropy",
                    metrics=["accuracy"])

# Libreria para ver como se comprotan nuestros modelos y poder comparar
# con otros modelos
from tensorflow.keras.callbacks import TensorBoard

# Entrenando el primer modelo
    # Variable que almacenará el modelo
tensorboardDenso = TensorBoard(log_dir="logs/denso")
modeloDenso.fit(x,y, batch_size=32, # definición del lote de entrenamiento
                validation_split=0.15, # permite separar la data en entrenamiento
                                       # y test
                epochs=100,
                callbacks=[tensorboardDenso]) # Permite guardar los resultados de cada una de las epocas en la variable
                                              # antes creada (una capeta especifica que es "logs/deso")

# Entrenando el segundo modelo
    # Variable que almacenará el modelo
tensorboardCNN = TensorBoard(log_dir="logs/cnn")
modeloCNN.fit(x,y, batch_size=32, # definición del lote de entrenamiento
                validation_split=0.15, # permite separar la data en entrenamiento
                                       # y test
                epochs=100,
                callbacks=[tensorboardCNN]) # Permite guardar los resultados de cada una de las epocas en la variable
                                            # antes creada (una capeta especifica que es "logs/deso")

# Entrenando el tercer modelo
    # Variable que almacenará el modelo
tensorboardCNN2 = TensorBoard(log_dir="logs/cnn2")
modeloCNN2.fit(x,y, batch_size=32, # definición del lote de entrenamiento
                validation_split=0.15, # permite separar la data en entrenamiento
                                       # y test
                epochs=100,
                callbacks=[tensorboardCNN2]) # Permite guardar los resultados de cada una de las epocas en la variable
                                            # antes creada (una capeta especifica que es "logs/deso")


# Lanzando tensorboard
%load_ext tensorboard
    # Incicando la carpeta donde se encuentran nuestros modelos
%tensorboard --logdir logs