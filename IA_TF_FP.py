#En esta parte se exportaron las librerias que se iban a usar para este proyecto, y asi se pudiera ejecutar en este proyecto
import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os
#En esta parte se declararon las variables en las cuales se iban a usar para este proyecto, en las categorias se colocaron los diferentes tipos objetos las cuales se iban a detectar que eran
#tambien se declararon las labels para saber la posiciones, como los indices, y las imagenes.
categorias= []
labels= []
imagenes= []
#en esta parte del codigo se uso para llegar a la libreria donde se encontraban las demas imagenes de los teclados y mouse, en este caso solo se colocaron 5 imagenes de cada uno, esto es para que vaya
#detectando y comparando cada objeto
categorias = os.listdir('C:\\Users\\javie\\OneDrive\\Escritorio\\Proyecto\\dataset\\')

#En esta parte se encuentra el directorio para el prosesamiento de las imagenes, es lo que se hace para que se pueda convertir en unas imagenes en unas dimensiones especificas, el for es para entrar 
#a las categorias,etrar a las carpetas, y entrar a cada una de las imagenes que se colocaron, y se le hcae un resize para que asi las imagenes obtengan un tamaño igual, luego se convierte en array, luego 
#despues se agrega a otro array normal, luego se agrega la x que es nuestro identificador, dependiendo de lo que sea
x=0
for directorio in categorias:
    for imagen in os.listdir('C:\\Users\\javie\\OneDrive\\Escritorio\\Proyecto\\dataset\\'+directorio):
        img= Image.open('C:\\Users\\javie\\OneDrive\\Escritorio\\Proyecto\\dataset\\'+directorio+'\\'+imagen).resize((200,200))
        img= np.asarray(img)
        imagenes.append(img)
        labels.append(x)
    x+=1

#esto  es para convertir el arreglo de imagenes para un arreglo numpy
imagenes =np.asanyarray(imagenes)
#esto es para que nos diga las dimensiones, y el canal de rgb, pero en este caso no se usa
imagenes.shape
#aqui es para eliminar el canal de rgb ya que no se utilizara
imagenes=imagenes[:,:,:,0]
#aqui es para comprobar si es que se eliminaron correctamente
imagenes.shape


#En esta parte del codigo se crea el modelo, donde se crea las capas,de la creacion de la red neuronal, del diseño, y las conexiones , en palabras cortascrear propia red
model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(200, 200)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(10, activation='softmax')
])
#Aqui como lo dice es para compilar modelo
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
#Aqui los labels son trabajados como un arreglo basico, se necesita pasar a numpy,aqui la difernecia es que los arreglos np no se usan comas.
labels= np.asarray(labels)

#Aqui se esta pasando los parametros del entrenamiento que se tendra, de la misma manera se puede apreciar que se utiliza epochs, esto es para la canrtidad de veecs de entrenamiento que se repetira
#es decir que agarrara todas las imagenes,todos los labes, tomara patrones y aprendera.
model.fit(imagenes,labels,epochs=20)
#Se igualar el modelo, tambien se procesa una imagen las cuales se agregaron en otra carpeta, aqui se agregaron 2 imagenes, una de teclado y otra de mouse,esto es para comparar
#se busca la ruta, se hace un resize, tambien se elimina una dimension, y se enseña la imagen que esta en la carpeta con el im2.show
im=0
im=Image.open('C:\\Users\\javie\\OneDrive\\Escritorio\\Proyecto\\test\\test1.jpg').resize((200,200))
im2=Image.open('C:\\Users\\javie\\OneDrive\\Escritorio\\Proyecto\\test\\test1.jpg')
im=np.asarray(im)
im=im[:,:,0]
im=np.array([im])
im.shape
test=im
im2.show()
#Se hace las predicciones de las cuales obtuvo el programa
predicciones = model.predict(test)
print(predicciones)

#Saca como el numero de predicciones que se dio, y asi finaliza el programa, al final dice que objeto es si es un mouse o un teclado
print (categorias[np.argmax(predicciones[0])])
