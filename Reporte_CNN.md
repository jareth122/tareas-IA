
# Reporte sobre la Implementación de una Red Neuronal Convolucional

## Introducción
Este documento describe el desarrollo e implementación de una red neuronal convolucional (CNN) para la clasificación de imágenes. Se detallan los pasos desde la preparación de los datos hasta el entrenamiento y evaluación del modelo. La CNN es una arquitectura popular para el reconocimiento de patrones y clasificación en imágenes debido a su capacidad para aprender características espaciales y jerárquicas.

## Desarrollo

### Importación de Librerías
El proyecto comienza con la importación de las librerías necesarias para el manejo de datos, visualización y construcción del modelo:

```python
import numpy as np
import os
import re
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import keras
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from keras.models import Sequential
from tensorflow.keras.layers import (BatchNormalization, SeparableConv2D, MaxPooling2D, Activation, Flatten, Dropout, Dense, Conv2D)
from keras.layers import LeakyReLU
```

### Carga del Conjunto de Imágenes
Las imágenes se cargan desde un directorio especificado y se almacenan en listas para su posterior procesamiento. Se utilizan expresiones regulares para identificar archivos de imágenes:

```python
dirname = os.path.join(os.getcwd(), 'C:/Users/agust/Documents/9SEMESTRE/IA/CNN/Dataset')
imgpath = dirname + os.sep

images = []
directories = []
dircount = []
prevRoot = ''
cant = 0

for root, dirnames, filenames in os.walk(imgpath):
    for filename in filenames:
        if re.search("\.(jpg|jpeg|png|bmp|tiff)$", filename):
            cant += 1
            filepath = os.path.join(root, filename)
            image = plt.imread(filepath)
            if len(image.shape) == 3:
                images.append(image)
            if prevRoot != root:
                directories.append(root)
                dircount.append(cant)
                cant = 0
dircount.append(cant)
```

### Creación de Etiquetas
Se crean etiquetas para las imágenes basadas en sus directorios:

```python
labels = []
indice = 0
for cantidad in dircount:
    for i in range(cantidad):
        labels.append(indice)
    indice += 1

y = np.array(labels)
X = np.array(images, dtype=np.uint8)
```

### División de Datos
Los datos se dividen en conjuntos de entrenamiento y prueba:

```python
train_X, test_X, train_Y, test_Y = train_test_split(X, y, test_size=0.2)
```

### Preprocesamiento
Se realiza el preprocesamiento de los datos, incluyendo la normalización y el one-hot encoding de las etiquetas:

```python
train_X = train_X.astype('float32') / 255.
test_X = test_X.astype('float32') / 255.
train_Y_one_hot = to_categorical(train_Y)
test_Y_one_hot = to_categorical(test_Y)
train_X, valid_X, train_label, valid_label = train_test_split(train_X, train_Y_one_hot, test_size=0.2, random_state=13)
```

### Construcción del Modelo CNN
Se define y compila el modelo de CNN utilizando Keras:

```python
disaster_model = Sequential()
disaster_model.add(Conv2D(32, kernel_size=(3, 3), activation='linear', padding='same', input_shape=(28, 28, 3)))
disaster_model.add(LeakyReLU(alpha=0.1))
disaster_model.add(MaxPooling2D((2, 2), padding='same'))
disaster_model.add(Dropout(0.5))

disaster_model.add(Flatten())
disaster_model.add(Dense(32, activation='linear'))
disaster_model.add(LeakyReLU(alpha=0.1))
disaster_model.add(Dropout(0.5))
disaster_model.add(Dense(nClasses, activation='softmax'))

disaster_model.compile(loss=keras.losses.categorical_crossentropy, optimizer=tf.keras.optimizers.SGD(learning_rate=1e-3), metrics=['accuracy'])
```

### Entrenamiento del Modelo
El modelo se entrena con los datos de entrenamiento y se valida con los datos de validación:

```python
sport_train = disaster_model.fit(train_X, train_label, batch_size=64, epochs=20, verbose=1, validation_data=(valid_X, valid_label))
```

### Evaluación del Modelo
Se evalúa el rendimiento del modelo con los datos de prueba:

```python
test_eval = disaster_model.evaluate(test_X, test_Y_one_hot, verbose=1)
print('Test loss:', test_eval[0])
print('Test accuracy:', test_eval[1])
```

### Visualización de Resultados
Se visualizan las métricas de precisión y pérdida durante el entrenamiento:

```python
accuracy = sport_train.history['accuracy']
val_accuracy = sport_train.history['val_accuracy']
loss = sport_train.history['loss']
val_loss = sport_train.history['val_loss']
epochs = range(len(accuracy))

plt.plot(epochs, accuracy, 'bo', label='Training accuracy')
plt.plot(epochs, val_accuracy, 'b', label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.legend()

plt.figure()
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.show()
```

### Análisis de Errores
Se identifican y visualizan los errores de clasificación:

```python
predicted_classes = disaster_model.predict(test_X)
predicted_classes = np.argmax(predicted_classes, axis=1)

correct = np.where(predicted_classes == test_Y)[0]
incorrect = np.where(predicted_classes != test_Y)[0]

for i, correct in enumerate(correct[0:9]):
    plt.subplot(3, 3, i+1)
    plt.imshow(test_X[correct].reshape(28, 28, 3), cmap='gray')
    plt.title("{} {}".format(desastres[predicted_classes[correct]], desastres[test_Y[correct]]))
plt.tight_layout()

for i, incorrect in enumerate(incorrect[0:9]):
    plt.subplot(3, 3, i+1)
    plt.imshow(test_X[incorrect].reshape(28, 28, 3), cmap='gray')
    plt.title("{} {}".format(desastres[predicted_classes[incorrect]], desastres[test_Y[incorrect]]))
plt.tight_layout()
```

### Generación del Reporte de Clasificación
Se genera un reporte detallado de las métricas de clasificación:

```python
target_names = ["Class {}".format(i) for i in range(nClasses)]
print(classification_report(test_Y, predicted_classes, target_names=target_names))
```

## Conclusión
En este proyecto, hemos desarrollado una red neuronal convolucional para la clasificación de imágenes, desde la carga y preprocesamiento de datos hasta la construcción, entrenamiento y evaluación del modelo. Los resultados muestran una precisión aceptable, lo que sugiere que el modelo es capaz de clasificar correctamente la mayoría de las imágenes. Se identificaron áreas de mejora en las imágenes mal clasificadas, lo cual es crucial para iteraciones futuras y mejoras del modelo. Este proyecto demuestra la efectividad de las CNN en tareas de visión por computadora y proporciona una base sólida para futuros trabajos en el campo.
