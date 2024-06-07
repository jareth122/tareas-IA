
# Reporte: Detección y Reconocimiento Facial

El código proporcionado se divide en tres partes principales que abordan la detección de rostros, el entrenamiento de un modelo de reconocimiento facial, y la utilización de dicho modelo para predecir y reconocer rostros en tiempo real. A continuación, presento un reporte detallado de cada sección del código:

## 1. Detección de rostros y guardado de imágenes

```python
import cv2 as cv
import numpy as np

face = cv.CascadeClassifier('haarcascade_frontalface_alt.xml')
cap = cv.VideoCapture(0)
i = 0
while True:
    ret, frame = cap.read()
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    faces = face.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in faces:
        frame2 = frame[y-10:y + h + 10, x-10:x + w + 10]
        frame2 = cv.resize(frame2, (100, 100), interpolation=cv.INTER_AREA)
        cv.imwrite("Emociones/Triste/jareth" + str(i) + ".png", frame2)
    cv.imshow('faces', frame)
    i = i + 1
    k = cv.waitKey(1)
    if k == 27:
        break
cap.release()
cv.destroyAllWindows()
```

### Funcionalidad:
- Captura video desde la cámara web.
- Convierte los fotogramas a escala de grises.
- Utiliza el clasificador Haar para detectar rostros.
- Guarda los rostros detectados como imágenes de 100x100 píxeles en una carpeta específica.
- Muestra los fotogramas con los rostros detectados.

## 2. Entrenamiento del modelo de reconocimiento facial

```python
import cv2 as cv
import numpy as np
import os

dataSet = 'emociones'
faces = os.listdir(dataSet)
print(faces)

labels = []
facesData = []
label = 0
for face in faces:
    facePath = dataSet + '/' + face
    for faceName in os.listdir(facePath):
        labels.append(label)
        facesData.append(cv.imread(facePath + '/' + faceName, 0))
    label = label + 1
print(np.count_nonzero(np.array(labels) == 0))

faceRecognizer = cv.face.LBPHFaceRecognizer_create()
faceRecognizer.train(facesData, np.array(labels))
faceRecognizer.write('Emociones1.xml')
print("Archivo xml creado con éxito")
```

### Funcionalidad:
- Lee las imágenes de rostros guardadas en el directorio emociones.
- Asigna etiquetas a cada conjunto de imágenes de rostros.
- Entrena un modelo de reconocimiento facial usando el algoritmo LBPH (Local Binary Patterns Histograms).
- Guarda el modelo entrenado en un archivo XML (Emociones1.xml).

## 3. Reconocimiento de rostros en tiempo real

```python
import cv2 as cv
import numpy as np
import os

dataSet = 'Emociones'
faces = os.listdir(dataSet)
faceRecognizer = cv.face.LBPHFaceRecognizer_create()
faceRecognizer.read('Emociones.xml')

cap = cv.VideoCapture(0)
rostro = cv.CascadeClassifier('haarcascade_frontalface_alt.xml')

while True:
    ret, frame = cap.read()
    if not ret: break
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    cpGray = gray.copy()
    rostros = rostro.detectMultiScale(gray, 1.3, 3)
    for(x, y, w, h) in rostros:
        frame2 = cpGray[y:y+h, x:x+w]
        frame2 = cv.resize(frame2, (100, 100), interpolation=cv.INTER_CUBIC)
        result = faceRecognizer.predict(frame2)
        if result[1] > 70:
            cv.putText(frame, '{}'.format(faces[result[0]]), (x, y - 25), 2, 1.1, (0, 255, 0), 1, cv.LINE_AA)
            cv.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        else:
            cv.putText(frame, 'Desconocido', (x, y - 20), 2, 0.8, (0, 0, 255), 1, cv.LINE_AA)
            cv.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
    cv.imshow('frame', frame)
    k = cv.waitKey(1)
    if k == 27:
        break

cap.release()
cv.destroyAllWindows()
```

### Funcionalidad:
- Captura video desde la cámara web.
- Convierte los fotogramas a escala de grises.
- Detecta rostros en los fotogramas utilizando el clasificador Haar.
- Redimensiona los rostros detectados y los compara con el modelo entrenado para predecir la identidad.
- Muestra los nombres de las personas reconocidas o "Desconocido" si la confianza de la predicción es baja.
- Muestra los fotogramas con los rostros detectados y sus etiquetas.

## Conclusión
El código proporcionado abarca tres aspectos clave del procesamiento de imágenes y el reconocimiento facial: captura de imágenes, entrenamiento de un modelo de reconocimiento facial, y reconocimiento en tiempo real.

### Captura de imágenes:
La primera sección se encarga de capturar imágenes de rostros desde la cámara web y guardarlas en un directorio específico. Esto es fundamental para crear un conjunto de datos que pueda ser utilizado para entrenar un modelo de reconocimiento facial.

### Entrenamiento del modelo:
La segunda sección utiliza las imágenes capturadas para entrenar un modelo de reconocimiento facial basado en el algoritmo LBPH. Este modelo se guarda en un archivo XML para su uso posterior.

### Reconocimiento en tiempo real:
La tercera sección emplea el modelo entrenado para reconocer rostros en tiempo real desde la cámara web, etiquetando los rostros detectados y mostrando los resultados en pantalla.
