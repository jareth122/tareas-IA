# Reporte del Código para Detección de Rostros en Imagen de Wally

## Introducción
El objetivo del código es detectar rostros en una imagen de Wally usando OpenCV y un clasificador en cascada preentrenado. A continuación, se detalla cada paso del código, las librerías utilizadas y el propósito de cada sección.

## Librerías Utilizadas
- **numpy**: Se utiliza para operaciones matemáticas y manejo de arrays.
- **cv2 (OpenCV)**: Librería principal para el procesamiento de imágenes y visión por computadora.

    ```python
    import numpy as np
    import cv2 as cv
    ```

## Cargar el Clasificador y la Imagen
Se carga un clasificador en cascada preentrenado para la detección de rostros y la imagen de Wally desde el sistema de archivos.

    ```python
    # Cargar el clasificador entrenado
    rostro = cv.CascadeClassifier('C:\\Users\\agust\\Documents\\9SEMESTRE\\IA\\wally\\classifier\\cascade.xml')

    # Cargar la imagen de Wally
    img = cv.imread('C:\\Users\\agust\\Documents\\9SEMESTRE\\IA\\wally\\classifier\\w4.jpg')
    ```

## Conversión a Escala de Grises
La imagen cargada se convierte a escala de grises para facilitar la detección de rostros, ya que los clasificadores en cascada suelen trabajar mejor con imágenes en este formato.

    ```python
    # Convertir la imagen a escala de grises
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    ```

## Detección de Rostros
Se detectan los rostros en la imagen en escala de grises utilizando el clasificador cargado. El método `detectMultiScale` retorna una lista de coordenadas (x, y, w, h) para cada rostro detectado.

    ```python
    # Detectar rostros en la imagen
    rostros = rostro.detectMultiScale(gray, 1.1, 40)
    ```

## Dibujar Rectángulos alrededor de los Rostros Detectados
Para cada rostro detectado, se dibuja un rectángulo verde alrededor de la región detectada en la imagen original.

    ```python
    # Procesar cada rostro detectado
    for (x, y, w, h) in rostros:
        # Dibujar un rectángulo verde alrededor del rostro detectado
        img = cv.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
    ```

## Mostrar la Imagen
Finalmente, se muestra la imagen con los rectángulos dibujados alrededor de los rostros detectados. La ventana se mantiene abierta hasta que se presiona una tecla.

    ```python
    # Mostrar la imagen con los rostros detectados y los rectángulos
    cv.imshow('Wally', img)
    cv.waitKey(0)
    cv.destroyAllWindows()
    ```

## Consideraciones Finales
- **Ruta de los Archivos**: Asegúrese de que las rutas del clasificador y la imagen sean correctas y accesibles desde el entorno donde se ejecuta el código.
- **Parámetros del Método `detectMultiScale`**: Los parámetros (1.1, 40) pueden ajustarse para optimizar la detección de rostros dependiendo de la imagen y el clasificador utilizado.
- **Entorno de Ejecución**: Este código está diseñado para ejecutarse en un entorno que soporte interfaces gráficas para mostrar la imagen (por ejemplo, una máquina local con un entorno de escritorio).
