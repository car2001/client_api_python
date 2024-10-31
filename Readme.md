# Apio Detection - Detección de Plagas de la Mosca Minadora

## Descripción

**Apio Detection** es una aplicación de escritorio desarrollada en Python que permite a los usuarios cargar imágenes de apios o usar la cámara para detectar la presencia de plagas, específicamente la mosca minadora (Liriomyza spp.). Esta aplicación utiliza un modelo de detección de objetos basado en YOLOv5 para identificar y clasificar las plagas en las imágenes.

## Características

- Carga de imágenes desde el sistema local.
- Detección en tiempo real utilizando la cámara.
- Visualización de resultados con cuadros delimitadores que indican la ubicación de las plagas detectadas.
- Interfaz gráfica de usuario (GUI) simple y accesible.

## Tecnologías Utilizadas

- **Python** - Lenguaje de programación utilizado para el desarrollo de la aplicación.
- [YOLOv5](https://github.com/ultralytics/yolov5) - Modelo de detección de objetos de alta precisión.
- [Tkinter](https://docs.python.org/3/library/tkinter.html) - Biblioteca para crear interfaces gráficas de usuario en Python.
- [OpenCV](https://opencv.org/) - Biblioteca para procesamiento de imágenes y visión por computadora.
- [Pillow](https://pillow.readthedocs.io/en/stable/) - Biblioteca para abrir, manipular y guardar imágenes.

## Requisitos

Antes de ejecutar la aplicación, asegúrate de tener instaladas las siguientes bibliotecas:

- Python 3.x
- PIP (gestor de paquetes de Python)

Instala las dependencias necesarias con el siguiente comando:

```bash
pip install torch opencv-python Pillow numpy
