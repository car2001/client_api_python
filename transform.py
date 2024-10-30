import torch
import pathlib
import cv2
from PIL import Image
from io import BytesIO
import base64

# Ajustar el Path para Windows
pathlib.PosixPath = pathlib.WindowsPath

# Cargar el modelo desde un archivo de pesos local
weights_path = "best.pt"  # Cambia esto a la ruta de tu modelo
model = torch.hub.load('ultralytics/yolov5', 'custom', path=weights_path)

# Leer la imagen
img = cv2.imread("./IMG_20241019_120508.jpg")
# Asegurarse de que la imagen esté en el formato correcto
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Ejecutar inferencia
results = model(img)

# Mostrar resultados
results.print()  # Imprime resultados en consola
results.show()   # Muestra la imagen con las detecciones

# Renderizar los resultados en la imagen
results.render()

# Convertir las imágenes con predicciones a Base64
base64_images = []
for img in results.ims:
    buffered = BytesIO()
    img_base64 = Image.fromarray(img)
    img_base64.save(buffered, format="JPEG")
    img_base64_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    base64_images.append(img_base64_str)

# Guardar Base64 completo en un archivo de texto
with open("base64_images.txt", "w") as f:
    for i, img_base64 in enumerate(base64_images):
        f.write(f"Base64 image {i + 1}:\n{img_base64}\n\n")  # Escribir cada imagen en el archivo

print("Las cadenas Base64 han sido guardadas en 'base64_images.txt'")