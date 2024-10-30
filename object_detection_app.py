import tkinter as tk
from tkinter import filedialog, messagebox
import torch
import cv2
from PIL import Image, ImageTk
import numpy as np
import pathlib
import threading

class ObjectDetectionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Object Detection")
        self.root.geometry("800x600")

        self.model = None
        self.cap = None
        self.is_camera_active = False

        # Botón para cargar el modelo
        self.load_model_button = tk.Button(root, text="Cargar Modelo", command=self.load_model)
        self.load_model_button.pack(pady=10)

        # Botón para usar la cámara
        self.camera_button = tk.Button(root, text="Usar Cámara", command=self.use_camera, state=tk.DISABLED)
        self.camera_button.pack(pady=10)

        # Botón para subir una imagen
        self.upload_button = tk.Button(root, text="Subir Imagen", command=self.upload_image, state=tk.DISABLED)
        self.upload_button.pack(pady=10)

        # Área para mostrar la imagen de predicción
        self.prediction_image_label = tk.Label(root)
        self.prediction_image_label.pack(pady=5)

    def load_model(self):
        weights_path = "best.pt"
        self.model = torch.hub.load('ultralytics/yolov5', 'custom', path=weights_path)
        messagebox.showinfo("Información", "Modelo cargado correctamente.")
        self.camera_button.config(state=tk.NORMAL)
        self.upload_button.config(state=tk.NORMAL)

    def use_camera(self):
        if self.is_camera_active:
            return
        
        self.is_camera_active = True
        self.cap = cv2.VideoCapture(0)
        
        def camera_loop():
            while self.is_camera_active:
                ret, frame = self.cap.read()
                if not ret:
                    break
                
                # Convertir BGR a RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                # Ejecutar inferencia
                results = self.model(frame_rgb)

                # Renderizar resultados
                results.render()

                # Convertir la imagen a PIL
                img = Image.fromarray(results.ims[0])

                # Redimensionar la imagen
                img = img.resize((800, 600), Image.LANCZOS)  # Ajustar a 800x600
                img_tk = ImageTk.PhotoImage(img)

                # Mantener referencia a la imagen
                self.prediction_image_label.img_tk = img_tk
                self.prediction_image_label.config(image=img_tk)

                self.prediction_image_label.update()

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            self.cap.release()

        # Iniciar el bucle de la cámara en un hilo separado
        threading.Thread(target=camera_loop, daemon=True).start()

    def upload_image(self):
        if self.is_camera_active:
            self.is_camera_active = False
            if self.cap:
                self.cap.release()  # Cerrar la cámara si está abierta

        file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.jpg;*.jpeg;*.png")])
        if file_path:

            img = cv2.imread(file_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            if img is None:
                messagebox.showerror("Error", "No se pudo cargar la imagen.")
                return

            results = self.model(img)

            results.render()

            # Convertir la imagen de predicción a PIL y mostrar
            prediction_img = Image.fromarray(results.ims[0])
            prediction_img = prediction_img.resize((800, 600), Image.LANCZOS)  # Ajustar a 800x600
            prediction_img_tk = ImageTk.PhotoImage(prediction_img)

            # Mantener referencia a la imagen de predicción
            self.prediction_image_label.img_tk = prediction_img_tk
            self.prediction_image_label.config(image=prediction_img_tk)

if __name__ == "__main__":
    pathlib.PosixPath = pathlib.WindowsPath
    root = tk.Tk()
    app = ObjectDetectionApp(root)
    root.mainloop()
