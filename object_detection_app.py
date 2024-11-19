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
        self.camera_index = tk.IntVar()

        # Botón para cargar el modelo
        self.load_model_button = tk.Button(root, text="Cargar Modelo", command=self.load_model)
        self.load_model_button.pack(pady=10)

        # Menú desplegable para seleccionar la cámara
        self.camera_selector = tk.OptionMenu(root, self.camera_index, *self.detect_cameras(), command=self.change_camera)
        self.camera_selector.pack(pady=10)
        self.camera_index.set(0)  # Seleccionar la primera cámara como predeterminada

        # Botón para usar la cámara seleccionada
        self.camera_button = tk.Button(root, text="Usar Cámara", command=self.use_camera, state=tk.DISABLED)
        self.camera_button.pack(pady=10)

        # Botón para subir una imagen
        self.upload_button = tk.Button(root, text="Subir Imagen", command=self.upload_image, state=tk.DISABLED)
        self.upload_button.pack(pady=10)

        # Área para mostrar la imagen de predicción
        self.prediction_image_label = tk.Label(root)
        self.prediction_image_label.pack(pady=5)

    def detect_cameras(self):
        available_cameras = []
        for index in range(5):  # Limitar a 5 cámaras para búsqueda
            cap = cv2.VideoCapture(index)
            if cap.isOpened():
                available_cameras.append(index)
                cap.release()
        if not available_cameras:
            messagebox.showerror("Error", "No se encontraron cámaras.")
        return available_cameras

    def load_model(self):
        weights_path = "best.pt"
        self.model = torch.hub.load('ultralytics/yolov5', 'custom', path=weights_path)
        messagebox.showinfo("Información", "Modelo cargado correctamente.")
        self.camera_button.config(state=tk.NORMAL)
        self.upload_button.config(state=tk.NORMAL)

    def change_camera(self, *args):
        # Cerrar la cámara activa si está abierta
        if self.is_camera_active:
            self.is_camera_active = False
            if self.cap:
                self.cap.release()
                self.cap = None
        # Reiniciar la cámara al cambiar de opción
        self.use_camera()

    def use_camera(self):
        if self.is_camera_active:
            return

        self.is_camera_active = True
        self.cap = cv2.VideoCapture(self.camera_index.get())  # Usar el índice seleccionado en el menú

        def camera_loop():
            while self.is_camera_active:
                ret, frame = self.cap.read()
                if not ret:
                    break
                
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = self.model(frame_rgb)
                results.render()

                img = Image.fromarray(results.ims[0])
                img = img.resize((800, 600), Image.LANCZOS)
                img_tk = ImageTk.PhotoImage(img)

                self.prediction_image_label.img_tk = img_tk
                self.prediction_image_label.config(image=img_tk)
                self.prediction_image_label.update()

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            self.cap.release()

        threading.Thread(target=camera_loop, daemon=True).start()

    def upload_image(self):
        if self.is_camera_active:
            self.is_camera_active = False
            if self.cap:
                self.cap.release()

        file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.jpg;*.jpeg;*.png")])
        if file_path:
            img = cv2.imread(file_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            if img is None:
                messagebox.showerror("Error", "No se pudo cargar la imagen.")
                return

            results = self.model(img)
            results.render()

            prediction_img = Image.fromarray(results.ims[0])
            prediction_img = prediction_img.resize((800, 600), Image.LANCZOS)
            prediction_img_tk = ImageTk.PhotoImage(prediction_img)

            self.prediction_image_label.img_tk = prediction_img_tk
            self.prediction_image_label.config(image=prediction_img_tk)

if __name__ == "__main__":
    pathlib.PosixPath = pathlib.WindowsPath
    root = tk.Tk()
    app = ObjectDetectionApp(root)
    root.mainloop()
