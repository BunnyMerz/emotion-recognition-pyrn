import cv2
import numpy as np
from train import NeuralNetwork
import torch

class Camera:
    cam = cv2.VideoCapture(0)
    fer = torch.load("fer.pth")
    fer.eval()

    def read():
        return Camera.cam.read()

    def get_face():
        ret, image = Camera.read()
        if not ret:
            return

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        facesCascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        face = facesCascade.detectMultiScale(
            gray,
            scaleFactor=1.3,
            minNeighbors=3,
            minSize=(30, 30)
        )[0]

        x, y, w, h = face
        return gray[y:y+h,x:x+w]

    def get_emotion(callback_fn=lambda x: None) -> int:
        face = Camera.get_face()

        resized_face = torch.from_numpy(np.expand_dims(cv2.resize(face,(48,48)),0).astype('float32'))
        batch = resized_face.unsqueeze(0)

        with torch.no_grad():
            output = Camera.fer(batch)[0]

        emotion = torch.argmax(output)
        callback_fn(emotion)
        