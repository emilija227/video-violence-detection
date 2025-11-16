from ultralytics import YOLO
import numpy as np
import cv2
import os

model_path = os.path.join("models", "yolo_small_weights.pt")

class MLModelHandler:
    def __init__(self):
        self.model = None
        self.violence_class_id = None

        try:
            self.model = YOLO(model_path)
            print("YOLOv8 model uspjeno ocitan")

            self.violence_class_id = 1

        except FileNotFoundError:
            print("GRESKA: Fajl modela nije pronadjen")
        except Exception as e:
            print(f"GRESKA pri ucitavanju YOLO modela: {e}")

    def detect_violence(self, frame_array):
        if self.model is None or frame_array is None:
            return []

        results = self.model.predict(frame_array, conf=0.5, iou=0.5, verbose=False)

        violence_detections = []

        if results and results[0].boxes:
            boxes = results[0].boxes

            for i in range(len(boxes.cls)):
                class_id = int(boxes.cls[i].item())

                if class_id == self.violence_class_id:
                    box_coords = boxes.xyxy[i].cpu().numpy().astype(int)
                    confidence = boxes.conf[i].item()

                    violence_detections.append({
                        'box': box_coords,
                        'score': confidence
                    })

        return violence_detections


# Blok za testiranje
if __name__ == '__main__':
    # 1. Inicijalizacija handlera
    ml_handler = MLModelHandler()

    # 2. Testiranje (potreban nam je dummy frejm)
    if ml_handler.model:
        print("\n--- TEST YOLO DETEKCIJE ---")

        # Kreiramo lažni frejm 640x480 (YOLO optimalna ulazna veličina)
        dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)

        # Testiramo da li funkcija radi
        detections = ml_handler.detect_violence(dummy_frame)

        # Ako je sve ispravno instalirano, ovo treba da radi bez grešaka
        print(f"Simulirana detekcija na praznom frejmu. Rezultat (lista detekcija): {detections}")
        print("Model je uspešno obradio frejm.")