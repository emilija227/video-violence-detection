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
            self.model = YOLO(model_path) #Ucitavam preuzeti model
            print("YOLOv8 model uspjeno ocitan")

            self.violence_class_id = 1 #Postavljam ID klase za detekciju na 1, jer je to u ovom modelu violence

        except FileNotFoundError:
            print("GRESKA: Fajl modela nije pronadjen")
        except Exception as e:
            print(f"GRESKA pri ucitavanju YOLO modela: {e}")

    def detect_violence(self, frame_array): #Prima 1 frejm videa i vraca listu detekcija nasilja
        if self.model is None or frame_array is None:
            return []

        results = self.model.predict(frame_array, conf=0.5, iou=0.5, verbose=False) #Pokrecem YOLO algoritam na datom frejmu
        #Prag detekcije 50%
        #iou rjesava problem detekcije istog objekta vise puta
        #verbose na False da YOLO ne bi ispisivao tehnicke poruke

        violence_detections = []

        if results and results[0].boxes: #Da li je predict() vratio uopste rezultate i da li oni sadrze detektovane kutije
            boxes = results[0].boxes #Objekat koji sadrzi sve koordinate, klase i skorove za frejm

            #Iteriramo svaki put kada pronadjemo detekciju
            for i in range(len(boxes.cls)):
                class_id = int(boxes.cls[i].item())

                if class_id == self.violence_class_id: #Ako je klasa koja je detektovana = 1 tj 'violence'
                    box_coords = boxes.xyxy[i].cpu().numpy().astype(int) #Izvlacenje koordinata
                    confidence = boxes.conf[i].item()

                    violence_detections.append({
                        'box': box_coords,
                        'score': confidence
                    })

        return violence_detections
