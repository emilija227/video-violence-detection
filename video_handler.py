import cv2
import numpy as np

class VideoHandler:
    # Postavljamo clip_size i target_res da odgovaraju modelu
    CLIP_SIZE = 64
    TARGET_RES = (224, 224)

    def __init__(self, video_path):
        self.cap = cv2.VideoCapture(video_path)  # Inicijalizuje video čitač i postavlja parametre klipa.
        self.clip = []  # Lista za čuvanje tekuće sekvence frejmova (Sliding Window)

        if not self.cap.isOpened():
            print("Greška: Nije moguće otvoriti video fajl. Proverite putanju i FFmpeg.")

    def read_single_frame(self):
        ret, frame = self.cap.read()   #Čita jedan frejm iz videa i obrađuje ga.

        if not ret:
            return None, None

        # Promena veličine (resize) frejma na 224x224 (za model)
        resized_frame = cv2.resize(frame, self.TARGET_RES, interpolation=cv2.INTER_AREA)

        return resized_frame, frame  # Vraćamo obrađeni frejm (za model) i originalni frejm (za prikaz)


    def get_sliding_window(self, processed_frame):
        self.clip.append(processed_frame) # Dodavanje novog frejma u listu klipova

        if len(self.clip) < self.CLIP_SIZE:
            return None # Klip još nije pun (prvih 64 frejma)

        if len(self.clip) > self.CLIP_SIZE:
            self.clip.pop(0) # Ako je klip prepun (Sliding Window logika), izbacujemo najstariji frejm

        clip_array = np.array(self.clip) # Klip je tačno 64 frejma, pretvaramo ga u tenzor za model

        return clip_array

    def release(self):
        self.cap.release() #Oslobadjanje resursa