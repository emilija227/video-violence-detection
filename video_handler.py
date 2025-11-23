import cv2
import numpy as np

class VideoHandler:
    #Konstante klase
    CLIP_SIZE = 64 #Frejmovi u sekvenci
    TARGET_RES = (224, 224) #Ciljna rezolucija na koju ce svaki frejm da bude skaliran

    def __init__(self, video_path):
        self.cap = cv2.VideoCapture(video_path)   #Inicijalizuje video čitač i postavlja parametre klipa.
        self.clip = []  #Lista u kojoj se cuvaju obradjeni frejmovi prije nego sto se formira cijeli klip od 64 frejma (Sliding Window)

        if not self.cap.isOpened():
            print("Greška: Nije moguće otvoriti video fajl. Proverite putanju i FFmpeg.")

    def read_single_frame(self):
        ret, frame = self.cap.read()   #Cita jedan frejm iz videa i obrađuje ga.

        if not ret: #Ako nije uspjesno ucitan frejm
            return None, None

        #Preprocesiranje, mijenja rezoluciju frejma na onu koju ocekuje model
        resized_frame = cv2.resize(frame, self.TARGET_RES, interpolation=cv2.INTER_AREA)

        return resized_frame, frame  #Vraćamo obrađeni frejm (za model) i originalni frejm (za prikaz)

    #Posto formiram klipove od 64 frejma, koristim metodu sliding window
    def get_sliding_window(self, processed_frame):
        self.clip.append(processed_frame) #Dodavanje novoobradjenog frejma u listu klipova

        #Provjeravam da li je lista manja od 64
        if len(self.clip) < self.CLIP_SIZE:
            return None

        if len(self.clip) > self.CLIP_SIZE:
            self.clip.pop(0) #Ako je klip prepun (Sliding Window logika), izbacujemo najstariji frejm

        clip_array = np.array(self.clip) #Klip je tačno 64 frejma, pretvaramo ga u tenzor za model

        return clip_array

    def release(self):
        self.cap.release() #Oslobadjanje resursa