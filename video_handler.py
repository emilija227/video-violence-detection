import cv2
import numpy as np

#Pravim klasu pomocu koje otvaram video i izvlacim njegove metapodatke
class VideoHandler:
    def __init__(self, video_path):
        self.video_path = video_path
        self.cap = cv2.VideoCapture(video_path)
        self.is_open = self.cap.isOpened()

        if not self.is_open:
            print("Greska pri otvaranju videa na putanji: {video_path}")
            self.fps = 0
            self.frame_counter = 0
            self.width = 0
            self.height = 0
        else:
            self._extract_metadata()

    def _extract_metadata(self):    #Citanje metapodataka, koristim funkcije iz OpenCV-a
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)   #Broj frejmova u sekundi
        self.frame_counter = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))    #Ukupan broj frejmova u videu
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))            #Rezolucija
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))          #Rezolucija
        self.duration_seconds = self.frame_counter / self.fps if self.fps > 0 else 0    #Ukupan broj frejmova / FPS, if da izbjegnem dijeljenje s nulom

    def get_metadata(self):
        #Vracam podatke kao rjecnik
        return {
            "path": self.video_path,
            "is_open": self.is_open,
            "FPS": self.fps,
            "Total frames": self.frame_counter,
            "Resolution": f"{self.width}x{self.height}",
            "Duration (s)": round(self.duration_seconds, 2)
        }

    def get_frame(self):
        if self.cap is None or not self.cap.isOpened():
            return None

        ret, frame = self.cap.read()    #ret govori da li je citanje bilo uspjesno, frame je sama slika tj numpy niz

        if not ret:
            return None

        return frame

    def get_frame_count(self): #Vraca indeks trenutno procitanog frejma
        if self.cap is None:
            return 0

        return int(self.cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1
    #CAP_PROP_POS_FRAMES vraca indeks sljedeceg frejma pa moramo da oduzmemo 1

    def get_timestamp_ms(self): #Vraca vremenski trenutak trenutnog frejma u ms
        if self.cap is None:
            return 0

        return self.cap.get(cv2.CAP_PROP_POS_MSEC)

    def get_fps(self): #U slucaju da mi zatreba fps
        if self.cap is None:
            return 0
        return self.cap.get(cv2.CAP_PROP_FPS)

    def release(self): #Zatvaram video, oslobadjam resurse
        if self.cap is not None:
            self.cap.release()
