import cv2
import numpy as np


class VideoHandler:
    # Postavljamo clip_size i target_res da odgovaraju modelu
    CLIP_SIZE = 64
    TARGET_RES = (224, 224)

    def __init__(self, video_path):
        """Inicijalizuje video čitač i postavlja parametre klipa."""
        self.cap = cv2.VideoCapture(video_path)
        self.clip = []  # Lista za čuvanje tekuće sekvence frejmova (Sliding Window)

        if not self.cap.isOpened():
            print("Greška: Nije moguće otvoriti video fajl. Proverite putanju i FFmpeg.")

    def read_single_frame(self):
        """Čita jedan frejm iz videa i obrađuje ga."""
        ret, frame = self.cap.read()

        if not ret:
            # Kraj videa ili greška u čitanju
            return None, None

            # Promena veličine (resize) frejma na 224x224 (za model)
        resized_frame = cv2.resize(frame, self.TARGET_RES, interpolation=cv2.INTER_AREA)

        # Vraćamo obrađeni frejm (za model) i originalni frejm (za prikaz)
        return resized_frame, frame

    def get_sliding_window(self, processed_frame):
        """
        Dodaje novi frejm u klizni prozor i vraća klip za predikciju.

        :param processed_frame: Frejm dimenzija (224, 224, 3)
        :return: Numpy array klipa (64, 224, 224, 3) ili None
        """
        # Dodavanje novog frejma u listu klipova
        self.clip.append(processed_frame)

        if len(self.clip) < self.CLIP_SIZE:
            # Klip još nije pun (prvih 64 frejma)
            return None

        if len(self.clip) > self.CLIP_SIZE:
            # Ako je klip prepun (Sliding Window logika), izbacujemo najstariji frejm
            self.clip.pop(0)

        # Klip je tačno 64 frejma, pretvaramo ga u tenzor za model
        clip_array = np.array(self.clip)

        # Vraćamo tenzor spreman za predikciju
        return clip_array

    def release(self):
        """Oslobađa resurs kamere/videa."""
        self.cap.release()