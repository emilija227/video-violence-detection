import cv2
import numpy as np

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

    def _extract_metadata(self):    #citanje metapodataka
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.frame_counter = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.duration_seconds = self.frame_counter / self.fps if self.fps > 0 else 0

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

        ret, frame = self.cap.read()

        if not ret:
            return None

        return frame

    def get_frame_count(self):
        """Vraća indeks trenutno pročitanog frejma (koristi ga OpenCV)."""
        if self.cap is None:
            return 0
        # CAP_PROP_POS_FRAMES vraća indeks frejma koji će biti obrađen SLEDEĆI
        # Stoga oduzimamo 1 da bismo dobili indeks upravo obrađenog frejma.
        return int(self.cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1

    def get_timestamp_ms(self):
        """Vraća vreme trenutnog frejma u milisekundama (koristi ga OpenCV)."""
        if self.cap is None:
            return 0
        return self.cap.get(cv2.CAP_PROP_POS_MSEC)

    def get_fps(self):
        """Vraća broj frejmova u sekundi (FPS) videa."""
        if self.cap is None:
            return 0
        return self.cap.get(cv2.CAP_PROP_FPS)

    def release(self):
        """Oslobađa video objekat (zatvara video fajl)."""
        if self.cap is not None:
            self.cap.release()


if __name__ == '__main__':
    # MORATE PROMENITI OVU PUTANJU NA VAŠ TEST VIDEO!
    TEST_VIDEO_PATH = r"C:\Users\emili\Videos\4K Video Downloader+\Short animation (no sound).mp4"
    #TEST_VIDEO_PATH = r"C:\Users\emili\PycharmProjects\praksa\Short animation (no sound) [dj3MA88qa3c].mp4"

    handler = VideoHandler(TEST_VIDEO_PATH)

    if handler.is_open:
        print("--- METAPODACI ---")
        print(handler.get_metadata())

        # Testiranje čitanja frejma
        frame_50 = handler.get_frame(50)

        if frame_50 is not None:
            print("\n--- TEST ČITANJA FREJMA ---")
            print(f"Uspešno pročitan frejm 50. Dimenzije: {frame_50.shape}, Tip: {frame_50.dtype}")
            # Prikaz frejma (Opcija, ovo je sporije za testiranje)
            # cv2.imshow('Frejm 50', frame_50)
            # cv2.waitKey(0) # Čeka dok se ne pritisne taster
            # cv2.destroyAllWindows()
        else:
            print("GREŠKA: Nije moguće pročitati frejm 50.")

    handler.close()
