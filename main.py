import cv2
import numpy as np
import os
import math

from video_handler import VideoHandler
from ml_model_handler import MLModelHandler

test_video_path = r"C:\Users\emili\Videos\4K Video Downloader+\whos got the best scream (not mine btw just sharing).mp4"
violence_color = (0, 0, 255)  # Crvena boja za detekciju nasilja

MAX_DISPLAY_HEIGHT = 800

def main():
    if not os.path.exists(test_video_path):
        print("GREŠKA: Video nije pronađen")
        print("Ažurirajte test_video_path na ispravnu putanju.")
        return

    video_handler = VideoHandler(test_video_path)
    ml_model_handler = MLModelHandler()

    if ml_model_handler.model is None:
        print("Neuspješno učitavanje ML modela. Prekidanje izvršavanja.")
        return

    print("\nPočetak obrade videa")

    frames_to_skip = 4
    last_known_detections = []
    # violence_in_previous_frame = False

    while True:  # Sve dok ima frejmova za čitanje
        frame = video_handler.get_frame()
        if frame is None:
            break

        frame_id = video_handler.get_frame_count()
        timestamp_ms = video_handler.get_timestamp_ms()
        timestamp_sec = timestamp_ms / 1000.0

        minutes = int(timestamp_sec / 60)
        seconds = timestamp_sec % 60
        time_display = f"{minutes:02}:{seconds:05.2f}"

        if frame_id % frames_to_skip == 0:
            detections = ml_model_handler.detect_violence(
                frame)  # Proslijedim frejm YOLO modelu i dobijam nazad listu detekcije nasilja
            last_known_detections = detections
        else:
            detections = last_known_detections

        if detections:
            print(
                f"Detektovano nasilje! Vrijeme: {time_display}, Frejm: {frame_id}, Pouzdanost: {detections[0]['score']:.2f}")

        # Iscrtava detekcione kutije na frejmu
        for det in detections:
            [xmin, ymin, xmax, ymax] = det['box']
            score = det['score']

            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), violence_color, 2)

            label = f'Fight: {score:.2f} @ {time_display}'
            cv2.putText(frame, label, (xmin, ymin - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, violence_color, 2)

        cv2.putText(frame, f"Time: {time_display}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        #Prilagođeno za vertikalne videe - shorts
        resized_frame = frame
        height = frame.shape[0]

        # Provera da li je visina frejma veća od maksimalne dozvoljene za prikaz
        if height > MAX_DISPLAY_HEIGHT:
            width = frame.shape[1]

            scale_factor = MAX_DISPLAY_HEIGHT / height

            #Nove dimenzije
            new_width = int(width * scale_factor)
            new_height = int(height * scale_factor)

            resized_frame = cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_AREA)

        cv2.imshow('Violence Detection System (YOLOv8)', resized_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video_handler.release()
    cv2.destroyAllWindows()
    print("\nObrada videa završena.")


if __name__ == '__main__':
    main()