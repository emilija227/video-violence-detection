import cv2
import numpy as np
import os
import math

from video_handler import VideoHandler
from ml_model_handler import MLModelHandler

test_video_path = r"C:\Users\emili\Videos\4K Video Downloader+\Superhero fighting editing in Capcut in Hindi   Superman vs General zod   video editing tutorial.mp4"
violence_color = (0,0,255)

def main():
    if not os.path.exists(test_video_path):
        print(f"FATALNA GREŠKA: Video nije pronađen: {test_video_path}")
        print("Ažurirajte test_video_path na ispravnu putanju.")
        return

    video_handler = VideoHandler(test_video_path)
    ml_model_handler = MLModelHandler()

    if ml_model_handler.model is None:
        print("Neuspešno učitavanje ML modela. Prekidanje izvršavanja.")
        return

    print("\n--- Početak obrade videa ---")

    while True:
        frame = video_handler.get_frame()
        if frame is None:
            break

        frame_id = video_handler.get_frame_count()
        timestamp_ms = video_handler.get_timestamp_ms()
        timestamp_sec = timestamp_ms / 1000.0

        minutes = int(timestamp_sec / 60)
        seconds = timestamp_sec % 60
        time_display = f"{minutes:02}:{seconds:05.2f}"

        detections = ml_model_handler.detect_violence(frame)

        if detections:
            print(f"🚨 NASILJE DETEKTOVANO u frejmu {frame_id} @ {time_display}s | Broj incidenata: {len(detections)}")

        for det in detections:
            [xmin, ymin, xmax, ymax] = det['box']
            score = det['score']

            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), violence_color, 2)

            label = f'Fight: {score:.2f} @ {time_display}'
            cv2.putText(frame, label, (xmin, ymin - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, violence_color, 2)

        cv2.putText(frame, f"Time: {time_display}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        cv2.imshow('Violence Detection System (YOLOv8)', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video_handler.release()
    cv2.destroyAllWindows()
    print("\n--- Obrada videa završena. ---")

if __name__ == '__main__':
    main()