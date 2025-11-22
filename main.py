import cv2
import os
from ml_model_handler import MLModelHandler
from video_handler import VideoHandler

VIDEO_PATH = r"C:\Users\emili\Videos\4K Video Downloader+\The Best Uppercut KO Of All Time.mp4"

MODEL_PATH = os.path.join('models', 'fight_detection_model.h5')

CONFIDENCE_THRESHOLD = 0.80

FRAME_SKIP = 2 #Ubrzavam, predikcija se vrsi na svakom (frame_skip+1) frejmu


def run_prediction():
    # 1. Inicijalizacija modela
    model_handler = MLModelHandler(MODEL_PATH)

    if model_handler.model is None:
        print("Aplikacija se zaustavlja jer model nije učitan.")
        return

    # 2. Inicijalizacija Video citača
    video_handler = VideoHandler(VIDEO_PATH)

    if not video_handler.cap.isOpened():
        video_handler.release()
        return

    #Podesavanja tako da odgovoraju i vertikalnim i horizontalnim videima
    original_width = int(video_handler.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    original_height = int(video_handler.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    MAX_DISPLAY_DIM = 720

    if original_width > original_height:
        #Horizontalni video
        new_width = MAX_DISPLAY_DIM
        new_height = int(original_height * (new_width / original_width))
    else:
        #Vertikalni video (Shorts)
        new_height = MAX_DISPLAY_DIM
        new_width = int(original_width * (new_height / original_height))

    DISPLAY_DIMS = (new_width, new_height)

    #Veličina fonta i debljina se skaliraju prema veličini ekrana
    FONT_SCALE = max(0.5, DISPLAY_DIMS[1] / 720.0)
    THICKNESS = max(1, int(DISPLAY_DIMS[1] / 300.0))

    #Položaj teksta
    TEXT_X = int(DISPLAY_DIMS[0] * 0.02)
    TEXT_Y = int(DISPLAY_DIMS[1] * 0.07)
    # --- KRAJ PODEŠAVANJA ---

    # Inicijalizacija varijabli za detekciju
    current_label = "Cekanje na klip (64 frejma)..."
    current_confidence = 0.0
    frames_read_counter = 0

    print(f"Pokretanje Detekcije Borbe (3D CNN) - Frame Skip: {FRAME_SKIP} ")

    # 3. Glavna Petlja (Frame-by-Frame čitanje)
    while True:
        #Čitanje JEDNOG frejma
        processed_frame_model, frame_for_display = video_handler.read_single_frame()

        #Ako je kraj videa
        if processed_frame_model is None:
            break

        #Brojač se povećava za svaki pročitani frejm
        frames_read_counter += 1

        #Dodajemo frejm u klizni prozor
        clip = video_handler.get_sliding_window(processed_frame_model)

        # 4. Predikcija: Vrši se tek kada je klip pun i ISTEKNE BROJAČ
        #Time se smanjuje broj poziva TensorFlow-u i ubrzava video
        if clip is not None and frames_read_counter % (FRAME_SKIP + 1) == 0:
            label, confidence = model_handler.predict(clip)

            #LOGIKA DETEKCIJE SA PRAGOM
            if label == 'Fight' and confidence >= CONFIDENCE_THRESHOLD:
                current_label = 'Fight'
                current_confidence = confidence
            else:
                current_label = 'NonFight'
                current_confidence = confidence

        # 5. Prikazivanje Frejma i Rezultata (Prikaz se dešava na SVAKOM frejmu)

        #Povećavanje frejma koristeći dinamički izračunate dimenzije
        display_frame = cv2.resize(frame_for_display, DISPLAY_DIMS, interpolation=cv2.INTER_LINEAR)

        #Aplikacija teksta na frame
        text_status = f"STATUS: {current_label} ({current_confidence:.2f})"

        #Postavljanje crvene boje ako je detektovana borba
        color = (0, 0, 255) if current_label == 'Fight' else (0, 255, 0)

        #Koristimo optimizovane, izračunate vrednosti za veličinu i poziciju teksta
        cv2.putText(display_frame, text_status, (TEXT_X, TEXT_Y),
                    cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE, color, THICKNESS, cv2.LINE_AA)

        cv2.imshow('3D CNN Detekcija Borbe', display_frame)

        #Ubrzani prikaz: Kašnjenje postavljeno na minimum (1 ms)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    #Oslobađanje resursa
    video_handler.release()
    cv2.destroyAllWindows()
    print("--- Detekcija završena. ---")


if __name__ == "__main__":
    if not os.path.exists(VIDEO_PATH):
        print(f"Greška: Video fajl nije pronađen na putanji: {VIDEO_PATH}")
    else:
        run_prediction()