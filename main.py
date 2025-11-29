import cv2
import os
from PIL import Image, ImageTk
from tkinter import filedialog
from ml_model_handler import MLModelHandler
from video_handler import VideoHandler
import customtkinter as ctk
import threading
import time  # Dodato za simulaciju kašnjenja (ako je potrebno)

MODEL_PATH = os.path.join('models', 'best_finetuned_model_v2.h5')
CONFIDENCE_THRESHOLD = 0.80
FRAME_SKIP = 2
MAX_DISPLAY_DIM = 720

# Globalne varijable za upravljanje
cap = None
video_handler = None
model_handler = None
root_app = None
video_label_ref = None
frames_read_counter = 0
current_label = "Cekanje na klip (64 frejma)"
current_confidence = 0.0
DISPLAY_DIMS = (0, 0)
FONT_SCALE = 0.5
THICKNESS = 1
TEXT_X = 0
TEXT_Y = 0

#Dodata globalna varijabla za kontrolu thread-a
stop_event = threading.Event()
current_thread = None

#Funkcija za inicijalizaciju

def initialize_video_display(path):
    global video_handler, model_handler, DISPLAY_DIMS, FONT_SCALE, THICKNESS, TEXT_X, TEXT_Y, frames_read_counter

    # 1. Inicijalizacija modela
    model_handler = MLModelHandler(MODEL_PATH)
    if model_handler.model is None:
        print("Aplikacija se zaustavlja jer model nije učitan.")
        return False

    # 2. Inicijalizacija Video čitača
    video_handler = VideoHandler(path)
    if not video_handler.cap.isOpened():
        # video_handler.release() # Release se dešava unutar VideoHandler-a pri kreiranju ako ne uspe
        return False

    #Podesavanja tako da odgovoraju i vertikalnim i horizontalnim videima
    original_width = int(video_handler.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    original_height = int(video_handler.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    #Logika za izračunavanje DISPLAY_DIMS
    if original_width > original_height:
        new_width = MAX_DISPLAY_DIM
        new_height = int(original_height * (new_width / original_width))
    else:
        new_height = MAX_DISPLAY_DIM
        new_width = int(original_width * (new_height / original_height))

    DISPLAY_DIMS = (new_width, new_height)

    # Veličina fonta i debljina
    FONT_SCALE = max(0.5, DISPLAY_DIMS[1] / 720.0)
    THICKNESS = max(1, int(DISPLAY_DIMS[1] / 300.0))

    # Položaj teksta
    TEXT_X = int(DISPLAY_DIMS[0] * 0.02)
    TEXT_Y = int(DISPLAY_DIMS[1] * 0.07)

    frames_read_counter = 0
    print(f"Pokretanje Detekcije Borbe (3D CNN) - Frame Skip: {FRAME_SKIP} ")
    return True


#Glavna petlja za obradu videa - u novom threadu

def video_processing_loop():
    #Glavna petlja koja čita frejmove, vrši predikciju i signalizira GUI-u
    #za ažuriranje. Radi u zasebnom thread-u.

    global video_handler, model_handler, frames_read_counter, current_label, current_confidence

    if video_handler is None or not video_handler.cap.isOpened():
        print("Video ili model nisu inicijalizovani.")
        return

    while not stop_event.is_set():
        # 1. Čitanje JEDNOG frejma (Brzo, I/O operacija)
        processed_frame_model, frame_for_display = video_handler.read_single_frame()

        # 2. Ako je kraj videa
        if processed_frame_model is None:
            break

        frames_read_counter += 1
        clip = video_handler.get_sliding_window(processed_frame_model)

        # 3. Predikcija (Teška operacija, dešava se u zasebnom thread-u)
        if clip is not None and frames_read_counter % (FRAME_SKIP + 1) == 0:
            label, confidence = model_handler.predict(clip)  # OVAJ DEO VIŠE NE BLOKIRA GUI

            # LOGIKA DETEKCIJE
            if label == 'Fight' and confidence >= CONFIDENCE_THRESHOLD:
                current_label = 'Fight'
                current_confidence = confidence
            else:
                current_label = 'NonFight'
                current_confidence = confidence

        # 4. Priprema frejma za prikaz
        display_frame = cv2.resize(frame_for_display, DISPLAY_DIMS, interpolation=cv2.INTER_LINEAR)

        # Aplikacija teksta na frame
        text_status = f"STATUS: {current_label} ({current_confidence:.2f})"
        color = (0, 0, 255) if current_label == 'Fight' else (0, 255, 0)

        cv2.putText(display_frame, text_status, (TEXT_X, TEXT_Y),
                    cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE, color, THICKNESS, cv2.LINE_AA)

        # Konverzija za prikaz u CTkinteru
        frame_rgb = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame_rgb)

        # --- GUI AŽURIRANJE SE MORA VRATITI U GLAVNI THREAD ---
        # Koristimo after() da to postignemo
        root_app.after(0, lambda frame=img: update_gui_frame(frame))

        # Kontrola brzine: Mogli biste ovde dodati time.sleep()
        # Ako je proces obrade brži od brzine videa, ali je često dovoljno
        # samo to što je deo obrade u threadu.
        # time.sleep(0.001) # Možete eksperimentisati ovde

    # Čišćenje resursa nakon što petlja završi (ili je zaustavljena)
    video_handler.release()
    print("Detekcija završena u radnom thread-u.")
    # Ažuriranje GUI-a nakon završetka
    root_app.after(0, lambda: video_label_ref.configure(text="Detekcija završena.", image=None))


def update_gui_frame(img):
    global video_label_ref
    if video_label_ref:
        imgTk = ctk.CTkImage(light_image=img, dark_image=img, size=DISPLAY_DIMS)
        video_label_ref.configure(image=imgTk, text="")
        video_label_ref.image = imgTk  # Čuvanje reference

#Funkcija za GUI, glavni thread
def load_video():
   #Otvori dijalog, učita video i POKRENE NOVI THREAD.

    global current_thread, stop_event, video_handler

    # 1. Zaustavljanje prethodnog thread-a ako je aktivan
    if current_thread and current_thread.is_alive():
        stop_event.set()
        current_thread.join()  #Čekamo da se stari thread završi
        if video_handler:
            video_handler.release()  #Oslobodi resurse

    stop_event.clear()  #Resetujemo event za novi thread

    path = filedialog.askopenfilename(
        title="Izaberi video fajl",
        filetypes=(("Video fajlovi", "*.mp4 *.avi *.mov"), ("Svi fajlovi", "*.*"))
    )

    if path:
        if initialize_video_display(path):
            # 2. Inicijalizacija i pokretanje NOVOG THREAD-a
            current_thread = threading.Thread(target=video_processing_loop, daemon=True)
            current_thread.start()
        else:
            if video_label_ref:
                video_label_ref.configure(text="Neuspelo učitavanje videa/modela.", image=None)


def on_closing():
    #Funkcija koja se poziva pri zatvaranju prozora kako bi se zaustavio radni thread.

    global stop_event, current_thread, video_handler

    # Zaustavi thread pre zatvaranja prozora
    stop_event.set()
    if current_thread and current_thread.is_alive():
        current_thread.join(timeout=1)

    if video_handler:
        video_handler.release()

    root_app.destroy()


if __name__ == "__main__":
    ctk.set_appearance_mode("System")
    ctk.set_default_color_theme("blue")

    root_app = ctk.CTk()
    root_app.title("Fight Detection Application (Multi-Threaded)")
    root_app.geometry("900x800")
    root_app.grid_columnconfigure(0, weight=1)
    root_app.grid_rowconfigure(1, weight=1)

    # Postavljanje funkcije koja se poziva pri zatvaranju prozora
    root_app.protocol("WM_DELETE_WINDOW", on_closing)

    # --- GUI Elementi (Isto kao pre) ---
    main_frame = ctk.CTkFrame(root_app)
    main_frame.grid(row=0, column=0, sticky="ew", padx=10, pady=(10, 5))
    main_frame.grid_columnconfigure(0, weight=1)

    load_video_btn = ctk.CTkButton(main_frame, text="Load Video", command=load_video)
    load_video_btn.grid(row=0, column=0, pady=10, padx=10)

    video_label_ref = ctk.CTkLabel(root_app, text="Učitaj video za pokretanje detekcije.",
                                   compound="top",
                                   width=MAX_DISPLAY_DIM, height=MAX_DISPLAY_DIM,
                                   fg_color=("gray80", "gray25"),
                                   corner_radius=5)
    video_label_ref.grid(row=1, column=0, sticky="nsew", padx=10, pady=(5, 10))

    root_app.mainloop()