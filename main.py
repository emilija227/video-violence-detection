import cv2
import numpy as np
import os
import math
import time
import customtkinter as ctk
from PIL import Image, ImageTk
import threading
from tkinter import filedialog

from video_handler import VideoHandler
from ml_model_handler import MLModelHandler

#GLOBALNE VARIJABLE
stop_event = threading.Event()
current_thread = None
video_handler = None
ml_model_handler = None
video_label_ref = None
root_app = None
report_textbox_ref = None
last_video_path = None
violence_color = (0, 0, 255)
MAX_DISPLAY_HEIGHT = 720
MAX_DISPLAY_WIDTH = 900
start_time_of_accident= None
is_violence_active = None
current_playback_speed = 1.0
speed_label_ref = None


def save_report_to_file():
    global report_textbox_ref

    if not report_textbox_ref:
        print("Izveštaj nije inicijalizovan.")
        return

    # 1. Otvaranje dijaloga za čuvanje fajla
    # Default ime je "izvestaj.txt", a filetypes filtrira fajlove
    filepath = filedialog.asksaveasfilename(
        defaultextension=".txt",
        filetypes=[("Text files", "*.txt"), ("All files", "*.*")],
        title="Sačuvaj izveštaj o detekciji"
    )

    if not filepath:
        # Korisnik je otkazao dijalog
        return

    try:
        # 2. Preuzimanje celog sadržaja iz Textboxa
        # Tekst se preuzima od prve linije (1.0) do kraja (end) minus zadnji novi red
        report_content = report_textbox_ref.get("1.0", "end-1c")

        # 3. Čuvanje sadržaja u izabrani fajl
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(report_content)

        # Optionalno: Ažuriranje statusa
        print(f"Izveštaj uspešno sačuvan: {filepath}")

    except Exception as e:
        print(f"Greška pri čuvanju fajla: {e}")

def set_playback_speed(value):
    global current_playback_speed, speed_label_ref

    valid_speeds = [0.25, 0.5, 1.0, 1.25, 1.5]
    closest_speed = min(valid_speeds, key=lambda x: abs(x - value))
    current_playback_speed = closest_speed

    if speed_label_ref:
        speed_label_ref.configure(text=f"Brzina: {current_playback_speed}x")


def update_gui_frame(cv2_frame, current_width, current_height):
    global video_label_ref

    if video_label_ref and cv2_frame is not None:
        #Konverzija BGR u RGB
        frame_rgb = cv2.cvtColor(cv2_frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame_rgb)

        imgTk = ctk.CTkImage(light_image=img, dark_image=img, size=(current_width, current_height))

        video_label_ref.configure(image=imgTk, text="")
        video_label_ref.image = imgTk

def update_report_textbox(message):

    global report_textbox_ref
    if report_textbox_ref:
        report_textbox_ref.configure(state="normal") # Privremeno omogući pisanje
        report_textbox_ref.insert("end", message + "\n")
        report_textbox_ref.see("end") # Skroluje do dna
        report_textbox_ref.configure(state="disabled") # Onemogući ponovo


def video_processing_loop():
    global video_handler, ml_model_handler, root_app, stop_event, violence_color
    global MAX_DISPLAY_WIDTH, is_violence_active, start_time_of_accident
    global current_playback_speed

    frames_to_skip = 4
    last_known_detections = []

    while not stop_event.is_set():
        frame = video_handler.get_frame()
        if frame is None:
            break

        frame_id = video_handler.get_frame_count()
        timestamp_ms = video_handler.get_timestamp_ms()
        timestamp_sec = timestamp_ms / 1000.0

        # 1. AŽURIRANA LOGIKA ZA FORMATIRANJE VREMENA (MM:SS.ms)
        minutes = int(timestamp_sec / 60)
        seconds_full = timestamp_sec % 60
        seconds_integer = int(seconds_full)
        milliseconds_decimal = seconds_full - seconds_integer
        milliseconds = int(milliseconds_decimal * 100)  # Prve dve decimale

        time_display = f"{minutes:02d}:{seconds_integer:02d}.{milliseconds:02d}"

        # 1. Detekcija (svaki frames_to_skip frejm)
        if frame_id % frames_to_skip == 0:
            detections = ml_model_handler.detect_violence(frame)
            last_known_detections = detections
        else:
            detections = last_known_detections

        # 2. DODAVANJE IZVEŠTAJA U TEXTBOX i Iscrtavanje
        if detections:
            if not is_violence_active:
                is_violence_active = True
                start_time_of_accident = time_display
                log_message = f"NASILJE! Pouzdanost: {detections[0]['score']:.2f}"
                root_app.after(0, lambda msg=log_message: update_report_textbox(msg))


            for det in detections:
                [xmin, ymin, xmax, ymax] = det['box']
                score = det['score']

                cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), violence_color, 2)

                label = f'Fight: {score:.2f} @ {time_display}'
                cv2.putText(frame, label, (xmin, ymin - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, violence_color, 2)
        else:
            if is_violence_active:
                is_violence_active = False
                end_time = time_display
                log_message = f"Početak: {start_time_of_accident} / Kraj detekcije: {end_time}  \n"
                root_app.after(0, lambda msg=log_message: update_report_textbox(msg))
                start_time_of_accident = None  # Resetujemo vreme


        cv2.putText(frame, f"Time: {time_display}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        # 3. AŽURIRANA LOGIKA SKALIRANJA (Uključuje i Širinu)
        current_height = frame.shape[0]
        current_width = frame.shape[1]

        scale_height = 1.0
        scale_width = 1.0

        # Izračunajte faktor skaliranja za visinu
        if current_height > MAX_DISPLAY_HEIGHT:
            scale_height = MAX_DISPLAY_HEIGHT / current_height

        # Izračunajte faktor skaliranja za širinu (potrebno je definisati MAX_DISPLAY_WIDTH globalno!)
        if 'MAX_DISPLAY_WIDTH' in globals() and current_width > MAX_DISPLAY_WIDTH:
            scale_width = MAX_DISPLAY_WIDTH / current_width

        # Uzmite manji faktor skaliranja da bi se slika uklopila unutar oba ograničenja
        final_scale = min(scale_height, scale_width)

        if final_scale < 1.0:
            new_width = int(current_width * final_scale)
            new_height = int(current_height * final_scale)

            resized_frame = cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_AREA)

            current_width = new_width
            current_height = new_height
        else:
            resized_frame = frame

        # 4. Ažuriranje GUI-a (MORA ići preko root_app.after)
        root_app.after(0, lambda frame=resized_frame, w=current_width, h=current_height: update_gui_frame(frame, w, h))

        base_sleep = 0.020

        sleep_duration = base_sleep / current_playback_speed
        time.sleep(sleep_duration)

    if video_handler:
        video_handler.release()
    print("Obrada videa završena u radnoj niti.")

    if is_violence_active:
        end_time = time_display
        log_message = f"Početak: {start_time_of_accident} / Kraj detekcije (KRAJ VIDEA): {end_time}\n"
        root_app.after(0, lambda msg=log_message: update_report_textbox(msg))

    root_app.after(0, lambda: video_label_ref.configure(text="Detekcija završena.", image=None))


def load_video():
    global current_thread, stop_event, video_handler, ml_model_handler, last_video_path
    # 1. Zaustavljanje prethodnog thread-a ako je aktivan
    if current_thread and current_thread.is_alive():
        stop_event.set()
        current_thread.join()
        if video_handler:
            video_handler.release()

    stop_event.clear()

    # 2. Otvaranje dijaloga za izbor video fajla
    path = filedialog.askopenfilename(
        title="Izaberi video fajl",
        filetypes=(("Video fajlovi", "*.mp4 *.avi *.mov"), ("Svi fajlovi", "*.*"))
    )

    if not path:
        return

    last_video_path = path
    start_processing(path)

def start_processing(path):
    global current_thread, stop_event, video_handler, ml_model_handler, stop_event, report_textbox_ref, video_label_ref

    # 3. Inicijalizacija i pokretanje
    ml_model_handler = MLModelHandler()  # Model inicijalizovan (ako ne uspe, GUI će javiti grešku)
    video_handler = VideoHandler(path)

    if report_textbox_ref:
        report_textbox_ref.configure(state="normal")
        report_textbox_ref.delete("1.0", "end")
        report_textbox_ref.insert("end", f"Izveštaj za: {os.path.basename(path)}\n")
        report_textbox_ref.configure(state="disabled")

    if ml_model_handler.model is None or not video_handler.cap.isOpened():
        if video_label_ref:
            video_label_ref.configure(text="Neuspelo učitavanje videa/modela.", image=None)
        return

    # 4. Pokretanje radne niti (glavna petlja ide ovde)
    current_thread = threading.Thread(target=video_processing_loop, daemon=True)
    current_thread.start()

def restart_video():
    global last_video_path, current_thread, video_handler, stop_event

    if last_video_path is None:
        if video_label_ref:
            video_label_ref.configure(text="Nijedan video nije prethodno učitan. Koristite 'Load Video'.", image=None)
        return

    if current_thread and current_thread.is_alive():
        stop_event.set()
        current_thread.join()
        if video_handler:
            video_handler.release()

    stop_event.clear()

    start_processing(last_video_path)

def on_closing():
    global stop_event, current_thread, video_handler, root_app

    stop_event.set()
    if current_thread and current_thread.is_alive():
        current_thread.join(timeout=1)  # Čekaj 1 sekundu da se nit završi

    if video_handler:
        video_handler.release()

    root_app.destroy()


if __name__ == '__main__':

    # 1. Postavljanje GUI stila
    ctk.set_appearance_mode("System")
    ctk.set_default_color_theme("blue")

    # 2. Inicijalizacija glavnog prozora
    root_app = ctk.CTk()
    root_app.title("Violence Detection (YOLOv8) GUI")
    root_app.geometry("1400x800")
    root_app.grid_columnconfigure(0, weight=1)

    # 3. Postavljanje sigurne funkcije za zatvaranje
    root_app.protocol("WM_DELETE_WINDOW", on_closing)

    # Red 0: Dugmad Load/Restart (Gornji Bar)
    main_frame = ctk.CTkFrame(root_app)
    main_frame.grid(row=0, column=0, sticky="ew", padx=10, pady=(10, 5))
    main_frame.grid_columnconfigure(0, weight=1) # Load Video se širi
    main_frame.grid_columnconfigure(1, weight=0) # Restart je fiksne širine

    load_video_btn = ctk.CTkButton(main_frame, text="Load Video", command=load_video)
    load_video_btn.grid(row=0, column=0, pady=10, padx=10, sticky="ew")

    # Restart dugme sa fiksnom širinom
    restart_video_btn = ctk.CTkButton(main_frame, text="Restart", command=restart_video, width=100)
    restart_video_btn.grid(row=0, column=1, pady=10, padx=5, sticky="e")


    # Red 1: KONTEJNER (Frame) za Video i Desni Panel
    content_frame = ctk.CTkFrame(root_app, fg_color="transparent")
    content_frame.grid(row=1, column=0, sticky="nsew", padx=10, pady=(5, 10))
    content_frame.grid_columnconfigure(0, weight=4) # Video
    content_frame.grid_columnconfigure(1, weight=3) # Desni Panel (Izveštaj + Kontrole)
    content_frame.grid_rowconfigure(0, weight=1)

    # --- LEVA STRANA (VIDEO) ---
    video_label_ref = ctk.CTkLabel(content_frame, text="Učitaj video za pokretanje detekcije.",
                                   compound="top",
                                   fg_color=("gray80", "gray25"),
                                   corner_radius=5)
    video_label_ref.grid(row=0, column=0, sticky="nsew", padx=(0, 5), pady=0)


    # --- DESNI PANEL (IZVEŠTAJ + KONTROLE BRZINE) ---
    right_panel_frame = ctk.CTkFrame(content_frame, fg_color="transparent")
    right_panel_frame.grid(row=0, column=1, sticky="nsew", padx=(5, 0), pady=0)
    right_panel_frame.grid_columnconfigure(0, weight=3) # Kontejner za Izveštaj
    right_panel_frame.grid_columnconfigure(1, weight=1) # Kontrole Brzine
    right_panel_frame.grid_rowconfigure(0, weight=1)

    # --- A. KONTROLE BRZINE (Desno u Desnom Panelu) ---
    speed_frame = ctk.CTkFrame(right_panel_frame, fg_color="transparent")
    speed_frame.grid(row=0, column=1, sticky="nsew", padx=(5, 0), pady=0)
    speed_frame.grid_columnconfigure(0, weight=1)
    speed_frame.grid_rowconfigure(1, weight=1)

    speed_label_ref = ctk.CTkLabel(
        speed_frame,
        text=f"Brzina: {current_playback_speed}x",
        font=ctk.CTkFont(size=14, weight="bold")
    )
    speed_label_ref.grid(row=0, column=0, pady=(10, 5), padx=5, sticky="ew")

    speed_slider = ctk.CTkSlider(
        speed_frame,
        from_=0.25,
        to=1.5,
        number_of_steps=5,
        command=set_playback_speed,
        orientation="vertical"
    )
    speed_slider.set(current_playback_speed)
    speed_slider.grid(row=1, column=0, pady=(0, 10), padx=10, sticky="ns")

    # --- B. IZVEŠTAJ I DUGME (Levo u Desnom Panelu) ---
    report_container = ctk.CTkFrame(right_panel_frame, fg_color=("gray80", "gray25"))
    report_container.grid(row=0, column=0, sticky="nsew", padx=(0, 5), pady=0)
    report_container.grid_columnconfigure(0, weight=1)
    report_container.grid_rowconfigure(0, weight=1)

    # Text Box (report_textbox_ref)
    report_textbox_ref = ctk.CTkTextbox(report_container,
                                        wrap="word",
                                        state="disabled",
                                        fg_color=("gray80", "gray25"))
    report_textbox_ref.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)

    # DUGME ZA SAČUVANJE
    save_report_btn = ctk.CTkButton(
        report_container,
        text="💾 Sačuvaj izveštaj (.txt)",
        command=save_report_to_file,
        width=150
    )
    save_report_btn.grid(row=1, column=0, pady=(0, 5), padx=5, sticky="se")

    # Inicijalni tekst
    report_textbox_ref.configure(state="normal")
    report_textbox_ref.insert("end", "Izveštaj o detekciji nasilja:\n")
    report_textbox_ref.configure(state="disabled")

    root_app.mainloop()