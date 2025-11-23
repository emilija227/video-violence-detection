import tensorflow as tf
import numpy as np

class MLModelHandler:
    def __init__(self, model_path):
        #Ucitava 3D CNN model iz .h5 fajla.
        self.model_path = model_path
        self.model = self._load_model()
        self.classes = ['NonFight', 'Fight'] #[0,1]

    def _load_model(self):
        #Ucitavam model
        try:
            model = tf.keras.models.load_model(self.model_path) #Koristim keras funkciju za ucitavanje cijelog modela
            print(f"Model uspešno učitan: {self.model_path}")
            return model
        except Exception as e:
            print(
                f"Greška pri učitavanju modela {self.model_path}. Proverite da li je TensorFlow instaliran i da li je putanja do .h5 fajla tačna. Greška: {e}")
            return None

    def predict(self, clip):
        if self.model is None:
            return "ERROR: Model nije učitan", 0.0

        #Normalizacija i dodavanje batch dimenzije
        normalized_clip = clip.astype('float32') / 255.0
        input_tensor = np.expand_dims(normalized_clip, axis=0)

        #Predikcija
        prediction = self.model.predict(input_tensor, verbose=0)[0]

        #Pronalazak najveće verovatnoće
        predicted_index = np.argmax(prediction) #Npr ako je [0.9, 0.1] indeks je 0

        label = self.classes[predicted_index] #Uzima se odgovarajuca klasa na osnovu indeksa
        confidence = prediction[predicted_index]

        return label, confidence