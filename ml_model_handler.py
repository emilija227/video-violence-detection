import tensorflow as tf
import numpy as np

class MLModelHandler:
    def __init__(self, model_path):
        """Učitava 3D CNN model iz .h5 fajla."""
        self.model_path = model_path
        self.model = self._load_model()
        # Klase modela (0 za NonFight, 1 za Fight) - ISPREDENA KLASA
        self.classes = ['NonFight', 'Fight']

    def _load_model(self):
        """Pokušava da učita Keras model."""
        try:
            # Učitavanje celog modela
            model = tf.keras.models.load_model(self.model_path)
            print(f"Model uspešno učitan: {self.model_path}")
            return model
        except Exception as e:
            print(
                f"Greška pri učitavanju modela {self.model_path}. Proverite da li je TensorFlow instaliran i da li je putanja do .h5 fajla tačna. Greška: {e}")
            return None

    def predict(self, clip):
        """
        Vrši predikciju na klipu frejmova.
        :param clip: Numpy array, dimenzije (64, 224, 224, 3)
        :return: Tekstualna labela ('Fight' ili 'NonFight') i verovatnoća (float)
        """
        if self.model is None:
            return "ERROR: Model nije učitan", 0.0

        # Normalizacija i dodavanje batch dimenzije
        normalized_clip = clip.astype('float32') / 255.0
        input_tensor = np.expand_dims(normalized_clip, axis=0)

        # Predikcija
        prediction = self.model.predict(input_tensor, verbose=0)[0]

        # Pronalazak najveće verovatnoće
        predicted_index = np.argmax(prediction)

        label = self.classes[predicted_index]
        confidence = prediction[predicted_index]

        return label, confidence