from typing import List, Text
import re
import spacy
import scispacy
from nltk.corpus import stopwords
from spacy.language import Language

class TextCleaner:
    def __init__(self):
        self.nlp = spacy.load("en_core_sci_lg")
        self.stopwords = set(stopwords.words('english'))
        # Stopwords biomédicas adicionales
        self.bio_stopwords = {
            'study', 'patient', 'result', 'method', 'conclusion',
            'background', 'objective', 'aim', 'purpose'
        }
        self.stopwords.update(self.bio_stopwords)

    def clean_text(self, text: str) -> str:
        """Limpieza básica del texto"""
        # Convertir a minúsculas
        text = text.lower()
        # Eliminar caracteres especiales
        text = re.sub(r'[^\w\s]', ' ', text)
        # Eliminar números
        text = re.sub(r'\d+', '', text)
        # Eliminar espacios múltiples
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    def remove_stopwords(self, text: str) -> str:
        """Eliminar stopwords"""
        words = text.split()
        words = [w for w in words if w not in self.stopwords]
        return ' '.join(words)

    def lemmatize(self, text: str) -> str:
        """Lematización usando scispaCy"""
        doc = self.nlp(text)
        return ' '.join([token.lemma_ for token in doc])