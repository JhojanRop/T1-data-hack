import re
import spacy
import logging
from nltk.corpus import stopwords
from spacy.language import Language

class TextCleaner:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        try:
            self.nlp = spacy.load("en_core_sci_lg")
        except OSError:
            self.logger.error("Failed to load en_core_sci_lg model")
            raise
            
        self.stopwords = set(stopwords.words('english'))
        # Stopwords biomédicas adicionales
        self.bio_stopwords = {
            'study', 'patient', 'result', 'method', 'conclusion',
            'background', 'objective', 'aim', 'purpose', 'material',
            'finding', 'analysis', 'data', 'evaluation', 'research',
            'significant', 'clinical', 'treatment', 'group', 'control',
            'baseline', 'follow', 'month', 'year', 'day', 'time',
            'measure', 'parameter', 'value', 'level', 'change'
        }
        self.stopwords.update(self.bio_stopwords)

    def clean(self, text: str) -> str:
        """
        Aplica todas las etapas de limpieza al texto
        
        Args:
            text (str): Texto a limpiar
            
        Returns:
            str: Texto limpio
        """
        try:
            if not text or not isinstance(text, str):
                self.logger.warning("Empty or invalid text input")
                return ""
                
            # Convertir a minúsculas
            text = text.lower()
            
            # Eliminar caracteres especiales pero mantener guiones
            text = re.sub(r'[^\w\s\-]', ' ', text)
            
            # Eliminar números excepto cuando son parte de identificadores
            text = re.sub(r'(?<![a-zA-Z])\d+(?![a-zA-Z])', '', text)
            
            # Procesar con spaCy
            doc = self.nlp(text)
            
            # Lematizar y eliminar stopwords
            tokens = []
            for token in doc:
                if (not token.is_stop and 
                    not token.is_punct and 
                    not token.is_space and
                    token.text not in self.stopwords):
                    tokens.append(token.lemma_)
            
            # Unir tokens y eliminar espacios múltiples
            text = ' '.join(tokens)
            text = re.sub(r'\s+', ' ', text).strip()
            
            return text
            
        except Exception as e:
            self.logger.error(f"Error cleaning text: {str(e)}")
            return text