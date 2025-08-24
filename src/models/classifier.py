from typing import List, Tuple, Dict
import numpy as np
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import MultiLabelBinarizer
from lightgbm import LGBMClassifier
from src.preprocessing.cleaner import TextCleaner
from src.preprocessing.normalizer import SemanticNormalizer
from src.features.embeddings import BiomedicalEmbeddings

class BiomedicalClassifier:
    def __init__(self):
        self.text_cleaner = TextCleaner()
        self.normalizer = SemanticNormalizer()
        self.embeddings = BiomedicalEmbeddings()
        self.mlb = MultiLabelBinarizer()
        self.model = OneVsRestClassifier(LGBMClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=7
        ))
        
    def preprocess_text(self, texts: List[str]) -> List[str]:
        """Preprocesa una lista de textos"""
        processed_texts = []
        for text in texts:
            # Limpieza básica
            clean_text = self.text_cleaner.clean_text(text)
            # Eliminar stopwords
            clean_text = self.text_cleaner.remove_stopwords(clean_text)
            # Lematización
            clean_text = self.text_cleaner.lemmatize(clean_text)
            # Normalización semántica
            clean_text = self.normalizer.normalize_entities(clean_text)
            processed_texts.append(clean_text)
        return processed_texts
    
    def get_features(self, titles: List[str], abstracts: List[str]) -> np.ndarray:
        """Genera features combinando título y abstract"""
        # Procesar textos
        processed_titles = self.preprocess_text(titles)
        processed_abstracts = self.preprocess_text(abstracts)
        
        # Obtener embeddings
        title_embeddings = self.embeddings.get_embeddings(processed_titles)
        abstract_embeddings = self.embeddings.get_embeddings(processed_abstracts)
        
        # Combinar features
        combined_features = np.concatenate([
            title_embeddings,
            abstract_embeddings
        ], axis=1)
        
        return combined_features
    
    def fit(self, titles: List[str], abstracts: List[str], labels: List[List[str]]):
        """Entrena el modelo"""
        # Preparar features
        X = self.get_features(titles, abstracts)
        # Preparar labels
        y = self.mlb.fit_transform(labels)
        # Entrenar modelo
        self.model.fit(X, y)
        
    def predict(self, titles: List[str], abstracts: List[str]) -> Tuple[List[List[str]], np.ndarray]:
        """Realiza predicciones"""
        # Preparar features
        X = self.get_features(titles, abstracts)
        # Predecir
        y_pred = self.model.predict(X)
        y_proba = self.model.predict_proba(X)
        # Convertir predicciones a labels
        predictions = self.mlb.inverse_transform(y_pred)
        return predictions, y_proba
