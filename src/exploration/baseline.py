from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.model_selection import KFold
import numpy as np
from typing import List, Dict, Tuple
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

class BaselineModel:
    def __init__(self, max_features: int = 5000):
        """
        Inicializa el modelo baseline
        Args:
            max_features: Número máximo de características para TF-IDF
        """
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=(1, 2),  # Incluir unigramas y bigramas
            stop_words='english'
        )
        self.mlb = MultiLabelBinarizer()
        self.model = OneVsRestClassifier(LogisticRegression(C=1.0, max_iter=1000))
        
    def preprocess_text(self, titles: List[str], abstracts: List[str]) -> np.ndarray:
        """
        Preprocesa el texto combinando título y abstract
        Args:
            titles: Lista de títulos
            abstracts: Lista de abstracts
        Returns:
            Matriz TF-IDF
        """
        # Combinar título y abstract para cada documento
        combined_text = [f"{title}. {abstract}" for title, abstract in zip(titles, abstracts)]
        return self.vectorizer.fit_transform(combined_text)
    
    def create_baseline(self, titles: List[str], abstracts: List[str], labels: List[List[str]], 
                       n_splits: int = 5) -> Tuple[Dict[str, List[float]], Dict[str, np.ndarray]]:
        """
        Crea un baseline usando TF-IDF + Logistic Regression con OneVsRest
        Args:
            titles: Lista de títulos
            abstracts: Lista de abstracts
            labels: Lista de listas de etiquetas
            n_splits: Número de folds para validación cruzada
        Returns:
            Tuple con scores y matrices de confusión promedio por clase
        """
        # Preparar features
        X = self.preprocess_text(titles, abstracts)
        y = self.mlb.fit_transform(labels)
        
        # Inicializar K-Fold
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
        scores = {
            'accuracy': [],
            'f1_weighted': [],
            'f1_macro': [],
            'f1_micro': []
        }
        conf_matrices = {}
        
        # Para cada fold
        for train_idx, val_idx in kf.split(X):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            # Entrenar y predecir
            self.model.fit(X_train, y_train)
            y_pred = self.model.predict(X_val)
            
            # Calcular métricas
            scores['accuracy'].append(accuracy_score(y_val, y_pred))
            scores['f1_weighted'].append(f1_score(y_val, y_pred, average='weighted'))
            scores['f1_macro'].append(f1_score(y_val, y_pred, average='macro'))
            scores['f1_micro'].append(f1_score(y_val, y_pred, average='micro'))
            
            # Calcular matriz de confusión para cada clase
            for i, class_name in enumerate(self.mlb.classes_):
                if class_name not in conf_matrices:
                    conf_matrices[class_name] = []
                conf_matrices[class_name].append(
                    confusion_matrix(y_val[:, i], y_pred[:, i])
                )
        
        # Promediar matrices de confusión
        avg_conf_matrices = {
            class_name: np.mean(matrices, axis=0)
            for class_name, matrices in conf_matrices.items()
        }
        
        return scores, avg_conf_matrices
    
    def plot_confusion_matrices(self, conf_matrices: Dict[str, np.ndarray]):
        """
        Visualiza las matrices de confusión para cada clase
        Args:
            conf_matrices: Diccionario con matrices de confusión por clase
        """
        n_classes = len(conf_matrices)
        n_cols = 3
        n_rows = (n_classes + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5*n_rows))
        axes = axes.ravel()
        
        for i, (class_name, conf_matrix) in enumerate(conf_matrices.items()):
            # Normalizar matriz
            conf_matrix_norm = conf_matrix.astype('float') / conf_matrix.sum(axis=1)[:, np.newaxis]
            
            # Plotear
            sns.heatmap(conf_matrix_norm, annot=True, fmt='.2f', cmap='Blues', ax=axes[i])
            axes[i].set_title(f'Matriz de Confusión - {class_name}')
            axes[i].set_xlabel('Predicho')
            axes[i].set_ylabel('Real')
        
        # Ocultar subplots vacíos
        for j in range(i + 1, len(axes)):
            axes[j].set_visible(False)
        
        plt.tight_layout()
        plt.show()
