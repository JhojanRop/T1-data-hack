from typing import List, Dict
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    confusion_matrix,
    roc_auc_score,
    precision_recall_curve,
    average_precision_score
)
import seaborn as sns
import matplotlib.pyplot as plt
import shap
import lime.lime_text

class ModelEvaluator:
    def __init__(self, classifier):
        self.classifier = classifier
        
    def calculate_metrics(self, y_true: List[List[str]], y_pred: List[List[str]], y_proba: np.ndarray) -> Dict:
        """Calcula todas las métricas relevantes"""
        # Convertir a formato binario
        y_true_bin = self.classifier.mlb.transform(y_true)
        y_pred_bin = self.classifier.mlb.transform(y_pred)
        
        metrics = {
            'accuracy': accuracy_score(y_true_bin, y_pred_bin),
            'f1_weighted': f1_score(y_true_bin, y_pred_bin, average='weighted'),
            'f1_micro': f1_score(y_true_bin, y_pred_bin, average='micro'),
            'f1_macro': f1_score(y_true_bin, y_pred_bin, average='macro'),
            'roc_auc': roc_auc_score(y_true_bin, y_proba, average='weighted'),
        }
        
        return metrics
    
    def plot_confusion_matrix(self, y_true: List[List[str]], y_pred: List[List[str]]):
        """Genera y visualiza la matriz de confusión"""
        # Convertir a formato binario
        y_true_bin = self.classifier.mlb.transform(y_true)
        y_pred_bin = self.classifier.mlb.transform(y_pred)
        
        # Calcular matriz de confusión
        cm = confusion_matrix(y_true_bin.argmax(axis=1), y_pred_bin.argmax(axis=1))
        
        # Normalizar
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        # Visualizar
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues')
        plt.title('Matriz de Confusión Normalizada')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        
    def explain_prediction(self, title: str, abstract: str):
        """Genera explicaciones usando SHAP y LIME"""
        # SHAP
        explainer = shap.Explainer(self.classifier.model)
        features = self.classifier.get_features([title], [abstract])
        shap_values = explainer(features)
        
        # LIME
        explainer = lime.lime_text.LimeTextExplainer(class_names=self.classifier.mlb.classes_)
        exp = explainer.explain_instance(
            title + " " + abstract,
            self.classifier.model.predict_proba,
            num_features=10
        )
        
        return {
            'shap_values': shap_values,
            'lime_explanation': exp
        }
    
    def analyze_errors(self, y_true: List[List[str]], y_pred: List[List[str]], texts: List[str]):
        """Analiza los errores de clasificación"""
        y_true_bin = self.classifier.mlb.transform(y_true)
        y_pred_bin = self.classifier.mlb.transform(y_pred)
        
        errors = []
        for i, (true, pred) in enumerate(zip(y_true_bin, y_pred_bin)):
            if not np.array_equal(true, pred):
                true_labels = self.classifier.mlb.inverse_transform([true])[0]
                pred_labels = self.classifier.mlb.inverse_transform([pred])[0]
                errors.append({
                    'text': texts[i],
                    'true_labels': true_labels,
                    'predicted_labels': pred_labels
                })
        
        return errors
