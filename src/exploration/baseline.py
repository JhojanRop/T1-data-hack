from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold
import numpy as np

class BaselineModel:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(max_features=5000)
        self.model = LogisticRegression(multi_class='ovr')
        
    def create_baseline(self, X_text: list, y: list, n_splits=5):
        """
        Crea un baseline usando TF-IDF + Logistic Regression
        """
        skf = StratifiedKFold(n_splits=n_splits)
        scores = []
        
        for train_idx, val_idx in skf.split(X_text, y):
            # Split data
            X_train_text = [X_text[i] for i in train_idx]
            X_val_text = [X_text[i] for i in val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            # Transform text
            X_train = self.vectorizer.fit_transform(X_train_text)
            X_val = self.vectorizer.transform(X_val_text)
            
            # Train and evaluate
            self.model.fit(X_train, y_train)
            y_pred = self.model.predict(X_val)
            
            # Calculate metrics
            scores.append({
                'accuracy': accuracy_score(y_val, y_pred),
                'f1_weighted': f1_score(y_val, y_pred, average='weighted')
            })
            
        return scores
