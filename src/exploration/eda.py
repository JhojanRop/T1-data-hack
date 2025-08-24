import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from typing import Dict, Tuple

class ExploratoryAnalysis:
    def __init__(self, data_path: str):
        self.df = pd.read_csv(data_path)
        
    def analyze_class_distribution(self) -> Dict:
        """Analiza la distribución de clases"""
        distribution = self.df['group'].value_counts()
        plt.figure(figsize=(10, 6))
        sns.barplot(x=distribution.index, y=distribution.values)
        plt.title('Distribución de Clases')
        plt.xticks(rotation=45)
        return distribution.to_dict()
    
    def analyze_text_length(self) -> Tuple[pd.Series, pd.Series]:
        """Analiza longitud de títulos y abstracts"""
        title_lengths = self.df['title'].str.len()
        abstract_lengths = self.df['abstract'].str.len()
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        sns.histplot(data=title_lengths, ax=ax1, bins=30)
        ax1.set_title('Distribución de longitud de títulos')
        sns.histplot(data=abstract_lengths, ax=ax2, bins=30)
        ax2.set_title('Distribución de longitud de abstracts')
        
        return title_lengths.describe(), abstract_lengths.describe()
    
    def check_duplicates(self):
        """Verifica duplicados en el dataset"""
        duplicates = self.df.duplicated().sum()
        return {
            'total_duplicates': duplicates,
            'percentage': (duplicates / len(self.df)) * 100
        }
    
    def analyze_class_balance(self):
        """Analiza el balance de clases"""
        class_balance = self.df['group'].value_counts(normalize=True)
        plt.figure(figsize=(10, 6))
        sns.barplot(x=class_balance.index, y=class_balance.values)
        plt.title('Balance de Clases (%)')
        plt.xticks(rotation=45)
        return class_balance
