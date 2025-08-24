import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, Tuple, List
from collections import Counter

class ExploratoryAnalysis:
    def __init__(self, data_path: str):
        """
        Inicializa el análisis exploratorio
        Args:
            data_path: Ruta al archivo CSV con los datos
        """
        self.df = pd.read_csv(data_path)
        # Convertir grupos a lista si están en formato string
        if isinstance(self.df['group'].iloc[0], str):
            self.df['group'] = self.df['group'].apply(lambda x: x.split(','))
        
    def analyze_class_distribution(self) -> Dict:
        """
        Analiza la distribución de clases considerando multilabel
        Returns:
            Dict con conteo de cada clase
        """
        # Flatten all groups into a single list
        all_labels = [label for labels in self.df['group'] for label in labels]
        distribution = Counter(all_labels)
        
        plt.figure(figsize=(12, 6))
        sns.barplot(x=list(distribution.keys()), y=list(distribution.values()))
        plt.title('Distribución de Clases (Multilabel)')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        return distribution
    
    def analyze_text_length(self) -> Dict[str, pd.Series]:
        """
        Analiza longitud de títulos y abstracts
        Returns:
            Dict con estadísticas descriptivas de longitudes
        """
        title_lengths = self.df['title'].str.len()
        abstract_lengths = self.df['abstract'].str.len()
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        
        # Título
        sns.histplot(data=title_lengths, ax=ax1, bins=30)
        ax1.set_title('Distribución de longitud de títulos')
        ax1.axvline(x=title_lengths.mean(), color='r', linestyle='--', label='Media')
        ax1.axvline(x=title_lengths.median(), color='g', linestyle='--', label='Mediana')
        ax1.legend()
        
        # Abstract
        sns.histplot(data=abstract_lengths, ax=ax2, bins=30)
        ax2.set_title('Distribución de longitud de abstracts')
        ax2.axvline(x=abstract_lengths.mean(), color='r', linestyle='--', label='Media')
        ax2.axvline(x=abstract_lengths.median(), color='g', linestyle='--', label='Mediana')
        ax2.legend()
        
        plt.tight_layout()
        
        return {
            'title_stats': title_lengths.describe(),
            'abstract_stats': abstract_lengths.describe()
        }
    
    def check_duplicates(self) -> Dict:
        """
        Verifica duplicados en el dataset
        Returns:
            Dict con información sobre duplicados
        """
        # Verificar duplicados exactos
        exact_duplicates = self.df.duplicated().sum()
        
        # Verificar duplicados en título
        title_duplicates = self.df['title'].duplicated().sum()
        
        # Verificar duplicados en abstract
        abstract_duplicates = self.df['abstract'].duplicated().sum()
        
        return {
            'exact_duplicates': {
                'count': exact_duplicates,
                'percentage': (exact_duplicates / len(self.df)) * 100
            },
            'title_duplicates': {
                'count': title_duplicates,
                'percentage': (title_duplicates / len(self.df)) * 100
            },
            'abstract_duplicates': {
                'count': abstract_duplicates,
                'percentage': (abstract_duplicates / len(self.df)) * 100
            }
        }
    
    def analyze_label_cardinality(self) -> Dict:
        """
        Analiza la cardinalidad de etiquetas (número de etiquetas por instancia)
        Returns:
            Dict con estadísticas de cardinalidad
        """
        label_counts = self.df['group'].apply(len)
        
        plt.figure(figsize=(10, 6))
        sns.countplot(x=label_counts)
        plt.title('Distribución de número de etiquetas por artículo')
        plt.xlabel('Número de etiquetas')
        plt.ylabel('Número de artículos')
        
        return {
            'stats': label_counts.describe().to_dict(),
            'distribution': label_counts.value_counts().to_dict()
        }
    
    def analyze_label_co_occurrence(self) -> pd.DataFrame:
        """
        Analiza la co-ocurrencia de etiquetas
        Returns:
            DataFrame con matriz de co-ocurrencia
        """
        # Obtener todas las etiquetas únicas
        all_labels = set([label for labels in self.df['group'] for label in labels])
        
        # Crear matriz de co-ocurrencia
        co_matrix = pd.DataFrame(0, index=all_labels, columns=all_labels)
        
        # Llenar matriz
        for labels in self.df['group']:
            for l1 in labels:
                for l2 in labels:
                    if l1 != l2:
                        co_matrix.loc[l1, l2] += 1
        
        # Visualizar
        plt.figure(figsize=(12, 10))
        sns.heatmap(co_matrix, annot=True, cmap='YlOrRd')
        plt.title('Matriz de co-ocurrencia de etiquetas')
        plt.tight_layout()
        
        return co_matrix
