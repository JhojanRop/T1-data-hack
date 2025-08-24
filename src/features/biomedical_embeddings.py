"""
Extractor de embeddings biomédicos usando sentence-transformers
Soporte para PubMedBERT, BioBERT y otros modelos biomédicos pre-entrenados
"""

import numpy as np
import pandas as pd
from typing import List, Tuple, Dict, Optional
import logging
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD

try:
    from sentence_transformers import SentenceTransformer
    import torch
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    print("⚠️ sentence-transformers no disponible. Usando fallback TF-IDF + SVD")

try:
    from tqdm import tqdm
except ImportError:
    # Fallback si tqdm no está disponible
    def tqdm(iterable, *args, **kwargs):
        return iterable

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BiomedicalEmbeddings:
    """
    Extractor de embeddings biomédicos usando sentence-transformers con fallback a TF-IDF
    """
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", device: str = None, embedding_dim: int = 384):
        """
        Inicializa el extractor de embeddings
        
        Args:
            model_name: Nombre del modelo de sentence-transformers
            device: 'cuda', 'cpu' o None (auto-detect)
            embedding_dim: Dimensión de embeddings para fallback TF-IDF
        """
        self.model_name = model_name
        self.use_sentence_transformers = SENTENCE_TRANSFORMERS_AVAILABLE
        self.embedding_dim = embedding_dim
        
        if self.use_sentence_transformers:
            try:
                self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
                logger.info(f"Inicializando modelo: {model_name}")
                logger.info(f"Dispositivo: {self.device}")
                
                self.model = SentenceTransformer(model_name, device=self.device)
                self.embedding_dim = self.model.get_sentence_embedding_dimension()
                logger.info("✅ Modelo sentence-transformers cargado exitosamente")
                
            except Exception as e:
                logger.error(f"❌ Error cargando sentence-transformers: {e}")
                self.use_sentence_transformers = False
        
        if not self.use_sentence_transformers:
            logger.info("🔄 Usando fallback TF-IDF + SVD")
            self.tfidf = TfidfVectorizer(
                max_features=5000,
                ngram_range=(1, 2),
                stop_words='english',
                sublinear_tf=True
            )
            self.svd = TruncatedSVD(n_components=min(embedding_dim, 384), random_state=42)
            self.model = None
            
        logger.info(f"Dimensión de embeddings: {self.embedding_dim}")
    
    def combine_title_abstract(self, titles: List[str], abstracts: List[str]) -> List[str]:
        """
        Combina títulos y abstracts de manera óptima para embeddings
        """
        combined = []
        for title, abstract in zip(titles, abstracts):
            if pd.isna(abstract) or abstract.strip() == "":
                text = f"Title: {title}"
            else:
                text = f"Title: {title} [SEP] Abstract: {abstract}"
            combined.append(text)
        
        return combined
    
    def _extract_tfidf_embeddings(self, texts: List[str], fit: bool = False) -> np.ndarray:
        """
        Extrae embeddings usando TF-IDF + SVD como fallback
        """
        logger.info("Extrayendo embeddings con TF-IDF + SVD...")
        
        # Limpiar textos vacíos
        texts_clean = [text if text and not pd.isna(text) else "empty text" for text in texts]
        
        if fit:
            # Entrenar TF-IDF
            tfidf_matrix = self.tfidf.fit_transform(texts_clean)
            # Entrenar SVD
            embeddings = self.svd.fit_transform(tfidf_matrix)
        else:
            # Solo transformar
            tfidf_matrix = self.tfidf.transform(texts_clean)
            embeddings = self.svd.transform(tfidf_matrix)
        
        # Normalizar L2
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1  # Evitar división por cero
        embeddings = embeddings / norms
        
        return embeddings.astype(np.float32)
    
    def extract_embeddings(self, texts: List[str], batch_size: int = 32, 
                          show_progress: bool = True) -> np.ndarray:
        """
        Extrae embeddings de una lista de textos
        """
        logger.info(f"Extrayendo embeddings de {len(texts)} textos...")
        
        if self.use_sentence_transformers:
            embeddings = self.model.encode(
                texts, 
                batch_size=batch_size,
                show_progress_bar=show_progress,
                convert_to_numpy=True,
                normalize_embeddings=True
            )
        else:
            # Usar fallback TF-IDF
            embeddings = self._extract_tfidf_embeddings(texts, fit=True)
        
        logger.info(f"✅ Embeddings extraídos: {embeddings.shape}")
        return embeddings
    
    def extract_from_dataframe(self, df: pd.DataFrame, 
                              title_col: str = 'title', 
                              abstract_col: str = 'abstract',
                              batch_size: int = 32) -> np.ndarray:
        """
        Extrae embeddings directamente de un DataFrame
        """
        titles = df[title_col].fillna("").tolist()
        abstracts = df[abstract_col].fillna("").tolist()
        
        combined_texts = self.combine_title_abstract(titles, abstracts)
        return self.extract_embeddings(combined_texts, batch_size=batch_size)
    
    def get_embedding_stats(self, embeddings: np.ndarray) -> Dict[str, float]:
        """
        Calcula estadísticas de los embeddings
        """
        return {
            'mean_norm': np.linalg.norm(embeddings, axis=1).mean(),
            'std_norm': np.linalg.norm(embeddings, axis=1).std(),
            'mean_cosine_sim': np.mean(np.dot(embeddings, embeddings.T)),
            'embedding_dim': embeddings.shape[1],
            'n_samples': embeddings.shape[0],
            'method': 'sentence-transformers' if self.use_sentence_transformers else 'tfidf-svd'
        }
    
    def save_embeddings(self, embeddings: np.ndarray, filepath: str):
        """Guarda embeddings en disco"""
        np.save(filepath, embeddings)
        logger.info(f"✅ Embeddings guardados en: {filepath}")
    
    def load_embeddings(self, filepath: str) -> np.ndarray:
        """Carga embeddings desde disco"""
        embeddings = np.load(filepath)
        logger.info(f"✅ Embeddings cargados desde: {filepath}")
        return embeddings


class EmbeddingAnalyzer:
    """
    Analizador para visualizar y entender los embeddings biomédicos
    """
    
    def __init__(self, embeddings: np.ndarray, labels: Optional[List] = None):
        """
        Args:
            embeddings: Array de embeddings (n_samples, embedding_dim)
            labels: Lista opcional de etiquetas para análisis
        """
        self.embeddings = embeddings
        self.labels = labels
        
    def plot_embedding_distribution(self, sample_size: int = 1000):
        """
        Visualiza la distribución de normas de embeddings
        """
        # Samplear si es muy grande
        if len(self.embeddings) > sample_size:
            idx = np.random.choice(len(self.embeddings), sample_size, replace=False)
            embeddings_sample = self.embeddings[idx]
        else:
            embeddings_sample = self.embeddings
            
        norms = np.linalg.norm(embeddings_sample, axis=1)
        
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 2, 1)
        plt.hist(norms, bins=50, alpha=0.7)
        plt.title('Distribución de Normas L2 de Embeddings')
        plt.xlabel('Norma L2')
        plt.ylabel('Frecuencia')
        
        plt.subplot(1, 2, 2)
        plt.boxplot(norms)
        plt.title('Boxplot de Normas L2')
        plt.ylabel('Norma L2')
        
        plt.tight_layout()
        plt.show()
        
        print(f"Estadísticas de normas:")
        print(f"  Media: {norms.mean():.4f}")
        print(f"  Std: {norms.std():.4f}")
        print(f"  Min: {norms.min():.4f}")
        print(f"  Max: {norms.max():.4f}")
    
    def compute_similarity_matrix(self, n_samples: int = 100) -> np.ndarray:
        """
        Computa matriz de similitud coseno entre muestras
        """
        if len(self.embeddings) > n_samples:
            idx = np.random.choice(len(self.embeddings), n_samples, replace=False)
            embeddings_sample = self.embeddings[idx]
        else:
            embeddings_sample = self.embeddings
            
        # Normalizar embeddings
        normalized = embeddings_sample / np.linalg.norm(embeddings_sample, axis=1, keepdims=True)
        
        # Calcular similitud coseno
        similarity_matrix = np.dot(normalized, normalized.T)
        
        return similarity_matrix
    
    def plot_similarity_heatmap(self, n_samples: int = 50):
        """
        Visualiza heatmap de similitudes
        """
        sim_matrix = self.compute_similarity_matrix(n_samples)
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(sim_matrix, cmap='viridis', center=0, 
                   xticklabels=False, yticklabels=False)
        plt.title(f'Matriz de Similitud Coseno\n({n_samples} muestras aleatorias)')
        plt.show()
        
        # Estadísticas de similitud
        upper_triangle = sim_matrix[np.triu_indices_from(sim_matrix, k=1)]
        print(f"Estadísticas de similitud coseno:")
        print(f"  Media: {upper_triangle.mean():.4f}")
        print(f"  Std: {upper_triangle.std():.4f}")
        print(f"  Min: {upper_triangle.min():.4f}")
        print(f"  Max: {upper_triangle.max():.4f}")
