from sentence_transformers import SentenceTransformer
import torch
import numpy as np
from typing import List

class BiomedicalEmbeddings:
    def __init__(self, model_name='pritamdeka/S-PubMedBert-MS-MARCO'):
        self.model = SentenceTransformer(model_name)
    
    def get_embeddings(self, texts: List[str]) -> np.ndarray:
        """Genera embeddings para una lista de textos"""
        embeddings = self.model.encode(
            texts,
            batch_size=32,
            show_progress_bar=True,
            normalize_embeddings=True
        )
        return embeddings
