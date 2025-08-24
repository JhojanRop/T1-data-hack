"""
Extractor de entidades biomédicas usando scispaCy
Soporte para reconocimiento de entidades médicas y mapeo a MeSH
"""

import spacy
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter, defaultdict
from typing import List, Dict, Tuple, Optional
import logging

try:
    from tqdm import tqdm
except ImportError:
    # Fallback si tqdm no está disponible
    def tqdm(iterable, *args, **kwargs):
        return iterable

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BiomedicalNER:
    """
    Extractor de entidades biomédicas usando scispaCy
    """
    
    def __init__(self, model_name: str = "en_core_sci_sm"):
        """
        Inicializa el modelo de NER biomédico
        
        Args:
            model_name: Nombre del modelo scispaCy
                      Opciones:
                      - "en_core_sci_sm" (pequeño, rápido)
                      - "en_core_sci_md" (mediano)
                      - "en_core_sci_lg" (grande, mejor precisión)
                      - "en_ner_bionlp13cg_md" (específico para biología)
        """
        self.model_name = model_name
        
        try:
            logger.info(f"Cargando modelo scispaCy: {model_name}")
            self.nlp = spacy.load(model_name)
            logger.info("✅ Modelo scispaCy cargado exitosamente")
            
            # Configurar pipeline para mejor rendimiento
            if "ner" not in self.nlp.pipe_names:
                logger.warning("⚠️ NER no encontrado en el pipeline")
            
        except OSError as e:
            logger.error(f"❌ Error cargando modelo {model_name}: {e}")
            logger.info("💡 Instalar con: python -m spacy download en_core_sci_sm")
            raise e
            
        # Tipos de entidades biomédicas relevantes
        self.biomedical_labels = {
            'CHEMICAL', 'DISEASE', 'DRUG', 'GENE', 'PROTEIN', 
            'SPECIES', 'CELL_TYPE', 'CELL_LINE', 'DNA', 'RNA',
            'ANATOMY', 'TISSUE', 'ORGAN', 'SYMPTOM', 'TREATMENT'
        }
    
    def extract_entities(self, text: str) -> Dict[str, List[str]]:
        """
        Extrae entidades biomédicas de un texto
        
        Args:
            text: Texto a procesar
            
        Returns:
            Diccionario con entidades por tipo
        """
        if pd.isna(text) or text.strip() == "":
            return {label: [] for label in self.biomedical_labels}
        
        # Procesar texto
        doc = self.nlp(text[:1000000])  # Limitar longitud para evitar memoria
        
        # Agrupar entidades por tipo
        entities_by_type = defaultdict(list)
        
        for ent in doc.ents:
            entity_text = ent.text.strip().lower()
            entity_label = ent.label_
            
            # Filtrar entidades muy cortas o inválidas
            if len(entity_text) > 2 and entity_text.isalpha():
                entities_by_type[entity_label].append(entity_text)
        
        # Asegurar que todos los tipos estén presentes
        result = {label: [] for label in self.biomedical_labels}
        for label, entities in entities_by_type.items():
            if label in self.biomedical_labels:
                result[label] = list(set(entities))  # Remover duplicados
        
        return result
    
    def extract_from_corpus(self, texts: List[str], 
                          show_progress: bool = True) -> List[Dict[str, List[str]]]:
        """
        Extrae entidades de un corpus de textos
        
        Args:
            texts: Lista de textos
            show_progress: Mostrar barra de progreso
            
        Returns:
            Lista de diccionarios con entidades por texto
        """
        logger.info(f"Extrayendo entidades de {len(texts)} textos...")
        
        results = []
        iterator = tqdm(texts, desc="Procesando NER") if show_progress else texts
        
        for text in iterator:
            entities = self.extract_entities(text)
            results.append(entities)
        
        logger.info("✅ Extracción de entidades completada")
        return results
    
    def create_entity_features(self, entity_list: List[Dict[str, List[str]]]) -> pd.DataFrame:
        """
        Convierte entidades extraídas en features numéricas
        
        Args:
            entity_list: Lista de diccionarios con entidades
            
        Returns:
            DataFrame con features de entidades
        """
        features = []
        
        for entities in entity_list:
            feature_row = {}
            
            # Conteo de entidades por tipo
            for entity_type in self.biomedical_labels:
                feature_row[f'count_{entity_type.lower()}'] = len(entities.get(entity_type, []))
            
            # Features adicionales
            total_entities = sum(len(ents) for ents in entities.values())
            feature_row['total_entities'] = total_entities
            feature_row['entity_diversity'] = len([t for t in entities.values() if len(t) > 0])
            
            # Ratios
            if total_entities > 0:
                for entity_type in self.biomedical_labels:
                    count = len(entities.get(entity_type, []))
                    feature_row[f'ratio_{entity_type.lower()}'] = count / total_entities
            else:
                for entity_type in self.biomedical_labels:
                    feature_row[f'ratio_{entity_type.lower()}'] = 0.0
            
            features.append(feature_row)
        
        return pd.DataFrame(features)
    
    def get_entity_vocabulary(self, entity_list: List[Dict[str, List[str]]]) -> Dict[str, Counter]:
        """
        Construye vocabulario de entidades por tipo
        
        Args:
            entity_list: Lista de diccionarios con entidades
            
        Returns:
            Diccionario con contadores de entidades por tipo
        """
        vocabulary = {entity_type: Counter() for entity_type in self.biomedical_labels}
        
        for entities in entity_list:
            for entity_type, entity_texts in entities.items():
                if entity_type in vocabulary:
                    vocabulary[entity_type].update(entity_texts)
        
        return vocabulary
    
    def analyze_entity_distribution(self, entity_list: List[Dict[str, List[str]]]) -> pd.DataFrame:
        """
        Analiza la distribución de entidades en el corpus
        
        Args:
            entity_list: Lista de diccionarios con entidades
            
        Returns:
            DataFrame con estadísticas de entidades
        """
        vocab = self.get_entity_vocabulary(entity_list)
        
        stats = []
        for entity_type, counter in vocab.items():
            total_count = sum(counter.values())
            unique_count = len(counter)
            
            stats.append({
                'entity_type': entity_type,
                'total_occurrences': total_count,
                'unique_entities': unique_count,
                'avg_frequency': total_count / max(unique_count, 1),
                'most_common': counter.most_common(1)[0] if counter else ('', 0)
            })
        
        return pd.DataFrame(stats).sort_values('total_occurrences', ascending=False)


class MeSHMapper:
    """
    Mapeador de entidades biomédicas a términos MeSH (Medical Subject Headings)
    """
    
    def __init__(self):
        """
        Inicializa el mapeador MeSH
        Nota: Para una implementación completa, se necesitaría descargar
        el vocabulario MeSH completo desde NCBI
        """
        logger.info("Inicializando MeSH mapper...")
        
        # Diccionario simplificado de términos MeSH comunes
        # En producción, esto vendría de una base de datos MeSH completa
        self.mesh_terms = {
            # Enfermedades
            'cancer': 'D009369',
            'diabetes': 'D003920', 
            'hypertension': 'D006973',
            'infection': 'D007239',
            'inflammation': 'D007249',
            'tumor': 'D009369',
            'disease': 'D004194',
            
            # Químicos/Drogas
            'protein': 'D011506',
            'enzyme': 'D004798',
            'hormone': 'D006728',
            'antibiotic': 'D000900',
            'drug': 'D004364',
            'medication': 'D004364',
            
            # Anatomía
            'heart': 'D006321',
            'brain': 'D001921',
            'liver': 'D008099',
            'kidney': 'D007668',
            'lung': 'D008168',
            'cell': 'D002477',
            'tissue': 'D014024',
            
            # Procesos biológicos
            'metabolism': 'D008660',
            'synthesis': 'D001692',
            'regulation': 'D005786',
            'expression': 'D015870',
            'signaling': 'D015398'
        }
        
        logger.info(f"✅ MeSH mapper inicializado con {len(self.mesh_terms)} términos")
    
    def map_entity_to_mesh(self, entity: str) -> Optional[str]:
        """
        Mapea una entidad a un código MeSH
        
        Args:
            entity: Texto de la entidad
            
        Returns:
            Código MeSH o None si no se encuentra
        """
        entity_clean = entity.lower().strip()
        
        # Búsqueda exacta
        if entity_clean in self.mesh_terms:
            return self.mesh_terms[entity_clean]
        
        # Búsqueda por contención (palabras clave)
        for term, mesh_code in self.mesh_terms.items():
            if term in entity_clean or entity_clean in term:
                return mesh_code
        
        return None
    
    def create_mesh_features(self, entity_list: List[Dict[str, List[str]]]) -> pd.DataFrame:
        """
        Crea features basadas en mapeo a MeSH
        
        Args:
            entity_list: Lista de diccionarios con entidades
            
        Returns:
            DataFrame con features MeSH
        """
        features = []
        
        for entities in entity_list:
            feature_row = {}
            
            # Contadores por categoría MeSH
            mesh_categories = defaultdict(int)
            total_mapped = 0
            
            for entity_type, entity_texts in entities.items():
                for entity in entity_texts:
                    mesh_code = self.map_entity_to_mesh(entity)
                    if mesh_code:
                        total_mapped += 1
                        # Usar primera letra del código MeSH como categoría
                        category = mesh_code[0] if mesh_code else 'X'
                        mesh_categories[f'mesh_cat_{category}'] += 1
            
            # Features MeSH
            feature_row['total_mesh_mapped'] = total_mapped
            feature_row['mesh_mapping_rate'] = total_mapped / max(sum(len(ents) for ents in entities.values()), 1)
            
            # Añadir contadores por categoría
            for category, count in mesh_categories.items():
                feature_row[category] = count
            
            features.append(feature_row)
        
        return pd.DataFrame(features).fillna(0)
