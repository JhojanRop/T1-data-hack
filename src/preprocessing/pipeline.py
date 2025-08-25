from typing import List, Dict, Union
import pandas as pd
from .cleaner import TextCleaner
from .normalizer import SemanticNormalizer
import spacy
import logging
from functools import lru_cache

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BiomedicalPreprocessingPipeline:
    """
    Pipeline de preprocesamiento para texto biomédico.
    Implementa limpieza, normalización y enriquecimiento semántico.
    """
    def __init__(
        self,
        model_name: str = "en_core_sci_lg",
        use_abbreviations: bool = True,
        use_mesh: bool = True,
        use_external_dicts: bool = True,
        cache_size: int = 1024,
        batch_size: int = 32
    ):
        """
        Inicializa el pipeline de preprocesamiento
        
        Args:
            model_name: Nombre del modelo de spaCy a usar
            use_abbreviations: Si resolver abreviaturas
            use_mesh: Si usar MeSH
            use_external_dicts: Si usar diccionarios externos
            cache_size: Tamaño de cache para memorización
            batch_size: Tamaño de batch para procesamiento
        """
        # Inicializar componentes base
        self.cleaner = TextCleaner()
        self.normalizer = SemanticNormalizer(
            model_name=model_name,
            use_abbreviations=use_abbreviations,
            use_mesh=use_mesh,
            use_external_dicts=use_external_dicts
        )
        self.batch_size = batch_size
        self.cache_size = cache_size
        
        # Cargar modelo spaCy para análisis adicional
        try:
            if not spacy.util.is_package(model_name):
                logger.info(f"Descargando modelo {model_name}...")
                spacy.cli.download(model_name)
            self.nlp = spacy.load(model_name)
            logger.info(f"✓ Modelo {model_name} cargado")
            
            # Configurar pipeline de spaCy
            if "sentencizer" not in self.nlp.pipe_names:
                self.nlp.add_pipe("sentencizer")
                
        except Exception as e:
            logger.warning(f"Error cargando {model_name}, usando fallback: {e}")
            self.nlp = spacy.load("en_core_web_sm")
            
        # Cache para stopwords
        self.biomedical_stopwords = self._load_biomedical_stopwords()
        
    def process_document(self, title: str, abstract: str) -> Dict[str, Union[str, List[str], Dict]]:
        """
        Procesa un documento biomédico completo
        
        Args:
            title: Título del documento
            abstract: Abstract del documento
            
        Returns:
            Dict con texto normalizado y features extraídas
        """
        if not title and not abstract:
            logger.warning("Documento vacío recibido")
            return {
                "normalized_text": {"title": "", "abstract": ""},
                "features": {},
                "entities": []
            }
            
        try:
            # 1. Limpieza básica
            clean_title = self.cleaner.clean_text(title) if title else ""
            clean_abstract = self.cleaner.clean_text(abstract) if abstract else ""
            
            # 2. Normalización semántica
            normalized = self.normalizer.normalize_document(clean_title, clean_abstract)
            
            # 3. Análisis spaCy
            doc = self.nlp(normalized["normalized_title"] + " " + normalized["normalized_abstract"])
            
            # 4. Extracción de features
            features = {
                "text_features": self._extract_text_features(
                    normalized["normalized_title"],
                    normalized["normalized_abstract"],
                    doc
                ),
                "linguistic_features": self._extract_linguistic_features(doc),
                "mesh_features": self._extract_mesh_features(normalized["mesh_terms"]),
                "external_features": normalized["external_terms"]
            }
            
            return {
                "normalized_text": normalized,
                "features": features,
                "entities": normalized["mesh_terms"],
                "linguistic_info": {
                    "sentences": [sent.text for sent in doc.sents],
                    "key_phrases": self._extract_key_phrases(doc)
                }
            }
            
        except Exception as e:
            logger.error(f"Error procesando documento: {e}")
            return {
                "normalized_text": {"title": title, "abstract": abstract},
                "features": {},
                "entities": []
            }
    
    def process_batch(self, documents: List[Dict[str, str]]) -> List[Dict]:
        """
        Procesa un batch de documentos
        
        Args:
            documents: Lista de documentos con title y abstract
            
        Returns:
            Lista de documentos procesados
        """
        results = []
        for i in range(0, len(documents), self.batch_size):
            batch = documents[i:i + self.batch_size]
            
            for doc in batch:
                try:
                    processed = self.process_document(
                        doc.get("title", ""),
                        doc.get("abstract", "")
                    )
                    results.append(processed)
                except Exception as e:
                    logger.error(f"Error en batch {i}: {e}")
                    results.append(None)
                    
        return results
    
    def _extract_text_features(
        self,
        normalized_title: str,
        normalized_abstract: str,
        doc: spacy.tokens.Doc
    ) -> Dict[str, Union[int, bool, List[str]]]:
        """
        Extrae features básicas del texto y características lingüísticas
        
        Args:
            normalized_title: Título normalizado
            normalized_abstract: Abstract normalizado
            doc: Documento spaCy procesado
            
        Returns:
            Dict con features del texto
        """
        # Features básicas
        features = {
            "title_length": len(normalized_title.split()),
            "abstract_length": len(normalized_abstract.split()),
            "has_mesh_terms": "[MeSH:" in (normalized_title + normalized_abstract),
            "sentence_count": len(list(doc.sents)),
            "avg_sentence_length": sum(len(sent.text.split()) for sent in doc.sents) / len(list(doc.sents)) if doc.text else 0
        }
        
        # Extraer n-gramas significativos (excluyendo stopwords)
        bigrams = []
        trigrams = []
        for i in range(len(doc)-1):
            if not (doc[i].is_stop or doc[i+1].is_stop):
                bigram = f"{doc[i].lemma_}_{doc[i+1].lemma_}"
                bigrams.append(bigram)
        
        for i in range(len(doc)-2):
            if not (doc[i].is_stop or doc[i+1].is_stop or doc[i+2].is_stop):
                trigram = f"{doc[i].lemma_}_{doc[i+1].lemma_}_{doc[i+2].lemma_}"
                trigrams.append(trigram)
                
        features.update({
            "bigrams": bigrams[:10],  # Top 10 bigrams
            "trigrams": trigrams[:10]  # Top 10 trigrams
        })
        
        return features
    
    def _extract_linguistic_features(
        self,
        doc: spacy.tokens.Doc
    ) -> Dict[str, Union[List[str], Dict[str, int]]]:
        """
        Extrae características lingüísticas avanzadas
        
        Args:
            doc: Documento spaCy procesado
            
        Returns:
            Dict con features lingüísticas
        """
        # Conteo de tipos de entidades
        ent_counts = {}
        for ent in doc.ents:
            ent_counts[ent.label_] = ent_counts.get(ent.label_, 0) + 1
            
        # Análisis morfosintáctico
        pos_counts = {}
        for token in doc:
            if not token.is_stop and not token.is_punct:
                pos_counts[token.pos_] = pos_counts.get(token.pos_, 0) + 1
                
        return {
            "entity_counts": ent_counts,
            "pos_counts": pos_counts,
            "root_verbs": [token.lemma_ for token in doc if token.dep_ == "ROOT" and token.pos_ == "VERB"],
            "noun_chunks": [chunk.text for chunk in doc.noun_chunks]
        }
    
    def _extract_key_phrases(
        self,
        doc: spacy.tokens.Doc,
        top_k: int = 10
    ) -> List[str]:
        """
        Extrae frases clave usando patrones sintácticos
        
        Args:
            doc: Documento spaCy procesado
            top_k: Número de frases a extraer
            
        Returns:
            Lista de frases clave
        """
        phrases = []
        
        # Patrones de extracción (ejemplo: NOUN + VERB, ADJ + NOUN)
        for chunk in doc.noun_chunks:
            if not any(token.is_stop for token in chunk):
                phrases.append(chunk.text)
                
        # Verbos significativos con sus argumentos
        for token in doc:
            if token.pos_ == "VERB" and not token.is_stop:
                phrase = []
                for child in token.children:
                    if child.dep_ in ["dobj", "pobj"] and not child.is_stop:
                        phrase.extend([token.text, child.text])
                if phrase:
                    phrases.append(" ".join(phrase))
                    
        return phrases[:top_k]
    
    def _extract_mesh_features(
        self,
        mesh_terms: List[Dict[str, str]]
    ) -> Dict[str, Union[int, List[str]]]:
        """
        Extrae features basadas en términos MeSH
        
        Args:
            mesh_terms: Lista de términos MeSH encontrados
            
        Returns:
            Dict con features MeSH
        """
        return {
            "mesh_term_count": len(mesh_terms),
            "unique_tree_numbers": len(set(
                tn for term in mesh_terms 
                for tn in term.get("tree_numbers", [])
            )),
            "terms": [term["preferred"] for term in mesh_terms],
            "tree_numbers": list(set(
                tn for term in mesh_terms 
                for tn in term.get("tree_numbers", [])
            )),
            "hierarchies": self._extract_mesh_hierarchies(mesh_terms)
        }
        
    def _extract_mesh_hierarchies(
        self,
        mesh_terms: List[Dict[str, str]]
    ) -> Dict[str, List[str]]:
        """
        Extrae jerarquías de términos MeSH
        
        Args:
            mesh_terms: Lista de términos MeSH
            
        Returns:
            Dict con jerarquías por categoría principal
        """
        hierarchies = {}
        for term in mesh_terms:
            for tree_num in term.get("tree_numbers", []):
                parts = tree_num.split(".")
                if parts[0] not in hierarchies:
                    hierarchies[parts[0]] = []
                hierarchies[parts[0]].append(term["preferred"])
        return hierarchies
        
    @lru_cache(maxsize=1024)
    def _load_biomedical_stopwords(self) -> set:
        """Carga stopwords específicas del dominio biomédico"""
        base_stopwords = set(self.nlp.Defaults.stop_words)
        biomedical_specific = {
            'study', 'studies', 'patient', 'patients', 'result', 'results',
            'method', 'methods', 'conclusion', 'conclusions', 'background',
            'objective', 'objectives', 'aim', 'aims', 'purpose',
            'finding', 'findings', 'data', 'analysis', 'research',
            'trial', 'group', 'groups', 'sample', 'samples'
        }
        return base_stopwords.union(biomedical_specific)
    
    def preprocess_text(self, text: str) -> Dict[str, Union[str, List[str]]]:
        """
        Preprocesa un único texto biomédico
        
        Args:
            text: Texto a preprocesar
            
        Returns:
            Dict con texto limpio y entidades extraídas
        """
        # Validación básica
        if not text or not isinstance(text, str):
            logger.warning("Texto inválido recibido")
            return {"clean_text": "", "entities": [], "normalized_text": ""}
            
        try:
            # Limpieza básica
            clean_text = self.cleaner.clean_text(text)
            
            # Normalización y extracción de entidades
            result = self.normalizer.process_text(clean_text)
            normalized_text = result["normalized_text"]
            entities = result.get("entities", [])
            
            # Procesamiento con spaCy
            doc = self.nlp(normalized_text)
            
            # Extracción de entidades
            entities = [
                {
                    "text": ent.text,
                    "label": ent.label_,
                    "start": ent.start_char,
                    "end": ent.end_char
                }
                for ent in doc.ents
            ]
            
            # Tokens sin stopwords
            tokens = [
                token.lemma_.lower()
                for token in doc
                if not token.is_stop 
                and not token.lemma_.lower() not in self.biomedical_stopwords
                and token.is_alpha
            ]
            
            return {
                "clean_text": normalized_text,
                "entities": entities,
                "tokens": tokens,
                "doc": doc  # Para procesamiento adicional si es necesario
            }
            
        except Exception as e:
            logger.error(f"Error en preprocesamiento: {str(e)}")
            return {"clean_text": text, "entities": [], "tokens": []}
    
    def preprocess_batch(
        self,
        texts: List[str],
        n_jobs: int = -1
    ) -> List[Dict[str, Union[str, List[str]]]]:
        """
        Preprocesa un lote de textos en paralelo
        
        Args:
            texts: Lista de textos a preprocesar
            n_jobs: Número de jobs para procesamiento paralelo
            
        Returns:
            Lista de diccionarios con textos procesados
        """
        from joblib import Parallel, delayed
        
        # Procesar en batches
        results = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]
            
            # Procesamiento paralelo del batch
            batch_results = Parallel(n_jobs=n_jobs)(
                delayed(self.preprocess_text)(text)
                for text in batch
            )
            
            results.extend(batch_results)
            
        return results
    
    def preprocess_dataframe(
        self,
        df: pd.DataFrame,
        text_columns: List[str],
        n_jobs: int = -1
    ) -> pd.DataFrame:
        """
        Preprocesa columnas de texto en un DataFrame
        
        Args:
            df: DataFrame con textos a procesar
            text_columns: Lista de columnas a procesar
            n_jobs: Número de jobs para procesamiento paralelo
            
        Returns:
            DataFrame con columnas procesadas
        """
        df_processed = df.copy()
        
        for col in text_columns:
            if col not in df.columns:
                logger.warning(f"Columna {col} no encontrada en DataFrame")
                continue
                
            # Procesar textos de la columna
            processed_texts = self.preprocess_batch(
                df[col].tolist(),
                n_jobs=n_jobs
            )
            
            # Agregar columnas procesadas
            df_processed[f"{col}_clean"] = [
                item["clean_text"] for item in processed_texts
            ]
            df_processed[f"{col}_entities"] = [
                item["entities"] for item in processed_texts
            ]
            df_processed[f"{col}_tokens"] = [
                item["tokens"] for item in processed_texts
            ]
            
        return df_processed
