import json
from pathlib import Path
import logging
from typing import Dict, Optional
import spacy
from scispacy.linking import EntityLinker

class SemanticNormalizer:
    def __init__(self):
        """Inicializa el normalizador con modelo scispaCy y datos MeSH"""
        self.logger = logging.getLogger(__name__)
        
        # Cargar modelo scispaCy
        try:
            self.nlp = spacy.load("en_core_sci_lg")
            # Agregar entity linker para MeSH
            self.nlp.add_pipe("scispacy_linker", 
                            config={"resolve_abbreviations": True,
                                  "linker_name": "mesh"})
            self.linker = self.nlp.get_pipe("scispacy_linker")
            self.logger.info("Loaded scispaCy model and MeSH linker successfully")
        except OSError as e:
            self.logger.error(f"Failed to load model: {str(e)}")
            raise
            
        # Cargar términos MeSH adicionales
        self.mesh_terms = self._load_mesh_data()
        self.mesh_synonyms = self._load_mesh_synonyms()

    def _load_mesh_data(self) -> Dict:
        """Carga términos MeSH del archivo JSON"""
        try:
            mesh_path = Path('data/processed/mesh/mesh_terms.json')
            with open(mesh_path, 'r', encoding='utf-8') as f:
                terms = json.load(f)
            self.logger.info(f"Loaded {len(terms)} MeSH terms")
            return terms
        except Exception as e:
            self.logger.warning(f"Could not load MeSH terms: {str(e)}")
            return {}

    def _load_mesh_synonyms(self) -> Dict:
        """Carga sinónimos MeSH del archivo JSON"""
        try:
            synonyms_path = Path('data/processed/mesh/mesh_synonyms.json')
            with open(synonyms_path, 'r', encoding='utf-8') as f:
                synonyms = json.load(f)
            self.logger.info(f"Loaded {len(synonyms)} MeSH synonyms")
            return synonyms
        except Exception as e:
            self.logger.warning(f"Could not load MeSH synonyms: {str(e)}")
            return {}

    def _get_mesh_id(self, term: str) -> Optional[str]:
        """Obtiene ID MeSH de un término, revisando términos directos y sinónimos"""
        term_lower = term.lower()
        
        # Revisar términos directos
        if term_lower in self.mesh_terms:
            return self.mesh_terms[term_lower]
            
        # Revisar sinónimos
        if term_lower in self.mesh_synonyms:
            return self.mesh_synonyms[term_lower]
            
        return None

    def normalize(self, text: str) -> str:
        """
        Aplica normalización semántica usando scispaCy y MeSH
        
        Args:
            text (str): Texto a normalizar
            
        Returns:
            str: Texto normalizado con anotaciones MeSH
        """
        if not text:
            return ""

        try:
            # Procesar con scispaCy
            doc = self.nlp(text)
            normalized_text = text
            
            # Normalizar entidades
            for ent in doc.ents:
                # Intentar primero con entity linker
                if len(ent._.kb_ents) > 0:
                    cui = ent._.kb_ents[0][0]  # Mejor match
                    mesh_term = f"{ent.text} [MeSH:{cui}]"
                    normalized_text = normalized_text.replace(ent.text, mesh_term)
                else:
                    # Intentar con términos cargados localmente
                    mesh_id = self._get_mesh_id(ent.text)
                    if mesh_id:
                        mesh_term = f"{ent.text} [MeSH:{mesh_id}]"
                        normalized_text = normalized_text.replace(ent.text, mesh_term)

            return normalized_text

        except Exception as e:
            self.logger.error(f"Error normalizing text: {str(e)}")
            return text