from typing import Dict, List, Text
import pandas as pd
from scispacy.linking import EntityLinker
import spacy

class SemanticNormalizer:
    def __init__(self):
        self.nlp = spacy.load("en_core_sci_lg")
        # Agregar entity linker para MeSH
        self.nlp.add_pipe("scispacy_linker", 
                         config={"resolve_abbreviations": True,
                                "linker_name": "mesh"})
        self.linker = self.nlp.get_pipe("scispacy_linker")

    def normalize_entities(self, text: str) -> str:
        """Normaliza entidades biomédicas usando MeSH"""
        doc = self.nlp(text)
        normalized_text = text
        
        for ent in doc.ents:
            mesh_links = self.linker.kb.cui_to_entity
            if len(ent._.kb_ents) > 0:
                cui = ent._.kb_ents[0][0]  # Mejor match
                if cui in mesh_links:
                    mesh_term = f"{mesh_links[cui].canonical_name} [MeSH:{cui}]"
                    normalized_text = normalized_text.replace(ent.text, mesh_term)
        
        return normalized_text