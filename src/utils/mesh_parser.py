from typing import Dict
from pathlib import Path
import logging
import json

logger = logging.getLogger(__name__)

class MeSHParser:
    """Parser para archivos MeSH en formato ASCII"""
    
    def __init__(self, mesh_file: str):
        self.mesh_file = Path(mesh_file)
        self.terms: Dict[str, Dict] = {}
        self.synonyms: Dict[str, str] = {}
    
    def _clean_entry(self, entry: str) -> str:
        """Limpia una entrada MeSH"""
        # Si hay pipe, tomar solo la parte antes del pipe
        if '|' in entry:
            entry = entry.split('|')[0]
        return entry.strip()
    
    def parse(self) -> None:
        """Parsea el archivo MeSH ASCII"""
        logger.info(f"Parseando archivo MeSH: {self.mesh_file}")
        
        current_record = None
        records_processed = 0
        
        try:
            # Verificar que el archivo existe y tiene contenido
            if not self.mesh_file.exists():
                raise FileNotFoundError(f"El archivo {self.mesh_file} no existe")
            
            file_size = self.mesh_file.stat().st_size
            logger.info(f"Tamaño del archivo MeSH: {file_size/1024:.2f} KB")
            
            # Leer todo el contenido del archivo
            with open(self.mesh_file, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            
            # Dividir por registros
            records = content.split('*NEWRECORD')
            logger.info(f"Found {len(records)} potential records")
            
            # Procesar cada registro
            for record in records[1:]:  # Skip first empty split
                lines = record.strip().split('\n')
                current_record = {
                    'terms': set(),
                    'tree_numbers': set(),
                    'entry_terms': set()
                }
                
                # Procesar líneas del registro
                for line in lines:
                    line = line.strip()
                    if not line or line == '*ENDRECORD':
                        continue
                    
                    if line.startswith('UI = '):
                        current_record['ui'] = line[5:].strip()
                        if records_processed < 2:
                            logger.info(f"Found MeSH ID: {current_record['ui']}")
                    
                    elif line.startswith('MH = '):
                        current_record['mh'] = self._clean_entry(line[5:])
                        if records_processed < 2:
                            logger.info(f"Found Main Heading: {current_record['mh']}")
                    
                    elif line.startswith('ENTRY = '):
                        entry_term = self._clean_entry(line[8:])
                        if entry_term:
                            current_record['entry_terms'].add(entry_term)
                            if records_processed < 2:
                                logger.info(f"Added Entry Term: {entry_term}")
                    
                    elif line.startswith('MN = '):
                        tree_number = line[5:].strip()
                        if tree_number:
                            current_record['tree_numbers'].add(tree_number)
                            if records_processed < 2:
                                logger.info(f"Added Tree Number: {tree_number}")
                
                # Procesar registro completo si tiene ID y término principal
                if 'ui' in current_record and 'mh' in current_record:
                    mesh_id = current_record['ui']
                    
                    # Agregar término principal a términos
                    current_record['terms'].add(current_record['mh'])
                    
                    # Unir con entry terms
                    current_record['terms'].update(current_record['entry_terms'])
                    
                    # Debug para los primeros registros
                    if records_processed < 2:
                        logger.info(f"Processing record {records_processed + 1}:")
                        logger.info(f"  MeSH ID: {mesh_id}")
                        logger.info(f"  Main Term: {current_record['mh']}")
                        logger.info(f"  All Terms: {current_record['terms']}")
                        logger.info(f"  Tree Numbers: {current_record['tree_numbers']}")
                    
                    self.terms[mesh_id] = {
                        'preferred': current_record['mh'],
                        'terms': list(current_record['terms']),
                        'tree_numbers': list(current_record['tree_numbers'])
                    }
                    
                    # Mapear sinónimos al ID
                    for term in current_record['terms']:
                        self.synonyms[term.lower()] = mesh_id
                    
                    records_processed += 1
                    
                    if records_processed % 1000 == 0:
                        logger.info(f"Processed {records_processed} records...")
            
            logger.info(f"Finished processing {records_processed} records")
            logger.info(f"Found {len(self.terms)} unique terms")
            logger.info(f"Generated {len(self.synonyms)} synonym mappings")
            
        except Exception as e:
            logger.error(f"Error parseando MeSH: {e}")
            raise
    
    def save_processed_mesh(self, output_dir: str) -> None:
        """Guarda los términos procesados en formato JSON"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Convert sets to lists for JSON serialization
        serializable_terms = {}
        for term_id, term_data in self.terms.items():
            serializable_terms[term_id] = {
                'preferred': term_data['preferred'],
                'terms': list(term_data['terms']) if isinstance(term_data['terms'], set) else term_data['terms'],
                'tree_numbers': list(term_data['tree_numbers']) if isinstance(term_data['tree_numbers'], set) else term_data['tree_numbers']
            }
        
        # Guardar términos principales
        terms_file = output_path / 'mesh_terms.json'
        with open(terms_file, 'w', encoding='utf-8') as f:
            json.dump(serializable_terms, f, ensure_ascii=False, indent=2)
            
        # Guardar mapeo de sinónimos
        synonyms_file = output_path / 'mesh_synonyms.json'
        with open(synonyms_file, 'w', encoding='utf-8') as f:
            json.dump(self.synonyms, f, ensure_ascii=False, indent=2)
            
        logger.info(f"✅ MeSH files saved to {output_path}")
        logger.info(f"  - Terms saved: {len(serializable_terms)}")
        logger.info(f"  - Synonyms saved: {len(self.synonyms)}")
    
    @staticmethod
    def filter_biomedical_terms(terms: Dict[str, Dict]) -> Dict[str, Dict]:
        """Filtra términos relevantes para biomedicina"""
        # Categorías relevantes:
        # A: Anatomía
        # B: Organismos
        # C: Enfermedades
        # D: Químicos y Drogas
        relevant_categories = {'A', 'B', 'C', 'D'}
        
        filtered_terms = {}
        
        for term_id, term_data in terms.items():
            for tree_number in term_data['tree_numbers']:
                if tree_number[0] in relevant_categories:
                    filtered_terms[term_id] = term_data
                    break
                    
        return filtered_terms
