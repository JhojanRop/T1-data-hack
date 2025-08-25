import json
import logging
from pathlib import Path
from typing import Dict, List, Set, Tuple
from tqdm import tqdm

# Configurar logging
logging.basicConfig(
    filename='logs/mesh_processing.log',
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class MeSHProcessor:
    def __init__(self, mesh_file: str = "data/external/mesh/d2025.bin"):
        """
        Inicializa el procesador de MeSH
        
        Args:
            mesh_file: Ruta al archivo binario de MeSH
        """
        self.mesh_file = Path(mesh_file)
        if not self.mesh_file.exists():
            raise FileNotFoundError(f"Archivo MeSH no encontrado: {mesh_file}")
            
        self.output_dir = Path("data/processed/mesh")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Estructuras de datos para términos MeSH
        self.mesh_terms: Dict[str, str] = {}  # descriptor_id -> término preferido
        self.mesh_synonyms: Dict[str, List[str]] = {}  # término -> [sinónimos]
        
    def _parse_mesh_file(self) -> None:
        """Lee y parsea el archivo binario de MeSH"""
        try:
            logger.info(f"Iniciando parseo de archivo MeSH: {self.mesh_file}")
            current_descriptor = None
            current_terms = set()
            
            with open(self.mesh_file, 'r', encoding='utf-8') as f:
                for line in tqdm(f, desc="Procesando archivo MeSH"):
                    line = line.strip()
                    
                    # Nuevo descriptor
                    if line.startswith("*NEWRECORD"):
                        if current_descriptor and current_terms:
                            self._process_descriptor(current_descriptor, current_terms)
                        current_descriptor = None
                        current_terms = set()
                        
                    # ID del descriptor
                    elif line.startswith("MH = "):
                        term = line[5:].strip()
                        current_terms.add(term)
                        
                    elif line.startswith("UI = "):
                        current_descriptor = line[5:].strip()
                        
                    # Términos de entrada (sinónimos)
                    elif line.startswith("ENTRY = "):
                        entry = line[8:].split("|")[0].strip()
                        if entry:
                            current_terms.add(entry)
                            
            # Procesar último descriptor
            if current_descriptor and current_terms:
                self._process_descriptor(current_descriptor, current_terms)
                
            logger.info(f"Parseo completado. {len(self.mesh_terms)} descriptores procesados")
            
        except Exception as e:
            logger.error(f"Error parseando archivo MeSH: {e}")
            raise
            
    def _process_descriptor(self, descriptor_id: str, terms: Set[str]) -> None:
        """
        Procesa un descriptor MeSH y sus términos asociados
        
        Args:
            descriptor_id: ID del descriptor MeSH
            terms: Conjunto de términos asociados
        """
        if not terms:
            return
            
        # El primer término es el preferido
        preferred_term = list(terms)[0]
        self.mesh_terms[descriptor_id] = preferred_term
        
        # Los demás son sinónimos
        for term in terms:
            if term != preferred_term:
                if term not in self.mesh_synonyms:
                    self.mesh_synonyms[term] = []
                self.mesh_synonyms[term].append(preferred_term)
                
    def _save_processed_data(self) -> None:
        """Guarda los términos procesados en archivos JSON"""
        try:
            # Guardar términos principales
            terms_file = self.output_dir / "mesh_terms.json"
            with open(terms_file, 'w', encoding='utf-8') as f:
                json.dump(self.mesh_terms, f, indent=2, ensure_ascii=False)
            logger.info(f"Términos MeSH guardados en {terms_file}")
            
            # Guardar sinónimos
            synonyms_file = self.output_dir / "mesh_synonyms.json"
            with open(synonyms_file, 'w', encoding='utf-8') as f:
                json.dump(self.mesh_synonyms, f, indent=2, ensure_ascii=False)
            logger.info(f"Sinónimos MeSH guardados en {synonyms_file}")
            
        except Exception as e:
            logger.error(f"Error guardando datos procesados: {e}")
            raise
            
    def process(self) -> Tuple[Dict[str, str], Dict[str, List[str]]]:
        """
        Procesa el archivo MeSH completo
        
        Returns:
            Tupla con (términos MeSH, sinónimos)
        """
        try:
            logger.info("Iniciando procesamiento de MeSH")
            
            # Parsear archivo
            self._parse_mesh_file()
            
            # Guardar resultados
            self._save_processed_data()
            
            logger.info("Procesamiento de MeSH completado exitosamente")
            return self.mesh_terms, self.mesh_synonyms
            
        except Exception as e:
            logger.error(f"Error en procesamiento de MeSH: {e}")
            raise

def main():
    """Punto de entrada principal"""
    try:
        # Inicializar procesador
        mesh_file = "data/external/mesh/d2025.bin"
        processor = MeSHProcessor(mesh_file=mesh_file)
        
        # Procesar archivo MeSH
        terms, synonyms = processor.process()
        
        # Imprimir estadísticas
        print("\n=== Estadísticas de Procesamiento ===")
        print(f"Términos MeSH procesados: {len(terms):,}")
        print(f"Sinónimos encontrados: {len(synonyms):,}")
        print("\nArchivos generados:")
        print("- data/processed/mesh/mesh_terms.json")
        print("- data/processed/mesh/mesh_synonyms.json")
        
    except Exception as e:
        logger.error(f"Error en ejecución principal: {e}")
        raise

if __name__ == "__main__":
    main()
