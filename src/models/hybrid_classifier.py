"""
Enriquecimiento semántico usando diccionarios biomédicos
FDA, OMS/WHO, INVIMA y otros vocabularios controlados
"""

import pandas as pd
import numpy as np
import re
from collections import defaultdict
from typing import List, Dict, Set, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SemanticEnrichment:
    """
    Enriquecedor semántico usando diccionarios biomédicos especializados
    """
    
    def __init__(self):
        """
        Inicializa los diccionarios biomédicos
        """
        logger.info("Inicializando diccionarios biomédicos...")
        
        # Diccionarios farmacológicos y médicos
        self._load_dictionaries()
        
        logger.info("✅ Diccionarios cargados exitosamente")
        logger.info(f"  - FDA terms: {len(self.fda_terms)}")
        logger.info(f"  - WHO terms: {len(self.who_terms)}")
        logger.info(f"  - INVIMA terms: {len(self.invima_terms)}")
        logger.info(f"  - Disease terms: {len(self.disease_terms)}")
        logger.info(f"  - Drug terms: {len(self.drug_terms)}")
    
    def _load_dictionaries(self):
        """
        Carga los diccionarios especializados
        En producción, estos vendrían de APIs oficiales o bases de datos actualizadas
        """
        
        # Términos FDA (Food and Drug Administration)
        self.fda_terms = {
            # Categorías de medicamentos FDA
            'antibiotic', 'antiviral', 'antifungal', 'antihistamine',
            'analgesic', 'anti-inflammatory', 'immunosuppressant',
            'chemotherapy', 'biologic', 'biosimilar', 'orphan drug',
            'clinical trial', 'phase i', 'phase ii', 'phase iii',
            'adverse event', 'side effect', 'contraindication',
            'dosage', 'pharmacokinetics', 'bioavailability',
            'fda approved', 'breakthrough therapy', 'fast track',
            
            # Dispositivos médicos
            'medical device', 'diagnostic', 'therapeutic device',
            'implant', 'prosthetic', 'pacemaker', 'stent',
            
            # Seguridad alimentaria
            'food safety', 'contamination', 'recall', 'inspection',
            'gmp', 'good manufacturing practice'
        }
        
        # Términos OMS/WHO (World Health Organization)
        self.who_terms = {
            # Clasificaciones WHO
            'icd-10', 'icd-11', 'who classification',
            'essential medicines', 'global health',
            'pandemic', 'epidemic', 'outbreak',
            'surveillance', 'public health emergency',
            
            # Enfermedades prioritarias WHO
            'tuberculosis', 'malaria', 'hiv', 'aids',
            'hepatitis', 'cholera', 'yellow fever',
            'zika', 'ebola', 'sars', 'mers',
            'antimicrobial resistance', 'superbugs',
            
            # Programas WHO
            'immunization', 'vaccination', 'eradication',
            'health systems', 'universal health coverage',
            'neglected diseases', 'maternal health',
            'child mortality', 'nutrition'
        }
        
        # Términos INVIMA (Instituto Nacional de Vigilancia de Medicamentos)
        self.invima_terms = {
            # Regulación Colombia
            'registro sanitario', 'farmacovigilancia',
            'reacciones adversas', 'medicamento generico',
            'biosimilar', 'fitoterapeutico',
            'suplemento dietario', 'dispositivo medico',
            'cosmetico', 'alimento', 'bebida alcoholica',
            
            # Procesos regulatorios
            'buenas practicas', 'inspeccion', 'certificacion',
            'autorizacion', 'licencia', 'permiso',
            'vigilancia sanitaria', 'control calidad',
            
            # Categorías productos
            'medicamento vital no disponible', 'mvnd',
            'principio activo', 'excipiente',
            'forma farmaceutica', 'concentracion'
        }
        
        # Diccionario de enfermedades expandido
        self.disease_terms = {
            # Cardiovasculares
            'hypertension', 'heart disease', 'myocardial infarction',
            'stroke', 'arrhythmia', 'heart failure', 'angina',
            'atherosclerosis', 'cardiomyopathy',
            
            # Cáncer
            'cancer', 'tumor', 'malignancy', 'carcinoma',
            'sarcoma', 'lymphoma', 'leukemia', 'metastasis',
            'oncology', 'chemotherapy', 'radiation therapy',
            
            # Neurológicas
            'alzheimer', 'parkinson', 'epilepsy', 'stroke',
            'multiple sclerosis', 'depression', 'anxiety',
            'schizophrenia', 'bipolar', 'dementia',
            
            # Infecciosas
            'infection', 'bacteria', 'virus', 'fungus',
            'pneumonia', 'sepsis', 'meningitis',
            'covid-19', 'influenza', 'respiratory infection',
            
            # Metabólicas
            'diabetes', 'obesity', 'metabolic syndrome',
            'thyroid disease', 'cholesterol', 'lipid disorder',
            
            # Inmunológicas
            'autoimmune', 'allergy', 'asthma', 'arthritis',
            'lupus', 'inflammatory bowel', 'psoriasis'
        }
        
        # Diccionario de medicamentos expandido
        self.drug_terms = {
            # Clases terapéuticas
            'ace inhibitor', 'beta blocker', 'calcium channel blocker',
            'diuretic', 'statin', 'anticoagulant', 'antiplatelet',
            'proton pump inhibitor', 'selective serotonin reuptake inhibitor',
            'nsaid', 'corticosteroid', 'immunosuppressant',
            
            # Antibióticos
            'penicillin', 'cephalosporin', 'fluoroquinolone',
            'macrolide', 'tetracycline', 'aminoglycoside',
            'carbapenem', 'vancomycin',
            
            # Antivirales
            'acyclovir', 'oseltamivir', 'ritonavir',
            'interferon', 'nucleoside analog',
            
            # Quimioterapia
            'cisplatin', 'doxorubicin', 'paclitaxel',
            'carboplatin', 'cyclophosphamide', 'methotrexate',
            
            # Formas farmacéuticas
            'tablet', 'capsule', 'injection', 'infusion',
            'topical', 'transdermal', 'inhalation', 'suppository'
        }
        
        # Compilar patrones de regex para búsqueda eficiente
        self._compile_patterns()
    
    def _compile_patterns(self):
        """
        Compila patrones de regex para búsqueda eficiente
        """
        self.patterns = {}
        
        for dict_name, terms in [
            ('fda', self.fda_terms),
            ('who', self.who_terms), 
            ('invima', self.invima_terms),
            ('disease', self.disease_terms),
            ('drug', self.drug_terms)
        ]:
            # Crear patrón que busca términos completos
            pattern = r'\b(?:' + '|'.join(re.escape(term) for term in terms) + r')\b'
            self.patterns[dict_name] = re.compile(pattern, re.IGNORECASE)
    
    def extract_dictionary_features(self, text: str) -> Dict[str, int]:
        """
        Extrae features basadas en diccionarios de un texto
        
        Args:
            text: Texto a analizar
            
        Returns:
            Diccionario con conteos por diccionario
        """
        if pd.isna(text) or text.strip() == "":
            return {dict_name: 0 for dict_name in self.patterns.keys()}
        
        features = {}
        text_lower = text.lower()
        
        for dict_name, pattern in self.patterns.items():
            matches = pattern.findall(text_lower)
            features[f'{dict_name}_terms'] = len(matches)
            features[f'{dict_name}_unique_terms'] = len(set(matches))
        
        # Features adicionales
        total_terms = sum(features[k] for k in features.keys() if k.endswith('_terms'))
        features['total_dict_terms'] = total_terms
        
        # Ratios
        if total_terms > 0:
            for dict_name in self.patterns.keys():
                count = features[f'{dict_name}_terms']
                features[f'{dict_name}_ratio'] = count / total_terms
        else:
            for dict_name in self.patterns.keys():
                features[f'{dict_name}_ratio'] = 0.0
        
        return features
    
    def extract_from_corpus(self, texts: List[str]) -> pd.DataFrame:
        """
        Extrae features de diccionarios de un corpus
        
        Args:
            texts: Lista de textos
            
        Returns:
            DataFrame con features de diccionarios
        """
        logger.info(f"Extrayendo features de diccionarios de {len(texts)} textos...")
        
        features_list = []
        for text in texts:
            features = self.extract_dictionary_features(text)
            features_list.append(features)
        
        df_features = pd.DataFrame(features_list)
        
        logger.info(f"✅ Features extraídas: {df_features.shape}")
        return df_features
    
    def get_matched_terms(self, text: str) -> Dict[str, List[str]]:
        """
        Obtiene los términos específicos que coincidieron en cada diccionario
        
        Args:
            text: Texto a analizar
            
        Returns:
            Diccionario con listas de términos encontrados
        """
        if pd.isna(text) or text.strip() == "":
            return {dict_name: [] for dict_name in self.patterns.keys()}
        
        matched_terms = {}
        text_lower = text.lower()
        
        for dict_name, pattern in self.patterns.items():
            matches = pattern.findall(text_lower)
            matched_terms[dict_name] = list(set(matches))  # Remover duplicados
        
        return matched_terms
    
    def analyze_dictionary_coverage(self, texts: List[str]) -> pd.DataFrame:
        """
        Analiza la cobertura de cada diccionario en el corpus
        
        Args:
            texts: Lista de textos
            
        Returns:
            DataFrame con estadísticas de cobertura
        """
        logger.info("Analizando cobertura de diccionarios...")
        
        stats = []
        
        for dict_name in self.patterns.keys():
            texts_with_terms = 0
            total_terms = 0
            unique_terms = set()
            
            for text in texts:
                features = self.extract_dictionary_features(text)
                terms_count = features[f'{dict_name}_terms']
                
                if terms_count > 0:
                    texts_with_terms += 1
                    total_terms += terms_count
                    
                    # Obtener términos únicos
                    matched = self.get_matched_terms(text)
                    unique_terms.update(matched[dict_name])
            
            stats.append({
                'dictionary': dict_name,
                'texts_with_terms': texts_with_terms,
                'coverage_percentage': (texts_with_terms / len(texts)) * 100,
                'total_term_occurrences': total_terms,
                'unique_terms_found': len(unique_terms),
                'avg_terms_per_text': total_terms / max(texts_with_terms, 1)
            })
        
        return pd.DataFrame(stats).sort_values('coverage_percentage', ascending=False)


class ClinicalTrialEnrichment:
    """
    Enriquecimiento específico para términos de ensayos clínicos
    """
    
    def __init__(self):
        """
        Inicializa términos específicos de ensayos clínicos
        """
        self.trial_phases = {
            'phase 0', 'phase i', 'phase ii', 'phase iii', 'phase iv',
            'preclinical', 'clinical trial', 'randomized controlled trial',
            'double blind', 'placebo controlled', 'crossover study'
        }
        
        self.trial_outcomes = {
            'primary endpoint', 'secondary endpoint', 'efficacy',
            'safety', 'tolerability', 'adverse event', 'serious adverse event',
            'dose limiting toxicity', 'maximum tolerated dose',
            'objective response rate', 'progression free survival',
            'overall survival', 'quality of life'
        }
        
        self.regulatory_terms = {
            'fda approval', 'ema approval', 'breakthrough designation',
            'orphan drug designation', 'fast track', 'accelerated approval',
            'priority review', 'compassionate use', 'expanded access'
        }
    
    def extract_trial_features(self, text: str) -> Dict[str, int]:
        """
        Extrae features específicas de ensayos clínicos
        
        Args:
            text: Texto a analizar
            
        Returns:
            Diccionario con features de ensayos clínicos
        """
        if pd.isna(text):
            text = ""
        
        text_lower = text.lower()
        
        features = {}
        
        # Conteo de términos por categoría
        for category, terms in [
            ('trial_phase', self.trial_phases),
            ('trial_outcome', self.trial_outcomes),
            ('regulatory', self.regulatory_terms)
        ]:
            count = sum(1 for term in terms if term in text_lower)
            features[f'{category}_terms'] = count
        
        # Detección de números de ensayos clínicos (patrón NCT)
        nct_pattern = r'NCT\d{8}'
        nct_matches = re.findall(nct_pattern, text, re.IGNORECASE)
        features['clinical_trial_ids'] = len(nct_matches)
        
        # Detección de códigos de protocolo
        protocol_pattern = r'protocol\s+\w+[-\w]*'
        protocol_matches = re.findall(protocol_pattern, text, re.IGNORECASE)
        features['protocol_codes'] = len(protocol_matches)
        
        return features