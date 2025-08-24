"""
Clasificador híbrido que combina embeddings biomédicos, NER y enriquecimiento semántico
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler, MultiLabelBinarizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import classification_report, confusion_matrix, f1_score
import xgboost as xgb
import lightgbm as lgb
from typing import List, Dict, Tuple, Optional, Union
import logging
import pickle
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

# Importar nuestros módulos
from ..features.biomedical_embeddings import BiomedicalEmbeddings
from ..features.biomedical_ner import BiomedicalNER, MeSHMapper
from ..features.semantic_enrichment import SemanticEnrichment, ClinicalTrialEnrichment

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class HybridBiomedicalClassifier:
    """
    Clasificador híbrido que combina múltiples fuentes de información biomédica
    """
    
    def __init__(self, 
                 embedding_model: str = "all-MiniLM-L6-v2",
                 ner_model: str = "en_core_sci_sm",
                 classifier_type: str = "xgboost",
                 use_embeddings: bool = True,
                 use_ner: bool = True,
                 use_semantic: bool = True,
                 use_mesh: bool = True,
                 use_clinical_trials: bool = True):
        """
        Inicializa el clasificador híbrido
        
        Args:
            embedding_model: Modelo para embeddings
            ner_model: Modelo para NER biomédico
            classifier_type: 'xgboost' o 'lightgbm'
            use_embeddings: Usar embeddings biomédicos
            use_ner: Usar features de NER
            use_semantic: Usar enriquecimiento semántico
            use_mesh: Usar mapeo a MeSH
            use_clinical_trials: Usar features de ensayos clínicos
        """
        self.embedding_model = embedding_model
        self.ner_model = ner_model
        self.classifier_type = classifier_type
        
        # Configuración de componentes
        self.use_embeddings = use_embeddings
        self.use_ner = use_ner
        self.use_semantic = use_semantic
        self.use_mesh = use_mesh
        self.use_clinical_trials = use_clinical_trials
        
        # Inicializar componentes
        self._initialize_components()
        
        # Objetos de entrenamiento
        self.mlb = MultiLabelBinarizer()
        self.scaler = StandardScaler()
        self.classifier = None
        self.feature_names = []
        
        logger.info("✅ HybridBiomedicalClassifier inicializado")
        self._log_configuration()
    
    def _initialize_components(self):
        """Inicializa los componentes según la configuración"""
        self.components = {}
        
        if self.use_embeddings:
            try:
                self.components['embeddings'] = BiomedicalEmbeddings(self.embedding_model)
                logger.info("✅ Componente de embeddings inicializado")
            except Exception as e:
                logger.warning(f"⚠️ Error inicializando embeddings: {e}")
                self.use_embeddings = False
        
        if self.use_ner or self.use_mesh:
            try:
                self.components['ner'] = BiomedicalNER(self.ner_model)
                logger.info("✅ Componente de NER inicializado")
                
                if self.use_mesh:
                    self.components['mesh'] = MeSHMapper()
                    logger.info("✅ Componente MeSH inicializado")
            except Exception as e:
                logger.warning(f"⚠️ Error inicializando NER: {e}")
                self.use_ner = False
                self.use_mesh = False
        
        if self.use_semantic:
            try:
                self.components['semantic'] = SemanticEnrichment()
                logger.info("✅ Componente semántico inicializado")
            except Exception as e:
                logger.warning(f"⚠️ Error inicializando enriquecimiento semántico: {e}")
                self.use_semantic = False
        
        if self.use_clinical_trials:
            try:
                self.components['clinical'] = ClinicalTrialEnrichment()
                logger.info("✅ Componente de ensayos clínicos inicializado")
            except Exception as e:
                logger.warning(f"⚠️ Error inicializando ensayos clínicos: {e}")
                self.use_clinical_trials = False
    
    def _log_configuration(self):
        """Log de la configuración actual"""
        logger.info("Configuración del clasificador híbrido:")
        logger.info(f"  - Embeddings: {'✅' if self.use_embeddings else '❌'}")
        logger.info(f"  - NER biomédico: {'✅' if self.use_ner else '❌'}")
        logger.info(f"  - Enriquecimiento semántico: {'✅' if self.use_semantic else '❌'}")
        logger.info(f"  - Mapeo MeSH: {'✅' if self.use_mesh else '❌'}")
        logger.info(f"  - Ensayos clínicos: {'✅' if self.use_clinical_trials else '❌'}")
        logger.info(f"  - Clasificador: {self.classifier_type}")
    
    def extract_features(self, df: pd.DataFrame, 
                        title_col: str = 'title', 
                        abstract_col: str = 'abstract') -> np.ndarray:
        """
        Extrae todas las features configuradas
        
        Args:
            df: DataFrame con los datos
            title_col: Columna de títulos
            abstract_col: Columna de abstracts
            
        Returns:
            Array de features combinadas
        """
        logger.info(f"Extrayendo features de {len(df)} documentos...")
        
        feature_matrices = []
        feature_names = []
        
        # Combinar título y abstract
        combined_texts = []
        for _, row in df.iterrows():
            title = row[title_col] if pd.notna(row[title_col]) else ""
            abstract = row[abstract_col] if pd.notna(row[abstract_col]) else ""
            combined_texts.append(f"{title}. {abstract}")
        
        # 1. Embeddings biomédicos
        if self.use_embeddings and 'embeddings' in self.components:
            logger.info("Extrayendo embeddings biomédicos...")
            embeddings = self.components['embeddings'].extract_embeddings(combined_texts)
            feature_matrices.append(embeddings)
            
            # Nombres de features para embeddings
            embed_names = [f'embed_{i}' for i in range(embeddings.shape[1])]
            feature_names.extend(embed_names)
            
            logger.info(f"✅ Embeddings: {embeddings.shape}")
        
        # 2. Features de NER biomédico
        if self.use_ner and 'ner' in self.components:
            logger.info("Extrayendo features de NER biomédico...")
            entities_list = self.components['ner'].extract_from_corpus(combined_texts)
            ner_features = self.components['ner'].create_entity_features(entities_list)
            
            feature_matrices.append(ner_features.values)
            feature_names.extend([f'ner_{col}' for col in ner_features.columns])
            
            logger.info(f"✅ Features NER: {ner_features.shape}")
            
            # 3. Features MeSH
            if self.use_mesh and 'mesh' in self.components:
                logger.info("Extrayendo features MeSH...")
                mesh_features = self.components['mesh'].create_mesh_features(entities_list)
                
                feature_matrices.append(mesh_features.values)
                feature_names.extend([f'mesh_{col}' for col in mesh_features.columns])
                
                logger.info(f"✅ Features MeSH: {mesh_features.shape}")
        
        # 4. Enriquecimiento semántico
        if self.use_semantic and 'semantic' in self.components:
            logger.info("Extrayendo features semánticas...")
            semantic_features = self.components['semantic'].extract_from_corpus(combined_texts)
            
            feature_matrices.append(semantic_features.values)
            feature_names.extend([f'semantic_{col}' for col in semantic_features.columns])
            
            logger.info(f"✅ Features semánticas: {semantic_features.shape}")
        
        # 5. Features de ensayos clínicos
        if self.use_clinical_trials and 'clinical' in self.components:
            logger.info("Extrayendo features de ensayos clínicos...")
            clinical_features = []
            
            for text in combined_texts:
                features = self.components['clinical'].extract_trial_features(text)
                clinical_features.append(features)
            
            clinical_df = pd.DataFrame(clinical_features)
            feature_matrices.append(clinical_df.values)
            feature_names.extend([f'clinical_{col}' for col in clinical_df.columns])
            
            logger.info(f"✅ Features ensayos clínicos: {clinical_df.shape}")
        
        # Combinar todas las features
        if feature_matrices:
            combined_features = np.hstack(feature_matrices)
            self.feature_names = feature_names
            
            logger.info(f"✅ Features combinadas: {combined_features.shape}")
            return combined_features
        else:
            raise ValueError("No se pudieron extraer features. Verifique la configuración.")
    
    def _get_classifier(self):
        """Obtiene el clasificador configurado"""
        if self.classifier_type.lower() == 'xgboost':
            base_classifier = xgb.XGBClassifier(
                objective='binary:logistic',
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=42,
                eval_metric='logloss'
            )
        elif self.classifier_type.lower() == 'lightgbm':
            base_classifier = lgb.LGBMClassifier(
                objective='binary',
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=42,
                verbosity=-1
            )
        else:
            raise ValueError(f"Clasificador no soportado: {self.classifier_type}")
        
        return OneVsRestClassifier(base_classifier)
    
    def train(self, df: pd.DataFrame, 
              title_col: str = 'title',
              abstract_col: str = 'abstract', 
              label_col: str = 'group') -> Dict:
        """
        Entrena el clasificador híbrido
        
        Args:
            df: DataFrame con los datos
            title_col: Columna de títulos
            abstract_col: Columna de abstracts
            label_col: Columna de etiquetas
            
        Returns:
            Diccionario con métricas de entrenamiento
        """
        logger.info("Iniciando entrenamiento del clasificador híbrido...")
        
        # Extraer features
        X = self.extract_features(df, title_col, abstract_col)
        
        # Preparar etiquetas
        labels = df[label_col].tolist()
        y = self.mlb.fit_transform(labels)
        
        logger.info(f"Etiquetas: {y.shape}, Clases: {len(self.mlb.classes_)}")
        
        # Escalar features
        X_scaled = self.scaler.fit_transform(X)
        
        # Entrenar clasificador
        self.classifier = self._get_classifier()
        self.classifier.fit(X_scaled, y)
        
        logger.info("✅ Entrenamiento completado")
        
        # Evaluar en datos de entrenamiento
        y_pred = self.classifier.predict(X_scaled)
        
        metrics = {}
        for avg_type in ['micro', 'macro', 'weighted']:
            score = f1_score(y, y_pred, average=avg_type, zero_division=0)
            metrics[f'f1_{avg_type}'] = score
        
        return metrics
    
    def cross_validate(self, df: pd.DataFrame,
                      title_col: str = 'title',
                      abstract_col: str = 'abstract', 
                      label_col: str = 'group',
                      cv_folds: int = 5) -> Dict:
        """
        Realiza validación cruzada
        
        Args:
            df: DataFrame con los datos
            title_col: Columna de títulos  
            abstract_col: Columna de abstracts
            label_col: Columna de etiquetas
            cv_folds: Número de folds
            
        Returns:
            Diccionario con métricas de validación cruzada
        """
        logger.info(f"Iniciando validación cruzada con {cv_folds} folds...")
        
        # Extraer features
        X = self.extract_features(df, title_col, abstract_col)
        
        # Preparar etiquetas
        labels = df[label_col].tolist()
        y = self.mlb.fit_transform(labels)
        
        # Escalar features
        X_scaled = self.scaler.fit_transform(X)
        
        # Validación cruzada manual para multilabel
        kf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
        
        # Para StratifiedKFold con multilabel, usar la clase más frecuente por muestra
        y_single = np.argmax(y, axis=1)
        
        cv_scores = {
            'f1_micro': [],
            'f1_macro': [],
            'f1_weighted': []
        }
        
        for fold, (train_idx, val_idx) in enumerate(kf.split(X_scaled, y_single)):
            logger.info(f"Procesando fold {fold + 1}/{cv_folds}")
            
            X_train, X_val = X_scaled[train_idx], X_scaled[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            # Entrenar
            classifier = self._get_classifier()
            classifier.fit(X_train, y_train)
            
            # Predecir
            y_pred = classifier.predict(X_val)
            
            # Calcular métricas
            for avg_type in ['micro', 'macro', 'weighted']:
                score = f1_score(y_val, y_pred, average=avg_type, zero_division=0)
                cv_scores[f'f1_{avg_type}'].append(score)
        
        # Calcular estadísticas
        results = {}
        for metric, scores in cv_scores.items():
            results[f'{metric}_mean'] = np.mean(scores)
            results[f'{metric}_std'] = np.std(scores)
        
        logger.info("✅ Validación cruzada completada")
        return results
    
    def get_feature_importance(self, top_n: int = 20) -> pd.DataFrame:
        """
        Obtiene la importancia de las features
        
        Args:
            top_n: Número de features top a retornar
            
        Returns:
            DataFrame con importancia de features
        """
        if self.classifier is None:
            raise ValueError("El modelo debe estar entrenado primero")
        
        # Obtener importancias de todos los clasificadores binarios
        importances = []
        
        for i, estimator in enumerate(self.classifier.estimators_):
            if hasattr(estimator, 'feature_importances_'):
                importances.append(estimator.feature_importances_)
        
        if not importances:
            logger.warning("El clasificador no proporciona importancia de features")
            return pd.DataFrame()
        
        # Promediar importancias
        avg_importance = np.mean(importances, axis=0)
        
        # Crear DataFrame
        importance_df = pd.DataFrame({
            'feature': self.feature_names[:len(avg_importance)],
            'importance': avg_importance
        }).sort_values('importance', ascending=False)
        
        return importance_df.head(top_n)
    
    def save_model(self, filepath: str):
        """Guarda el modelo entrenado"""
        model_data = {
            'classifier': self.classifier,
            'mlb': self.mlb,
            'scaler': self.scaler,
            'feature_names': self.feature_names,
            'config': {
                'embedding_model': self.embedding_model,
                'ner_model': self.ner_model,
                'classifier_type': self.classifier_type,
                'use_embeddings': self.use_embeddings,
                'use_ner': self.use_ner,
                'use_semantic': self.use_semantic,
                'use_mesh': self.use_mesh,
                'use_clinical_trials': self.use_clinical_trials
            }
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        logger.info(f"✅ Modelo guardado en: {filepath}")
    
    def load_model(self, filepath: str):
        """Carga un modelo previamente entrenado"""
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        self.classifier = model_data['classifier']
        self.mlb = model_data['mlb']
        self.scaler = model_data['scaler']
        self.feature_names = model_data['feature_names']
        
        # Restaurar configuración
        config = model_data['config']
        self.embedding_model = config['embedding_model']
        self.ner_model = config['ner_model']
        self.classifier_type = config['classifier_type']
        self.use_embeddings = config['use_embeddings']
        self.use_ner = config['use_ner']
        self.use_semantic = config['use_semantic']
        self.use_mesh = config['use_mesh']
        self.use_clinical_trials = config['use_clinical_trials']
        
        # Reinicializar componentes
        self._initialize_components()
        
        logger.info(f"✅ Modelo cargado desde: {filepath}")


def compare_models(baseline_scores: Dict, hybrid_scores: Dict) -> pd.DataFrame:
    """
    Compara las métricas del baseline vs híbrido
    
    Args:
        baseline_scores: Métricas del modelo baseline
        hybrid_scores: Métricas del modelo híbrido
        
    Returns:
        DataFrame con comparación
    """
    comparison = []
    
    for metric in baseline_scores.keys():
        if metric in hybrid_scores:
            baseline_val = baseline_scores[metric]
            hybrid_val = hybrid_scores[metric]
            improvement = hybrid_val - baseline_val
            improvement_pct = (improvement / baseline_val) * 100 if baseline_val > 0 else 0
            
            comparison.append({
                'metric': metric,
                'baseline': baseline_val,
                'hybrid': hybrid_val,
                'improvement': improvement,
                'improvement_pct': improvement_pct
            })
    
    return pd.DataFrame(comparison)
