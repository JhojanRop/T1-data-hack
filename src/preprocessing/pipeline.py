import logging
from typing import List, Dict
from .cleaner import TextCleaner
from .normalizer import SemanticNormalizer

class PreprocessingPipeline:
    def __init__(self):
        """Initialize the preprocessing pipeline with cleaner and normalizer components"""
        self.cleaner = TextCleaner()
        self.normalizer = SemanticNormalizer()
        self.logger = logging.getLogger(__name__)
        
        # Configure logging
        logging.basicConfig(
            filename='logs/preprocessing.log',
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )

    def process_text(self, text: str) -> str:
        """
        Apply full preprocessing pipeline to a single text
        
        Args:
            text (str): Input text to process
            
        Returns:
            str: Processed text
        """
        try:
            self.logger.info("Starting text processing")
            
            if not text or not isinstance(text, str):
                self.logger.warning("Empty or invalid text input")
                return ""
            
            # 1. Basic cleaning
            self.logger.debug("Applying text cleaning")
            text = self.cleaner.clean(text)
            
            # 2. Semantic normalization
            self.logger.debug("Applying semantic normalization")
            text = self.normalizer.normalize(text)
            
            self.logger.info("Text processing completed successfully")
            return text
        
        except Exception as e:
            self.logger.error(f"Error processing text: {str(e)}")
            return text

    def process_batch(self, texts: List[str]) -> List[str]:
        """
        Process a batch of texts through the pipeline
        
        Args:
            texts (List[str]): List of texts to process
            
        Returns:
            List[str]: List of processed texts
        """
        self.logger.info(f"Processing batch of {len(texts)} texts")
        return [self.process_text(text) for text in texts]

    def process_document(self, title: str, abstract: str) -> Dict[str, str]:
        """
        Process both title and abstract of a document
        
        Args:
            title (str): Document title
            abstract (str): Document abstract
            
        Returns:
            Dict[str, str]: Dictionary with processed title and abstract
        """
        self.logger.info("Processing document (title + abstract)")
        return {
            'title': self.process_text(title),
            'abstract': self.process_text(abstract)
        }
