"""
Sophisticated Multi-Model AI Manager using Hugging Face Transformers
Manages loading, caching, and inference with multiple pre-trained models
"""

import json
import os
from typing import Dict, List, Any, Optional
import logging
import torch
from transformers import (
    pipeline,
    AutoTokenizer,
    AutoModelForSequenceClassification,
    AutoModelForTokenClassification,
    AutoModelForQuestionAnswering,
    AutoModelForCausalLM,
)
from huggingface_hub import hf_hub_download
import warnings

warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelManager:
    """Sophisticated manager for multiple Hugging Face models"""
    
    def __init__(self, config_path: str = "config.json"):
        """
        Initialize ModelManager with configuration
        
        Args:
            config_path: Path to configuration JSON file
        """
        self.config = self._load_config(config_path)
        self.models_cache: Dict[str, Any] = {}
        self.pipelines_cache: Dict[str, Any] = {}
        self.device = self._setup_device()
        logger.info(f"ModelManager initialized. Using device: {self.device}")
    
    def _load_config(self, config_path: str) -> Dict:
        """Load configuration from JSON file"""
        try:
            with open(config_path, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            logger.error(f"Config file not found: {config_path}")
            return self._default_config()
    
    def _default_config(self) -> Dict:
        """Return default configuration if config file not found"""
        return {
            "models": {
                "text_generation": {
                    "model_id": "openai-community/gpt2",
                    "task": "text-generation"
                },
                "text_classification": {
                    "model_id": "distilbert-base-uncased-finetuned-sst-2-english",
                    "task": "text-classification"
                }
            },
            "settings": {
                "cache_dir": "./model_cache",
                "device": "cpu",
                "enable_gpu": False
            }
        }
    
    def _setup_device(self) -> str:
        """Setup device (CPU or GPU)"""
        if self.config["settings"].get("enable_gpu") and torch.cuda.is_available():
            logger.info("GPU available and enabled")
            return "cuda"
        return "cpu"
    
    def get_model(self, model_name: str) -> Any:
        """
        Get or load a model by name
        
        Args:
            model_name: Name of the model from config
            
        Returns:
            Loaded model or None if not found
        """
        if model_name in self.models_cache:
            logger.info(f"Loading model '{model_name}' from cache")
            return self.models_cache[model_name]
        
        if model_name not in self.config["models"]:
            logger.error(f"Model '{model_name}' not found in configuration")
            return None
        
        model_config = self.config["models"][model_name]
        model_id = model_config.get("model_id")
        
        try:
            logger.info(f"Loading model '{model_name}' from Hugging Face: {model_id}")
            model = AutoModelForSequenceClassification.from_pretrained(
                model_id,
                cache_dir=self.config["settings"].get("cache_dir")
            )
            self.models_cache[model_name] = model
            return model
        except Exception as e:
            logger.error(f"Error loading model '{model_name}': {str(e)}")
            return None
    
    def get_pipeline(self, model_name: str) -> Optional[Any]:
        """
        Get or create a pipeline for a model
        
        Args:
            model_name: Name of the model from config
            
        Returns:
            Pipeline object or None if error
        """
        if model_name in self.pipelines_cache:
            logger.info(f"Using cached pipeline for '{model_name}'")
            return self.pipelines_cache[model_name]
        
        if model_name not in self.config["models"]:
            logger.error(f"Model '{model_name}' not found in configuration")
            return None
        
        model_config = self.config["models"][model_name]
        model_id = model_config.get("model_id")
        task = model_config.get("task")
        
        try:
            logger.info(f"Creating pipeline for '{model_name}' (task: {task})")
            pipe = pipeline(
                task=task,
                model=model_id,
                device=0 if self.device == "cuda" else -1,
                cache_dir=self.config["settings"].get("cache_dir")
            )
            self.pipelines_cache[model_name] = pipe
            return pipe
        except Exception as e:
            logger.error(f"Error creating pipeline for '{model_name}': {str(e)}")
            return None
    
    def list_models(self) -> List[str]:
        """List all available models"""
        return list(self.config["models"].keys())
    
    def get_model_info(self, model_name: str) -> Optional[Dict]:
        """Get information about a specific model"""
        if model_name not in self.config["models"]:
            return None
        return self.config["models"][model_name]
    
    def infer_text_generation(self, model_name: str, prompt: str, max_length: int = 100) -> Optional[str]:
        """Generate text using specified model"""
        pipe = self.get_pipeline(model_name)
        if pipe is None:
            return None
        
        try:
            result = pipe(prompt, max_length=max_length, num_return_sequences=1)
            return result[0]["generated_text"] if result else None
        except Exception as e:
            logger.error(f"Error in text generation: {str(e)}")
            return None
    
    def infer_text_classification(self, model_name: str, text: str) -> Optional[Dict]:
        """Classify text sentiment or category"""
        pipe = self.get_pipeline(model_name)
        if pipe is None:
            return None
        
        try:
            result = pipe(text)
            return result[0] if result else None
        except Exception as e:
            logger.error(f"Error in text classification: {str(e)}")
            return None
    
    def infer_named_entity_recognition(self, model_name: str, text: str) -> Optional[List[Dict]]:
        """Extract named entities from text"""
        pipe = self.get_pipeline(model_name)
        if pipe is None:
            return None
        
        try:
            result = pipe(text)
            return result
        except Exception as e:
            logger.error(f"Error in NER: {str(e)}")
            return None
    
    def infer_question_answering(self, model_name: str, question: str, context: str) -> Optional[Dict]:
        """Answer questions based on context"""
        pipe = self.get_pipeline(model_name)
        if pipe is None:
            return None
        
        try:
            result = pipe(question=question, context=context)
            return result
        except Exception as e:
            logger.error(f"Error in QA: {str(e)}")
            return None
    
    def infer_summarization(self, model_name: str, text: str, max_length: int = 130, min_length: int = 30) -> Optional[str]:
        """Summarize text"""
        pipe = self.get_pipeline(model_name)
        if pipe is None:
            return None
        
        try:
            result = pipe(text, max_length=max_length, min_length=min_length, do_sample=False)
            return result[0]["summary_text"] if result else None
        except Exception as e:
            logger.error(f"Error in summarization: {str(e)}")
            return None
    
    def infer_zero_shot(self, model_name: str, text: str, labels: List[str]) -> Optional[Dict]:
        """Zero-shot classification without training"""
        pipe = self.get_pipeline(model_name)
        if pipe is None:
            return None
        
        try:
            result = pipe(text, labels, multi_class=False)
            return result
        except Exception as e:
            logger.error(f"Error in zero-shot classification: {str(e)}")
            return None
    
    def batch_inference(self, model_name: str, task: str, inputs: List[str], **kwargs) -> List[Optional[Dict]]:
        """Perform batch inference on multiple inputs"""
        results = []
        batch_size = self.config["settings"].get("batch_size", 32)
        
        for i in range(0, len(inputs), batch_size):
            batch = inputs[i:i + batch_size]
            logger.info(f"Processing batch {i//batch_size + 1} ({len(batch)} items)")
            
            for input_text in batch:
                if task == "text-generation":
                    result = self.infer_text_generation(model_name, input_text, **kwargs)
                elif task == "text-classification":
                    result = self.infer_text_classification(model_name, input_text)
                elif task == "token-classification":
                    result = self.infer_named_entity_recognition(model_name, input_text)
                elif task == "summarization":
                    result = self.infer_summarization(model_name, input_text, **kwargs)
                else:
                    result = None
                
                results.append(result)
        
        return results
    
    def clear_cache(self):
        """Clear all cached models and pipelines"""
        self.models_cache.clear()
        self.pipelines_cache.clear()
        logger.info("Cache cleared")
    
    def get_cache_info(self) -> Dict:
        """Get information about cached models"""
        return {
            "cached_models": len(self.models_cache),
            "cached_pipelines": len(self.pipelines_cache),
            "model_names": list(self.models_cache.keys()),
            "pipeline_names": list(self.pipelines_cache.keys())
        }


if __name__ == "__main__":
    # Example usage
    manager = ModelManager()
    print("Available models:", manager.list_models())
