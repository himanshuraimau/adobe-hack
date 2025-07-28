"""
Heading classification module using MobileBERT for text classification.

This module adapts the local MobileBERT model for heading classification,
combining textual content with extracted features to predict heading levels.
"""

import logging
import re
import torch
from pathlib import Path
from typing import Optional, Tuple, Dict, Any, List
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import numpy as np

try:
    from ..models.models import ProcessedBlock, ClassificationResult, FeatureVector
    from ..config.config import config
    from .error_handler import (
        ClassificationError, ModelLoadingError, global_error_handler,
        timeout_manager, safe_operation, retry_manager
    )
except ImportError:
    from src.pdf_extractor.models.models import ProcessedBlock, ClassificationResult, FeatureVector
    from src.pdf_extractor.config.config import config
    from src.pdf_extractor.core.error_handler import (
        ClassificationError, ModelLoadingError, global_error_handler,
        timeout_manager, safe_operation, retry_manager
    )

logger = logging.getLogger(__name__)


class HeadingClassifier:
    """Main classification logic using MobileBERT for heading detection."""
    
    def __init__(self):
        self.config = config.get_classification_config()
        self.model_adapter = MobileBERTAdapter()
        self.model_loaded = False
        self.fallback_classifier = FallbackRuleBasedClassifier()
    
    @safe_operation("text block classification")
    def classify_block(self, features: FeatureVector, text: str) -> ClassificationResult:
        """
        Classify a text block as title, heading, or regular text.
        
        Args:
            features: FeatureVector containing extracted features
            text: Text content to classify
            
        Returns:
            ClassificationResult with predicted class and confidence
        """
        try:
            # Validate inputs
            if not text or not text.strip():
                raise ClassificationError(
                    "Empty text provided for classification",
                    error_code="EMPTY_TEXT",
                    details={"text_length": len(text) if text else 0}
                )
            
            if not features:
                raise ClassificationError(
                    "No features provided for classification",
                    error_code="NO_FEATURES",
                    details={"text": text[:50]}
                )
            
            predicted_class = "text"
            confidence = 0.1
            
            # Try model-based classification first
            if self.model_loaded:
                try:
                    predicted_class, confidence = self.model_adapter.predict(text, features)
                    
                    # Detect if model is not working properly (always returns same class with high confidence)
                    model_seems_broken = (
                        confidence >= 0.99 and predicted_class == "title"  # Always predicting title with max confidence
                    )
                    
                    # Use fallback if confidence is too low OR model seems broken
                    if (confidence < self.config.get('confidence_threshold', 0.5)) or model_seems_broken:
                        if self.config.get('use_fallback_rules', True):
                            fallback_class, fallback_confidence = self.fallback_classifier.classify(text, features)
                            if model_seems_broken:
                                logger.debug(f"Model seems broken (always predicting {predicted_class} with {confidence:.3f}), using fallback: {text[:50]}...")
                            else:
                                logger.debug(f"Used fallback classification for low confidence: {text[:50]}...")
                            predicted_class, confidence = fallback_class, fallback_confidence
                            
                except Exception as model_error:
                    # Handle model prediction errors gracefully
                    predicted_class, confidence = global_error_handler.handle_classification_error(
                        model_error, text, self.fallback_classifier
                    )
                    logger.warning(f"Model classification failed, used fallback: {model_error}")
            else:
                # Use fallback if model not loaded
                predicted_class, confidence = self.fallback_classifier.classify(text, features)
                logger.debug("Model not loaded, using fallback classification")
            
            # Create a dummy ProcessedBlock for the result
            dummy_block = ProcessedBlock(
                text=text,
                page_number=1,  # This will be set properly by the caller
                features=features,
                original_block=None
            )
            
            return ClassificationResult(
                block=dummy_block,
                predicted_class=predicted_class,
                confidence=confidence
            )
            
        except ClassificationError:
            # Re-raise our custom errors
            raise
        except Exception as e:
            # Handle unexpected errors
            logger.error(f"Unexpected classification error: {e}")
            
            # Ultimate fallback
            try:
                predicted_class, confidence = self.fallback_classifier.classify(text, features)
            except Exception as fallback_error:
                logger.error(f"Fallback classification also failed: {fallback_error}")
                predicted_class, confidence = "text", 0.1
            
            dummy_block = ProcessedBlock(
                text=text,
                page_number=1,
                features=features,
                original_block=None
            )
            
            return ClassificationResult(
                block=dummy_block,
                predicted_class=predicted_class,
                confidence=confidence
            )
    
    @safe_operation("model loading")
    def load_model(self, model_path: str) -> None:
        """
        Load the MobileBERT model from the specified path.
        
        Args:
            model_path: Path to the MobileBERT model directory
        """
        try:
            # Validate model path
            if not model_path:
                raise ModelLoadingError(
                    "Empty model path provided",
                    error_code="EMPTY_MODEL_PATH"
                )
            
            model_path_obj = Path(model_path)
            if not model_path_obj.exists():
                raise ModelLoadingError(
                    f"Model directory does not exist: {model_path}",
                    error_code="MODEL_PATH_NOT_FOUND",
                    details={"model_path": model_path}
                )
            
            # Check for required model files
            required_files = ['config.json', 'tokenizer.json', 'tokenizer_config.json']
            missing_files = []
            for file_name in required_files:
                if not (model_path_obj / file_name).exists():
                    missing_files.append(file_name)
            
            if missing_files:
                raise ModelLoadingError(
                    f"Missing required model files: {missing_files}",
                    error_code="MISSING_MODEL_FILES",
                    details={"model_path": model_path, "missing_files": missing_files}
                )
            
            # Load model with timeout and retry
            def load_with_retry():
                with timeout_manager.timeout_context(30, "model loading"):
                    self.model_adapter.load_model(model_path)
            
            retry_manager.retry_with_backoff(load_with_retry)
            self.model_loaded = True
            logger.info(f"Successfully loaded MobileBERT model from {model_path}")
            
        except (ModelLoadingError, TimeoutError):
            # Re-raise our custom errors
            self.model_loaded = False
            raise
        except Exception as e:
            # Handle unexpected errors
            self.model_loaded = False
            error_response = global_error_handler.handle_model_loading_error(e, model_path)
            
            if error_response.get('fallback_mode'):
                logger.info("Model loading failed, will use rule-based classification")
            else:
                raise ModelLoadingError(
                    f"Unexpected error loading model: {e}",
                    error_code="UNEXPECTED_MODEL_ERROR",
                    details={"model_path": model_path, "original_error": str(e)}
                )
    
    def predict_heading_level(self, block: ProcessedBlock) -> str:
        """
        Predict the heading level for a processed text block.
        
        Args:
            block: ProcessedBlock to classify
            
        Returns:
            Predicted heading level ('title', 'h1', 'h2', 'h3', 'text')
        """
        try:
            result = self.classify_block(block.features, block.text)
            return result.predicted_class
        except Exception as e:
            logger.error(f"Error predicting heading level: {e}")
            return "text"  # Safe fallback


class MobileBERTAdapter:
    """Adapts pre-trained MobileBERT model for heading classification."""
    
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.device = torch.device('cpu')  # CPU-only as per requirements
        self.class_labels = ['text', 'title', 'h1', 'h2', 'h3']
        self._tokenizer_cache = {}  # Cache tokenized inputs
        self._max_cache_size = 1000
    
    @safe_operation("model loading")
    def load_model(self, model_path: str):
        """
        Load MobileBERT model and tokenizer with optimizations.
        
        Args:
            model_path: Path to the model directory
        """
        model_path = Path(model_path)
        if not model_path.exists():
            raise ModelLoadingError(
                f"Model directory not found: {model_path}",
                error_code="MODEL_DIR_NOT_FOUND",
                details={"model_path": str(model_path)}
            )
        
        try:
            logger.info("Loading MobileBERT tokenizer...")
            # Load tokenizer with optimizations
            self.tokenizer = AutoTokenizer.from_pretrained(
                str(model_path),
                local_files_only=True,
                use_fast=True  # Use fast tokenizer for better performance
            )
            
            logger.info("Loading MobileBERT model...")
            # Load model for sequence classification with optimizations
            self.model = AutoModelForSequenceClassification.from_pretrained(
                str(model_path),
                local_files_only=True,
                num_labels=len(self.class_labels),
                ignore_mismatched_sizes=True,  # Allow size mismatch for adaptation
                torch_dtype=torch.float32,  # Ensure consistent dtype
                low_cpu_mem_usage=True  # Optimize memory usage during loading
            )
            
            # Move to device and optimize for inference
            self.model.to(self.device)
            self.model.eval()
            
            # Optimize model for inference
            if hasattr(torch, 'jit') and hasattr(torch.jit, 'optimize_for_inference'):
                try:
                    self.model = torch.jit.optimize_for_inference(self.model)
                    logger.info("Applied JIT optimization for inference")
                except Exception as e:
                    logger.warning(f"JIT optimization failed: {e}")
            
            # Set number of threads for CPU inference
            torch.set_num_threads(min(4, torch.get_num_threads()))  # Limit threads to avoid overhead
            
            logger.info("MobileBERT model and tokenizer loaded successfully")
            
        except Exception as e:
            raise ModelLoadingError(
                f"Error loading model: {e}",
                error_code="MODEL_LOADING_FAILED",
                details={"model_path": str(model_path), "original_error": str(e)}
            )
    
    @safe_operation("model prediction", fallback_value=("text", 0.1))
    def predict(self, text: str, features: FeatureVector) -> Tuple[str, float]:
        """
        Make prediction using the loaded model with caching optimization.
        
        Args:
            text: Text to classify
            features: Additional features for classification
            
        Returns:
            Tuple of (predicted_class, confidence_score)
        """
        if self.model is None or self.tokenizer is None:
            raise ClassificationError(
                "Model not loaded. Call load_model() first.",
                error_code="MODEL_NOT_LOADED"
            )
        
        try:
            # Prepare input text with feature information
            enhanced_text = self._enhance_text_with_features(text, features)
            
            # Check cache first
            cache_key = hash(enhanced_text)
            if cache_key in self._tokenizer_cache:
                inputs = self._tokenizer_cache[cache_key]
            else:
                # Tokenize input
                inputs = self.tokenizer(
                    enhanced_text,
                    return_tensors="pt",
                    truncation=True,
                    padding=True,
                    max_length=config.get_classification_config().get('max_sequence_length', 512)
                )
                
                # Cache tokenized input if cache not full
                if len(self._tokenizer_cache) < self._max_cache_size:
                    self._tokenizer_cache[cache_key] = inputs
            
            # Move to device
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Make prediction with optimizations
            with torch.no_grad():
                # Use torch.inference_mode for better performance
                with torch.inference_mode():
                    outputs = self.model(**inputs)
                    logits = outputs.logits
                    
                    # Apply softmax to get probabilities
                    probabilities = torch.softmax(logits, dim=-1)
                    
                    # Get prediction and confidence
                    predicted_idx = torch.argmax(probabilities, dim=-1).item()
                    confidence = probabilities[0][predicted_idx].item()
                    
                    predicted_class = self.class_labels[predicted_idx]
                    
                    return predicted_class, confidence
                
        except Exception as e:
            raise ClassificationError(
                f"Prediction failed: {e}",
                error_code="PREDICTION_FAILED",
                details={"text": text[:100], "original_error": str(e)}
            )
    
    @safe_operation("batch prediction", fallback_value=[])
    def predict_batch(self, texts_and_features: List[Tuple[str, FeatureVector]]) -> List[Tuple[str, float]]:
        """
        Make batch predictions for better performance.
        
        Args:
            texts_and_features: List of (text, features) tuples
            
        Returns:
            List of (predicted_class, confidence_score) tuples
        """
        if self.model is None or self.tokenizer is None:
            raise ClassificationError(
                "Model not loaded. Call load_model() first.",
                error_code="MODEL_NOT_LOADED"
            )
        
        if not texts_and_features:
            return []
        
        try:
            # Prepare enhanced texts
            enhanced_texts = [
                self._enhance_text_with_features(text, features)
                for text, features in texts_and_features
            ]
            
            # Batch tokenize
            inputs = self.tokenizer(
                enhanced_texts,
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=config.get_classification_config().get('max_sequence_length', 512)
            )
            
            # Move to device
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Make batch prediction
            with torch.no_grad():
                with torch.inference_mode():
                    outputs = self.model(**inputs)
                    logits = outputs.logits
                    
                    # Apply softmax to get probabilities
                    probabilities = torch.softmax(logits, dim=-1)
                    
                    # Get predictions and confidences
                    predicted_indices = torch.argmax(probabilities, dim=-1)
                    confidences = torch.max(probabilities, dim=-1)[0]
                    
                    results = []
                    for idx, conf in zip(predicted_indices, confidences):
                        predicted_class = self.class_labels[idx.item()]
                        confidence = conf.item()
                        results.append((predicted_class, confidence))
                    
                    return results
                    
        except Exception as e:
            raise ClassificationError(
                f"Batch prediction failed: {e}",
                error_code="BATCH_PREDICTION_FAILED",
                details={"batch_size": len(texts_and_features), "original_error": str(e)}
            )
    
    def _enhance_text_with_features(self, text: str, features: FeatureVector) -> str:
        """
        Enhance text with feature information for better classification.
        
        Args:
            text: Original text
            features: Feature vector
            
        Returns:
            Enhanced text with feature hints
        """
        # Add feature hints to help the model
        hints = []
        
        if features.font_size_ratio > 1.5:
            hints.append("[LARGE_FONT]")
        if features.is_bold:
            hints.append("[BOLD]")
        if features.is_italic:
            hints.append("[ITALIC]")
        if features.position_y < 0.2:  # Top of page
            hints.append("[TOP_POSITION]")
        if features.capitalization_score > 0.8:
            hints.append("[ALL_CAPS]")
        if features.text_length < 50:
            hints.append("[SHORT_TEXT]")
        
        # Combine hints with text
        if hints:
            enhanced_text = " ".join(hints) + " " + text
        else:
            enhanced_text = text
        
        return enhanced_text


class FallbackRuleBasedClassifier:
    """Rule-based fallback classifier for when ML model fails."""
    
    def __init__(self):
        # Enhanced multilingual title patterns
        self.title_patterns = [
            r'^[A-Z\s\d]+$',  # All caps titles - only uppercase letters, spaces, and digits
            r'^(TITLE|SUBJECT|TOPIC|TÍTULO|TITRE|TITEL|TITOLO|ЗАГОЛОВОК|标题|العنوان)[:]\s*',  # Multilingual title markers
            r'^(Abstract|Resumen|Résumé|Zusammenfassung|Riassunto|Аннотация|摘要|الملخص)[:]\s*',  # Abstract markers
            r'^(Introduction|Introducción|Introduction|Einleitung|Introduzione|Введение|介绍|مقدمة)[:]\s*',  # Introduction markers
            r'^(Conclusion|Conclusión|Conclusion|Schlussfolgerung|Conclusione|Заключение|结论|خاتمة)[:]\s*',  # Conclusion markers
        ]
        
        # Enhanced multilingual heading patterns
        self.heading_patterns = [
            r'^\d+\.\s+',  # Simple numbered headings (1. Introduction)
            r'^\d+\.\d+\s+',  # Two-level numbering (1.1 Overview)
            r'^\d+\.\d+\.\d+\s+',  # Three-level numbering (1.1.1 Details)
            r'^[A-Z][a-z]+\s+\d+',  # Chapter/Section patterns
            r'^(Chapter|Section|Part|Capítulo|Chapitre|Kapitel|Capitolo|Глава|章节|فصل)\s+\d+',  # Multilingual chapter/section
            r'^(Appendix|Apéndice|Annexe|Anhang|Appendice|Приложение|附录|ملحق)\s*[A-Z]?',  # Appendix patterns
            r'^[IVX]+\.\s+',  # Roman numerals (I. II. III.)
            r'^[A-Z]\.\s+',  # Letter headings (A. B. C.)
            r'^[a-z]\)\s+',  # Lowercase letter with parenthesis (a) b) c))
            r'^\([a-z]\)\s+',  # Parenthesized lowercase letters ((a) (b) (c))
            r'^\([0-9]+\)\s+',  # Parenthesized numbers ((1) (2) (3))
        ]
        
        # Language-specific patterns for better accuracy
        self.language_specific_patterns = {
            'chinese': [
                r'^第[一二三四五六七八九十\d]+章',  # Chinese chapter markers
                r'^[一二三四五六七八九十]+、',  # Chinese numbered lists
            ],
            'arabic': [
                r'^الفصل\s+\d+',  # Arabic chapter
                r'^القسم\s+\d+',  # Arabic section
            ],
            'japanese': [
                r'^第[一二三四五六七八九十\d]+章',  # Japanese chapter markers
                r'^[一二三四五六七八九十]+、',  # Japanese numbered lists
            ],
            'korean': [
                r'^제\s*\d+\s*장',  # Korean chapter markers
                r'^제\s*\d+\s*절',  # Korean section markers
            ]
        }
    
    @safe_operation("rule-based classification", fallback_value=("text", 0.1))
    def classify(self, text: str, features: FeatureVector) -> Tuple[str, float]:
        """
        Classify text using rule-based approach.
        
        Args:
            text: Text to classify
            features: Feature vector
            
        Returns:
            Tuple of (predicted_class, confidence_score)
        """
        try:
            text_clean = text.strip()
            
            # Rule 1: Check for numbered headings first (more selective)
            if (self._is_numbered_heading(text_clean) and 
                features.text_length > 5 and        # Not too short
                features.text_length < 100 and      # Not too long
                not '\n' in text_clean):            # Single line only
                level = self._determine_heading_level_from_numbering(text_clean)
                return level, 0.6
            
            # Rule 2: Check for title patterns - TEMPORARILY DISABLED
            # if self._is_title(text_clean, features):
            #     return 'title', 0.8
            
            # Rule 3: ONLY very large, bold headings
            if (features.font_size_ratio > 2.0 and  # Very high threshold
                features.is_bold and 
                features.text_length > 10 and       # Meaningful length
                features.text_length < 80 and       # Not too long
                not '\n' in text_clean):            # Single line only
                return 'h1', 0.7
            
            # Rule 4: Title detection - DISABLED
            # if (features.position_y < 0.1 and 
            #     features.text_length > 10 and       # Meaningful length
            #     features.text_length < 60 and       # Not too long
            #     features.font_size_ratio > 1.8 and  # Large font
            #     not '\n' in text_clean):            # Single line only
            #     return 'title', 0.7
            
            # Rule 5: DISABLED
            # if (features.text_length > 10 and       # Meaningful length
            #     features.text_length < 80 and       # Reasonable length
            #     features.is_bold and 
            #     features.font_size_ratio > 1.3 and  # Noticeably larger
            #     features.position_x < 0.1 and       # Left-aligned
            #     not '\n' in text_clean and          # Single line
            #     not any(exclude in text_clean.lower() for exclude in 
            #            ['copyright', 'version', 'notice', 'board', 'international'])):
            #     return 'h3', 0.5
            
            # Default to regular text
            return 'text', 0.9
            
        except Exception as e:
            logger.error(f"Rule-based classification failed: {e}")
            return 'text', 0.1
    
    def classify_text_only(self, text: str) -> Tuple[str, float]:
        """
        Classify text using only text-based rules (no features).
        
        Args:
            text: Text to classify
            
        Returns:
            Tuple of (predicted_class, confidence_score)
        """
        try:
            text_clean = text.strip()
            
            # Check for numbered headings
            if self._is_numbered_heading(text_clean):
                level = self._determine_heading_level_from_numbering(text_clean)
                return level, 0.5
            
            # Check for title patterns
            for pattern in self.title_patterns:
                if re.match(pattern, text_clean, re.IGNORECASE):
                    return 'title', 0.6
            
            # Check for all caps (likely title or heading)
            if text_clean.isupper() and len(text_clean) > 3:
                return 'h1', 0.4
            
            # Default to regular text
            return 'text', 0.8
            
        except Exception as e:
            logger.error(f"Text-only classification failed: {e}")
            return 'text', 0.1
    
    def _is_title(self, text: str, features: FeatureVector) -> bool:
        """Check if text matches title patterns."""
        # Check explicit patterns
        for i, pattern in enumerate(self.title_patterns):
            if i == 0:  # First pattern (all caps) should be case-sensitive
                if re.match(pattern, text):
                    return True
            else:  # Other patterns can be case-insensitive
                if re.match(pattern, text, re.IGNORECASE):
                    return True
        
        # Check characteristics - but not if it's a numbered heading
        # Also require very top position and specific characteristics for title
        if not self._is_numbered_heading(text):
            if (features.font_size_ratio > 2.2 and  # Very large font
                features.position_y < 0.15 and  # Very top of page
                features.text_length < 200 and
                (features.capitalization_score > 0.7 or features.position_x < 0.05)):  # Centered or all caps
                return True
        
        return False
    
    def _is_numbered_heading(self, text: str) -> bool:
        """Check if text is a numbered heading with multilingual support."""
        # Check standard patterns
        for pattern in self.heading_patterns:
            if re.match(pattern, text, re.IGNORECASE):
                return True
        
        # Check language-specific patterns
        for lang_patterns in self.language_specific_patterns.values():
            for pattern in lang_patterns:
                if re.match(pattern, text):
                    return True
        
        return False
    
    def _determine_heading_level_from_numbering(self, text: str) -> str:
        """Determine heading level from numbering pattern."""
        # Count dots in numbering (1.1.1 = h3, 1.1 = h2, 1 = h1)
        match = re.match(r'^(\d+(?:\.\d+)*)', text)
        if match:
            numbering = match.group(1)
            dot_count = numbering.count('.')
            if dot_count == 0:
                return 'h1'
            elif dot_count == 1:
                return 'h2'
            else:
                return 'h3'
        
        return 'h2'  # Default