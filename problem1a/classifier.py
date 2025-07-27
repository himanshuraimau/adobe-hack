"""
Heading classification module using MobileBERT for text classification.

This module adapts the local MobileBERT model for heading classification,
combining textual content with extracted features to predict heading levels.
"""

import logging
import re
import torch
from pathlib import Path
from typing import Optional, Tuple, Dict, Any
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import numpy as np

from models import ProcessedBlock, ClassificationResult, FeatureVector
from config import config

logger = logging.getLogger(__name__)


class HeadingClassifier:
    """Main classification logic using MobileBERT for heading detection."""
    
    def __init__(self):
        self.config = config.get_classification_config()
        self.model_adapter = MobileBERTAdapter()
        self.model_loaded = False
        self.fallback_classifier = FallbackRuleBasedClassifier()
    
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
            if self.model_loaded:
                predicted_class, confidence = self.model_adapter.predict(text, features)
                
                # Use fallback if confidence is too low
                if confidence < self.config.get('confidence_threshold', 0.5):
                    if self.config.get('use_fallback_rules', True):
                        predicted_class, confidence = self.fallback_classifier.classify(text, features)
                        logger.debug(f"Used fallback classification for: {text[:50]}...")
            else:
                # Use fallback if model not loaded
                predicted_class, confidence = self.fallback_classifier.classify(text, features)
                logger.warning("Model not loaded, using fallback classification")
            
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
            
        except Exception as e:
            logger.error(f"Classification failed: {e}")
            # Fallback to rule-based classification
            predicted_class, confidence = self.fallback_classifier.classify(text, features)
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
    
    def load_model(self, model_path: str) -> None:
        """
        Load the MobileBERT model from the specified path.
        
        Args:
            model_path: Path to the MobileBERT model directory
        """
        try:
            self.model_adapter.load_model(model_path)
            self.model_loaded = True
            logger.info(f"Successfully loaded MobileBERT model from {model_path}")
        except Exception as e:
            logger.error(f"Failed to load model from {model_path}: {e}")
            self.model_loaded = False
            logger.info("Will use fallback rule-based classification")
    
    def predict_heading_level(self, block: ProcessedBlock) -> str:
        """
        Predict the heading level for a processed text block.
        
        Args:
            block: ProcessedBlock to classify
            
        Returns:
            Predicted heading level ('title', 'h1', 'h2', 'h3', 'text')
        """
        result = self.classify_block(block.features, block.text)
        return result.predicted_class


class MobileBERTAdapter:
    """Adapts pre-trained MobileBERT model for heading classification."""
    
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.device = torch.device('cpu')  # CPU-only as per requirements
        self.class_labels = ['text', 'title', 'h1', 'h2', 'h3']
    
    def load_model(self, model_path: str):
        """
        Load MobileBERT model and tokenizer.
        
        Args:
            model_path: Path to the model directory
        """
        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"Model directory not found: {model_path}")
        
        try:
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                str(model_path),
                local_files_only=True
            )
            
            # Load model for sequence classification
            # Since we don't have a fine-tuned model, we'll adapt the base model
            self.model = AutoModelForSequenceClassification.from_pretrained(
                str(model_path),
                local_files_only=True,
                num_labels=len(self.class_labels),
                ignore_mismatched_sizes=True  # Allow size mismatch for adaptation
            )
            
            self.model.to(self.device)
            self.model.eval()
            
            logger.info("MobileBERT model and tokenizer loaded successfully")
            
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise
    
    def predict(self, text: str, features: FeatureVector) -> Tuple[str, float]:
        """
        Make prediction using the loaded model.
        
        Args:
            text: Text to classify
            features: Additional features for classification
            
        Returns:
            Tuple of (predicted_class, confidence_score)
        """
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        try:
            # Prepare input text with feature information
            enhanced_text = self._enhance_text_with_features(text, features)
            
            # Tokenize input
            inputs = self.tokenizer(
                enhanced_text,
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=config.get_classification_config().get('max_sequence_length', 512)
            )
            
            # Move to device
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Make prediction
            with torch.no_grad():
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
            logger.error(f"Prediction failed: {e}")
            raise
    
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
        self.title_patterns = [
            r'^[A-Z\s\d]+$',  # All caps titles - only uppercase letters, spaces, and digits
            r'^(TITLE|SUBJECT|TOPIC)[:]\s*',  # Explicit title markers
        ]
        
        self.heading_patterns = [
            r'^\d+\.\s+',  # Simple numbered headings (1. Introduction)
            r'^\d+\.\d+\s+',  # Two-level numbering (1.1 Overview)
            r'^\d+\.\d+\.\d+\s+',  # Three-level numbering (1.1.1 Details)
            r'^[A-Z][a-z]+\s+\d+',  # Chapter/Section patterns
            r'^(Chapter|Section|Part)\s+\d+',  # Explicit chapter/section
        ]
    
    def classify(self, text: str, features: FeatureVector) -> Tuple[str, float]:
        """
        Classify text using rule-based approach.
        
        Args:
            text: Text to classify
            features: Feature vector
            
        Returns:
            Tuple of (predicted_class, confidence_score)
        """
        text_clean = text.strip()
        
        # Rule 1: Check for numbered headings first (more specific)
        if self._is_numbered_heading(text_clean):
            level = self._determine_heading_level_from_numbering(text_clean)
            return level, 0.6
        
        # Rule 2: Check for title patterns
        if self._is_title(text_clean, features):
            return 'title', 0.8
        
        # Rule 3: Check font size and formatting for headings
        if features.font_size_ratio > 1.3:
            if features.is_bold or features.capitalization_score > 0.5:
                # Determine heading level based on font size
                if features.font_size_ratio > 2.0:
                    return 'h1', 0.7
                elif features.font_size_ratio > 1.6:
                    return 'h2', 0.7
                else:
                    return 'h3', 0.7
        
        # Rule 4: Position-based classification for titles
        if features.position_y < 0.1 and features.text_length < 100:
            if features.font_size_ratio > 1.1:
                return 'title', 0.6
        
        # Rule 5: Short, bold text at beginning of line
        if (features.text_length < 80 and 
            features.is_bold and 
            features.position_x < 0.2):
            return 'h3', 0.5
        
        # Default to regular text
        return 'text', 0.9
    
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
        """Check if text is a numbered heading."""
        for pattern in self.heading_patterns:
            if re.match(pattern, text, re.IGNORECASE):
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