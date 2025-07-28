"""Semantic ranking system for PDF content analysis."""

import numpy as np
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import logging

from .pdf_analyzer import Section


@dataclass
class RankedSection:
    """Represents a document section with ranking information."""
    document: str
    section_title: str
    importance_rank: int
    page_number: int
    similarity_score: float


class SemanticRanker:
    """Handles embedding generation and similarity scoring for content ranking."""
    
    def __init__(self):
        self.model = None
        self.model_name = "all-MiniLM-L6-v2"
        
    def load_model(self) -> None:
        """Initialize sentence-transformer model (all-MiniLM-L6-v2) with error handling."""
        if self.model is not None:
            logging.debug("Model already loaded")
            return
            
        try:
            logging.info(f"Loading sentence transformer model: {self.model_name}")
            
            # Check available memory before loading
            try:
                import psutil
                available_memory_gb = psutil.virtual_memory().available / (1024**3)
                if available_memory_gb < 1:
                    logging.warning(f"Low available memory: {available_memory_gb:.1f}GB. Model loading may fail.")
            except ImportError:
                logging.debug("psutil not available for memory check")
            
            # Load model with CPU-only configuration
            self.model = SentenceTransformer(self.model_name, device='cpu')
            
            # Verify model is working
            test_embedding = self.model.encode(["test"])
            if test_embedding is None or len(test_embedding) == 0:
                raise RuntimeError("Model loaded but failed to generate test embedding")
            
            logging.info(f"Model loaded successfully. Embedding dimension: {len(test_embedding[0])}")
            
        except ImportError as e:
            logging.error(f"Missing required dependencies for model {self.model_name}: {e}")
            raise RuntimeError(f"Could not import sentence transformer dependencies: {e}")
        except Exception as e:
            logging.error(f"Failed to load model {self.model_name}: {e}")
            logging.debug("Model loading error details", exc_info=True)
            raise RuntimeError(f"Could not load sentence transformer model: {e}")
    
    def create_query_embedding(self, persona: Dict[str, str], job_to_be_done: Dict[str, str]) -> np.ndarray:
        """Combine persona and task into query embedding with error handling."""
        if self.model is None:
            self.load_model()
        
        try:
            # Extract persona role and job task with validation
            persona_role = persona.get("role", "") if isinstance(persona, dict) else ""
            job_task = job_to_be_done.get("task", "") if isinstance(job_to_be_done, dict) else ""
            
            # Combine persona and task into a single query string
            query_text = f"{persona_role} {job_task}".strip()
            
            if not query_text:
                raise ValueError("Both persona role and job task must be provided and non-empty")
            
            if len(query_text) > 1000:  # Reasonable limit for query length
                logging.warning(f"Query text is very long ({len(query_text)} chars), truncating")
                query_text = query_text[:1000]
            
            logging.info(f"Creating query embedding for: '{query_text}'")
            
            # Generate embedding for the combined query
            query_embedding = self.model.encode([query_text], show_progress_bar=False)
            
            if query_embedding is None or len(query_embedding) == 0:
                raise RuntimeError("Failed to generate query embedding")
            
            return query_embedding[0]  # Return single embedding vector
            
        except Exception as e:
            logging.error(f"Error creating query embedding: {e}")
            raise RuntimeError(f"Failed to create query embedding: {e}")
    
    def _generate_section_embeddings(self, sections: List[Section], batch_size: int = 32) -> List[np.ndarray]:
        """Generate embeddings for all sections with batch processing for memory efficiency."""
        if self.model is None:
            self.load_model()
        
        if not sections:
            return []
        
        # Combine section title and content for better semantic representation
        section_texts = []
        for section in sections:
            # Combine title and content, giving more weight to title
            combined_text = f"{section.section_title}. {section.content}"
            # Truncate very long texts to prevent memory issues
            if len(combined_text) > 5000:
                logging.debug(f"Truncating long section text ({len(combined_text)} chars)")
                combined_text = combined_text[:5000] + "..."
            section_texts.append(combined_text)
        
        logging.info(f"Generating embeddings for {len(section_texts)} sections")
        
        try:
            # Process in batches for memory efficiency
            all_embeddings = []
            total_batches = (len(section_texts) + batch_size - 1) // batch_size
            
            for batch_idx in range(total_batches):
                start_idx = batch_idx * batch_size
                end_idx = min(start_idx + batch_size, len(section_texts))
                batch_texts = section_texts[start_idx:end_idx]
                
                logging.debug(f"Processing embedding batch {batch_idx + 1}/{total_batches}")
                
                try:
                    batch_embeddings = self.model.encode(
                        batch_texts, 
                        show_progress_bar=False,
                        convert_to_numpy=True
                    )
                    all_embeddings.extend(batch_embeddings)
                    
                except Exception as e:
                    logging.error(f"Error processing embedding batch {batch_idx + 1}: {e}")
                    # Create zero embeddings for failed batch to maintain alignment
                    embedding_dim = 384  # all-MiniLM-L6-v2 dimension
                    zero_embeddings = [np.zeros(embedding_dim) for _ in batch_texts]
                    all_embeddings.extend(zero_embeddings)
            
            if len(all_embeddings) != len(sections):
                raise RuntimeError(f"Embedding count mismatch: {len(all_embeddings)} != {len(sections)}")
            
            return all_embeddings
            
        except Exception as e:
            logging.error(f"Error generating section embeddings: {e}")
            raise RuntimeError(f"Failed to generate section embeddings: {e}")
    
    def rank_sections(self, query_embedding: np.ndarray, sections: List[Section]) -> List[RankedSection]:
        """Rank sections by cosine similarity to query embedding."""
        if not sections:
            return []
        
        # Generate embeddings for all sections
        section_embeddings = self._generate_section_embeddings(sections)
        
        # Calculate cosine similarity between query and each section
        similarities = cosine_similarity([query_embedding], section_embeddings)[0]
        
        # Create ranked sections with similarity scores
        ranked_sections = []
        for i, (section, similarity_score) in enumerate(zip(sections, similarities)):
            ranked_sections.append(RankedSection(
                document=section.document,
                section_title=section.section_title,
                importance_rank=0,  # Will be set after sorting
                page_number=section.page_number,
                similarity_score=float(similarity_score)
            ))
        
        # Sort by similarity score (highest first) and assign ranks
        ranked_sections.sort(key=lambda x: x.similarity_score, reverse=True)
        
        # Assign importance ranks (1 = highest relevance)
        for rank, section in enumerate(ranked_sections, 1):
            section.importance_rank = rank
        
        logging.info(f"Ranked {len(ranked_sections)} sections by semantic similarity")
        
        return ranked_sections
    
    def analyze_sentences(self, query_embedding: np.ndarray, top_sections: List[Section], max_sentences: int = 10) -> List[Dict[str, Any]]:
        """Perform sentence-level analysis on top-ranked sections with error handling."""
        if self.model is None:
            self.load_model()
        
        if not top_sections:
            return []
        
        sentence_analysis = []
        failed_sections = 0
        
        logging.info(f"Starting sentence-level analysis for {len(top_sections)} sections")
        
        for section_idx, section in enumerate(top_sections):
            try:
                logging.debug(f"Analyzing sentences for section {section_idx + 1}/{len(top_sections)}: {section.section_title}")
                
                # Split section content into sentences
                sentences = self._split_into_sentences(section.content)
                
                if not sentences:
                    logging.debug(f"No sentences found in section: {section.section_title}")
                    continue
                
                # Limit number of sentences to process for memory efficiency
                if len(sentences) > 100:
                    logging.debug(f"Limiting sentences from {len(sentences)} to 100 for section: {section.section_title}")
                    sentences = sentences[:100]
                
                try:
                    # Generate embeddings for sentences with batch processing
                    sentence_embeddings = self.model.encode(
                        sentences, 
                        show_progress_bar=False,
                        convert_to_numpy=True,
                        batch_size=16  # Smaller batch size for sentences
                    )
                    
                    # Calculate similarity scores for sentences
                    similarities = cosine_similarity([query_embedding], sentence_embeddings)[0]
                    
                    # Create sentence-score pairs and sort by relevance
                    sentence_scores = list(zip(sentences, similarities))
                    sentence_scores.sort(key=lambda x: x[1], reverse=True)
                    
                    # Take top sentences up to max_sentences
                    top_sentences = sentence_scores[:max_sentences]
                    
                    if not top_sentences:
                        logging.debug(f"No top sentences found for section: {section.section_title}")
                        continue
                    
                    # Compile refined text from top sentences
                    refined_text = " ".join([sentence for sentence, _ in top_sentences])
                    
                    # Calculate average similarity score
                    avg_similarity = np.mean([score for _, score in top_sentences])
                    
                    sentence_analysis.append({
                        "document": section.document,
                        "refined_text": refined_text,
                        "page_number": section.page_number,
                        "sentence_count": len(top_sentences),
                        "avg_similarity": float(avg_similarity)
                    })
                    
                    logging.debug(f"Processed {len(sentences)} sentences, selected top {len(top_sentences)} for section: {section.section_title}")
                    
                except Exception as e:
                    logging.error(f"Error processing sentences for section '{section.section_title}': {e}")
                    failed_sections += 1
                    continue
                    
            except Exception as e:
                logging.error(f"Error analyzing section {section_idx + 1} '{section.section_title}': {e}")
                failed_sections += 1
                continue
        
        if failed_sections > 0:
            logging.warning(f"Failed to analyze {failed_sections}/{len(top_sections)} sections")
        
        logging.info(f"Completed sentence-level analysis: {len(sentence_analysis)} sections processed successfully")
        
        return sentence_analysis
    
    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences using simple heuristics."""
        if not text:
            return []
        
        # Simple sentence splitting based on punctuation
        import re
        
        # Split on sentence-ending punctuation followed by whitespace and capital letter
        sentences = re.split(r'[.!?]+\s+(?=[A-Z])', text)
        
        # Clean up sentences
        cleaned_sentences = []
        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) > 10:  # Filter out very short fragments
                # Ensure sentence ends with punctuation
                if not sentence.endswith(('.', '!', '?')):
                    sentence += '.'
                cleaned_sentences.append(sentence)
        
        return cleaned_sentences