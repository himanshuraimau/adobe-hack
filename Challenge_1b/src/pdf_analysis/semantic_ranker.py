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
        """Initialize sentence-transformer model (all-MiniLM-L6-v2)."""
        try:
            logging.info(f"Loading sentence transformer model: {self.model_name}")
            self.model = SentenceTransformer(self.model_name)
            logging.info("Model loaded successfully")
        except Exception as e:
            logging.error(f"Failed to load model {self.model_name}: {e}")
            raise RuntimeError(f"Could not load sentence transformer model: {e}")
    
    def create_query_embedding(self, persona: Dict[str, str], job_to_be_done: Dict[str, str]) -> np.ndarray:
        """Combine persona and task into query embedding."""
        if self.model is None:
            self.load_model()
        
        # Extract persona role and job task
        persona_role = persona.get("role", "")
        job_task = job_to_be_done.get("task", "")
        
        # Combine persona and task into a single query string
        query_text = f"{persona_role} {job_task}".strip()
        
        if not query_text:
            raise ValueError("Both persona role and job task must be provided")
        
        logging.info(f"Creating query embedding for: '{query_text}'")
        
        # Generate embedding for the combined query
        query_embedding = self.model.encode([query_text])
        return query_embedding[0]  # Return single embedding vector
    
    def _generate_section_embeddings(self, sections: List[Section]) -> List[np.ndarray]:
        """Generate embeddings for all sections."""
        if self.model is None:
            self.load_model()
        
        # Combine section title and content for better semantic representation
        section_texts = []
        for section in sections:
            # Combine title and content, giving more weight to title
            combined_text = f"{section.section_title}. {section.content}"
            section_texts.append(combined_text)
        
        logging.info(f"Generating embeddings for {len(section_texts)} sections")
        
        # Generate embeddings for all sections at once for efficiency
        embeddings = self.model.encode(section_texts)
        return embeddings
    
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
        """Perform sentence-level analysis on top-ranked sections."""
        if self.model is None:
            self.load_model()
        
        if not top_sections:
            return []
        
        sentence_analysis = []
        
        for section in top_sections:
            # Split section content into sentences
            # Simple sentence splitting - could be enhanced with NLTK
            sentences = self._split_into_sentences(section.content)
            
            if not sentences:
                continue
            
            # Generate embeddings for sentences
            sentence_embeddings = self.model.encode(sentences)
            
            # Calculate similarity scores for sentences
            similarities = cosine_similarity([query_embedding], sentence_embeddings)[0]
            
            # Create sentence-score pairs and sort by relevance
            sentence_scores = list(zip(sentences, similarities))
            sentence_scores.sort(key=lambda x: x[1], reverse=True)
            
            # Take top sentences up to max_sentences
            top_sentences = sentence_scores[:max_sentences]
            
            # Compile refined text from top sentences
            refined_text = " ".join([sentence for sentence, _ in top_sentences])
            
            sentence_analysis.append({
                "document": section.document,
                "refined_text": refined_text,
                "page_number": section.page_number,
                "sentence_count": len(top_sentences),
                "avg_similarity": np.mean([score for _, score in top_sentences])
            })
        
        logging.info(f"Completed sentence-level analysis for {len(top_sections)} sections")
        
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