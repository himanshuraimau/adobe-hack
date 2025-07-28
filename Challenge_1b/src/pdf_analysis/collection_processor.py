"""Collection discovery and processing orchestration module."""

import os
import json
import logging
from typing import List, Dict, Any, Optional
from pathlib import Path
from dataclasses import dataclass

from .pdf_analyzer import PDFAnalyzer, Section
from .semantic_ranker import SemanticRanker, RankedSection
from .output_generator import OutputGenerator


@dataclass
class InputConfig:
    """Represents the input configuration from challenge1b_input.json."""
    challenge_info: Dict[str, str]
    documents: List[Dict[str, str]]
    persona: Dict[str, str]
    job_to_be_done: Dict[str, str]


@dataclass
class CollectionResult:
    """Represents the processing result for a single collection."""
    collection_path: str
    success: bool
    error_message: Optional[str] = None
    sections_processed: int = 0
    documents_processed: int = 0


class CollectionProcessor:
    """Main orchestrator that discovers and processes each collection."""
    
    def __init__(self, batch_size: int = 5):
        self.pdf_analyzer = PDFAnalyzer()
        self.semantic_ranker = SemanticRanker()
        self.output_generator = OutputGenerator()
        self.input_filename = "challenge1b_input.json"
        self.output_filename = "challenge1b_output.json"
        self.batch_size = batch_size  # Process PDFs in batches for memory efficiency
        self.logger = logging.getLogger(__name__)
        
    def discover_collections(self, base_path: str = ".") -> List[str]:
        """Find all Collection directories in the base path."""
        collection_paths = []
        
        try:
            # Look for directories that start with "Collection"
            for item in os.listdir(base_path):
                item_path = os.path.join(base_path, item)
                if os.path.isdir(item_path) and item.startswith("Collection"):
                    collection_paths.append(item_path)
            
            # Sort collections for consistent processing order
            collection_paths.sort()
            
            logging.info(f"Discovered {len(collection_paths)} collections: {collection_paths}")
            
        except Exception as e:
            logging.error(f"Error discovering collections in {base_path}: {e}")
            raise RuntimeError(f"Failed to discover collections: {e}")
        
        return collection_paths
    
    def _load_input_config(self, collection_path: str) -> InputConfig:
        """Load and parse the challenge1b_input.json file from a collection."""
        input_file_path = os.path.join(collection_path, self.input_filename)
        
        if not os.path.exists(input_file_path):
            raise FileNotFoundError(f"Input file not found: {input_file_path}")
        
        try:
            with open(input_file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Validate required fields
            required_fields = ['challenge_info', 'documents', 'persona', 'job_to_be_done']
            for field in required_fields:
                if field not in data:
                    raise ValueError(f"Missing required field '{field}' in input config")
            
            config = InputConfig(
                challenge_info=data['challenge_info'],
                documents=data['documents'],
                persona=data['persona'],
                job_to_be_done=data['job_to_be_done']
            )
            
            logging.info(f"Loaded input config from {input_file_path}")
            logging.info(f"  - Persona: {config.persona.get('role', 'Unknown')}")
            logging.info(f"  - Task: {config.job_to_be_done.get('task', 'Unknown')}")
            logging.info(f"  - Documents: {len(config.documents)}")
            
            return config
            
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in {input_file_path}: {e}")
        except Exception as e:
            raise RuntimeError(f"Error loading input config from {input_file_path}: {e}")
    
    def _find_pdf_files(self, collection_path: str, document_filenames: List[str]) -> List[str]:
        """Find PDF files in the collection's PDFs directory."""
        pdf_dir = os.path.join(collection_path, "PDFs")
        
        if not os.path.exists(pdf_dir):
            raise FileNotFoundError(f"PDFs directory not found: {pdf_dir}")
        
        found_pdfs = []
        missing_pdfs = []
        
        for filename in document_filenames:
            pdf_path = os.path.join(pdf_dir, filename)
            if os.path.exists(pdf_path):
                found_pdfs.append(pdf_path)
            else:
                missing_pdfs.append(filename)
        
        if missing_pdfs:
            logging.warning(f"Missing PDF files in {pdf_dir}: {missing_pdfs}")
        
        if not found_pdfs:
            raise FileNotFoundError(f"No PDF files found in {pdf_dir}")
        
        logging.info(f"Found {len(found_pdfs)} PDF files in {pdf_dir}")
        return found_pdfs
    
    def _process_pdf_batch(self, pdf_paths: List[str], batch_start: int, batch_end: int) -> List[Section]:
        """Process a batch of PDF files for memory efficiency."""
        batch_sections = []
        batch_paths = pdf_paths[batch_start:batch_end]
        
        self.logger.info(f"Processing PDF batch {batch_start//self.batch_size + 1}: "
                        f"{len(batch_paths)} files")
        
        for pdf_path in batch_paths:
            try:
                self.logger.debug(f"Extracting sections from: {pdf_path}")
                sections = self.pdf_analyzer.get_section_content(pdf_path)
                batch_sections.extend(sections)
                self.logger.debug(f"Extracted {len(sections)} sections from {pdf_path}")
                
            except FileNotFoundError as e:
                self.logger.error(f"PDF file not found: {pdf_path} - {e}")
                continue
            except PermissionError as e:
                self.logger.error(f"Permission denied accessing PDF: {pdf_path} - {e}")
                continue
            except Exception as e:
                self.logger.error(f"Error processing PDF {pdf_path}: {type(e).__name__}: {e}")
                # Log more details for debugging
                self.logger.debug(f"PDF processing error details for {pdf_path}", exc_info=True)
                continue
        
        return batch_sections

    def process_collection(self, collection_path: str) -> CollectionResult:
        """Process a single collection directory with comprehensive error handling."""
        self.logger.info(f"Processing collection: {collection_path}")
        
        try:
            # Load input configuration with validation
            try:
                config = self._load_input_config(collection_path)
            except FileNotFoundError as e:
                raise RuntimeError(f"Input configuration missing: {e}")
            except (json.JSONDecodeError, ValueError) as e:
                raise RuntimeError(f"Invalid input configuration: {e}")
            
            # Find PDF files with error handling
            try:
                document_filenames = [doc['filename'] for doc in config.documents]
                pdf_paths = self._find_pdf_files(collection_path, document_filenames)
            except FileNotFoundError as e:
                raise RuntimeError(f"PDF directory or files not found: {e}")
            
            # Process PDFs in batches for memory efficiency
            all_sections = []
            documents_processed = 0
            total_batches = (len(pdf_paths) + self.batch_size - 1) // self.batch_size
            
            self.logger.info(f"Processing {len(pdf_paths)} PDFs in {total_batches} batches")
            
            for batch_num in range(total_batches):
                batch_start = batch_num * self.batch_size
                batch_end = min(batch_start + self.batch_size, len(pdf_paths))
                
                try:
                    batch_sections = self._process_pdf_batch(pdf_paths, batch_start, batch_end)
                    all_sections.extend(batch_sections)
                    documents_processed += len(pdf_paths[batch_start:batch_end])
                    
                    self.logger.info(f"Batch {batch_num + 1}/{total_batches} complete: "
                                   f"{len(batch_sections)} sections extracted")
                    
                except Exception as e:
                    self.logger.error(f"Error processing batch {batch_num + 1}: {e}")
                    # Continue with next batch
                    continue
            
            if not all_sections:
                raise RuntimeError("No sections extracted from any PDF files")
            
            self.logger.info(f"Total sections extracted: {len(all_sections)}")
            
            # Create query embedding with error handling
            try:
                query_embedding = self.semantic_ranker.create_query_embedding(
                    config.persona, 
                    config.job_to_be_done
                )
            except Exception as e:
                raise RuntimeError(f"Failed to create query embedding: {e}")
            
            # Rank sections by semantic similarity
            try:
                ranked_sections = self.semantic_ranker.rank_sections(query_embedding, all_sections)
                self.logger.info(f"Ranked {len(ranked_sections)} sections by relevance")
            except Exception as e:
                raise RuntimeError(f"Failed to rank sections: {e}")
            
            # Perform sentence-level analysis on top 5-10 sections (Requirement 4.1)
            top_sections_count = min(10, max(5, len(ranked_sections)))
            top_sections = []
            
            # Get the original Section objects for the top-ranked sections
            for ranked_section in ranked_sections[:top_sections_count]:
                # Find the corresponding original Section object
                for section in all_sections:
                    if (section.document == ranked_section.document and 
                        section.section_title == ranked_section.section_title and
                        section.page_number == ranked_section.page_number):
                        top_sections.append(section)
                        break
            
            self.logger.info(f"Selected top {len(top_sections)} sections for detailed analysis")
            
            # Perform sentence-level analysis (Requirements 4.2, 4.3, 4.4)
            try:
                sentence_analysis = self.semantic_ranker.analyze_sentences(query_embedding, top_sections)
                self.logger.info(f"Completed sentence analysis for {len(sentence_analysis)} sections")
            except Exception as e:
                raise RuntimeError(f"Failed to perform sentence analysis: {e}")
            
            # Generate and save output JSON (Requirements 5.1, 5.2, 5.3, 5.4)
            try:
                input_data = {
                    'documents': config.documents,
                    'persona': config.persona,
                    'job_to_be_done': config.job_to_be_done
                }
                
                output_file_path = self.output_generator.generate_and_save_output(
                    collection_path, input_data, ranked_sections, sentence_analysis
                )
                
                self.logger.info(f"Output saved to: {output_file_path}")
                
            except Exception as e:
                raise RuntimeError(f"Failed to generate or save output: {e}")
            
            # Log success summary
            self.logger.info(f"Successfully processed collection {collection_path}")
            self.logger.info(f"  - Documents processed: {documents_processed}")
            self.logger.info(f"  - Sections extracted: {len(all_sections)}")
            self.logger.info(f"  - Sections ranked: {len(ranked_sections)}")
            self.logger.info(f"  - Top sections analyzed: {len(top_sections)}")
            self.logger.info(f"  - Sentence analysis results: {len(sentence_analysis)}")
            
            return CollectionResult(
                collection_path=collection_path,
                success=True,
                sections_processed=len(all_sections),
                documents_processed=documents_processed
            )
            
        except Exception as e:
            error_msg = f"Error processing collection {collection_path}: {e}"
            self.logger.error(error_msg)
            self.logger.debug("Collection processing error details", exc_info=True)
            
            return CollectionResult(
                collection_path=collection_path,
                success=False,
                error_message=str(e)
            )
    
    def process_all_collections(self, base_path: str = ".") -> List[CollectionResult]:
        """Main processing loop that handles each collection independently."""
        logging.info("Starting collection processing")
        
        # Discover all collections
        collection_paths = self.discover_collections(base_path)
        
        if not collection_paths:
            logging.warning("No collections found to process")
            return []
        
        # Process each collection independently
        results = []
        successful = 0
        
        for collection_path in collection_paths:
            result = self.process_collection(collection_path)
            results.append(result)
            
            if result.success:
                successful += 1
            else:
                logging.error(f"Failed to process {collection_path}: {result.error_message}")
        
        # Log summary
        total = len(collection_paths)
        logging.info(f"Collection processing complete: {successful}/{total} collections processed successfully")
        
        return results