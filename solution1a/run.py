import os
import time
import json
from typing import Optional, Dict, Any, List

from src.config import INPUT_DIR, OUTPUT_DIR
from src.pdf_parser import parse_pdf
from src.enhanced_document_analyzer_backup import EnhancedDocumentAnalyzer
from src.multilingual_pdf_extractor import MultilingualLanguageDetector
from src.json_builder import build_json

# Optional BERT support
try:
    import torch
    from transformers import pipeline, AutoTokenizer, AutoModelForQuestionAnswering
    BERT_AVAILABLE = True
except ImportError:
    BERT_AVAILABLE = False

LOCAL_MODEL_PATH = "./models/local_mobilebert"

class OptimizedProcessor:
    """Streamlined processor with minimal redundancy"""
    
    def __init__(self, nlp_pipeline=None):
        self.nlp_pipeline = nlp_pipeline
        self.enhanced_analyzer = None
        self.multilingual_extractor = None
        self.language_detector = None
        self.processing_stats = {
            'total_processed': 0,
            'english_docs': 0,
            'multilingual_docs': 0,
            'failed_docs': 0,
            'scanned_docs': 0
        }
    
    def initialize_extractors(self):
        """Initialize extractors with single model loading"""
        try:
            model_path = LOCAL_MODEL_PATH if os.path.exists(LOCAL_MODEL_PATH) else None
            # Use enhanced analyzer for all documents - it's more comprehensive
            self.enhanced_analyzer = EnhancedDocumentAnalyzer(model_path=model_path)
            # Keep multilingual extractor only for language detection
            self.language_detector = MultilingualLanguageDetector()
        except Exception as e:
            # Fallback without BERT
            self.enhanced_analyzer = EnhancedDocumentAnalyzer()
            self.language_detector = MultilingualLanguageDetector()
    
    def detect_language_with_fallback(self, text_blocks: List[Dict], sample_text: str) -> str:
        """Simplified language detection"""
        try:
            return self.language_detector.detect_language(sample_text)
        except:
            return "en"  # Default to English
    
    def process_document(self, extracted_data: Dict, filename: str) -> Dict[str, Any]:
        """Unified document processing using enhanced analyzer"""
        # Enhanced analyzer handles all document types efficiently
        analysis_result = self.enhanced_analyzer.analyze_document(extracted_data)
        
        if not analysis_result.get("title"):
            analysis_result["title"] = f"Document: {os.path.splitext(filename)[0]}"
        
        if not isinstance(analysis_result.get("outline"), list):
            analysis_result["outline"] = []
        
        # Detect language for metadata
        text_blocks = extracted_data.get("text_blocks", [])
        sample_text = " ".join([tb.get("text", "") for tb in text_blocks[:10]]) if text_blocks else ""
        detected_language = self.detect_language_with_fallback(text_blocks, sample_text)
        
        analysis_result["processing_info"] = {
            "analyzer": "EnhancedDocumentAnalyzer",
            "language": detected_language,
            "processing_time": 0
        }
        
        self.processing_stats['total_processed'] += 1
        return analysis_result
    
    def create_scanned_document_result(self, extracted_data: Dict, filename: str) -> Dict[str, Any]:
        result = {
            "title": f"Scanned Document: {os.path.splitext(filename)[0]}",
            "outline": [],
            "document_type": "scanned",
            "has_images": len(extracted_data.get("images", [])) > 0,
            "page_count": len(extracted_data.get("pages", [])),
            "processing_info": {
                "analyzer": "scanned_document_handler",
                "language": "unknown",
                "processing_time": 0.0
            }
        }
        if "page_dimensions" in extracted_data:
            result["page_dimensions"] = extracted_data["page_dimensions"]
        self.processing_stats['scanned_docs'] += 1
        return result
    
    def create_empty_document_result(self, filename: str) -> Dict[str, Any]:
        return {
            "title": f"Empty Document: {os.path.splitext(filename)[0]}",
            "outline": [],
            "document_type": "empty",
            "processing_info": {
                "analyzer": "empty_document_handler",
                "language": "unknown",
                "processing_time": 0.0
            }
        }
    
    def create_error_document_result(self, filename: str, error: Exception) -> Dict[str, Any]:
        error_str = str(error)
        return {
            "title": f"Error processing {filename}",
            "outline": [],
            "error": error_str,
            "processing_error": True,
            "processing_info": {
                "analyzer": "error_handler",
                "language": "unknown",
                "processing_time": 0.0
            }
        }

def load_nlp_model():
    """Load NLP model if available"""
    if not BERT_AVAILABLE or not os.path.isdir(LOCAL_MODEL_PATH):
        return None
    try:
        tokenizer = AutoTokenizer.from_pretrained(LOCAL_MODEL_PATH, use_fast=True, local_files_only=True)
        model = AutoModelForQuestionAnswering.from_pretrained(LOCAL_MODEL_PATH, local_files_only=True, torch_dtype=torch.float32)
        nlp_pipeline = pipeline('question-answering', model=model, tokenizer=tokenizer, device=-1)
        return nlp_pipeline
    except:
        return None

def process_single_pdf(pdf_path: str, output_path: str, processor: OptimizedProcessor) -> bool:
    filename = os.path.basename(pdf_path)
    try:
        extracted_data = parse_pdf(pdf_path, preset="heading_extraction")
        text_blocks = extracted_data.get("text_blocks", [])
        
        if not text_blocks:
            if extracted_data.get("likely_document_type") == "scanned":
                analysis_result = processor.create_scanned_document_result(extracted_data, filename)
            else:
                analysis_result = processor.create_empty_document_result(filename)
        else:
            # Use unified processing for all documents
            analysis_result = processor.process_document(extracted_data, filename)
        
        build_json(analysis_result, output_path)
        return True
    except Exception as e:
        try:
            fallback_result = processor.create_error_document_result(filename, e)
            build_json(fallback_result, output_path)
            processor.processing_stats['failed_docs'] += 1
            return False
        except:
            processor.processing_stats['failed_docs'] += 1
            return False

def print_processing_summary(processor: OptimizedProcessor, total_files: int, total_time: float):
    stats = processor.processing_stats
    print("\nSummary:")
    print(f"Total: {total_files}")
    print(f"Success: {stats['total_processed']}")
    print(f"Failed: {stats['failed_docs']}")
    print(f"English: {stats['english_docs']}")
    print(f"Multilingual: {stats['multilingual_docs']}")
    print(f"Scanned: {stats['scanned_docs']}")
    print(f"Time: {total_time:.2f}s")
    if total_files > 0:
        print(f"Avg per file: {total_time/total_files:.2f}s")

def main():
    """Main processing function with Docker support."""
    print("Starting PDF processing...")
    
    # Check for Docker environment (/app/input)
    docker_input = "/app/input"
    docker_output = "/app/output"
    
    # Use Docker paths if available, otherwise use local paths
    if os.path.exists(docker_input) and os.path.isdir(docker_input):
        input_dir = docker_input
        output_dir = docker_output
        print("🐳 Running in Docker environment")
    else:
        input_dir = INPUT_DIR
        output_dir = OUTPUT_DIR
        print("💻 Running in local environment")
    
    nlp_pipeline = load_nlp_model()
    processor = OptimizedProcessor(nlp_pipeline)
    processor.initialize_extractors()
    
    if not os.path.isdir(input_dir):
        print(f"Missing input folder: {input_dir}")
        return
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    pdf_files = [f for f in os.listdir(input_dir) if f.lower().endswith(".pdf") and not f.startswith('.')]
    if not pdf_files:
        print(f"No PDFs found in {input_dir}")
        return
    
    pdf_files.sort()
    total_start_time = time.time()
    
    for i, pdf_filename in enumerate(pdf_files, 1):
        print(f"[{i}/{len(pdf_files)}] {pdf_filename}... ", end="")
        pdf_path = os.path.join(input_dir, pdf_filename)
        json_filename = os.path.splitext(pdf_filename)[0] + ".json"
        json_path = os.path.join(output_dir, json_filename)
        success = process_single_pdf(pdf_path, json_path, processor)
        print("✅" if success else "❌")
    
    total_time = time.time() - total_start_time
    print_processing_summary(processor, len(pdf_files), total_time)

if __name__ == "__main__":
    main()
