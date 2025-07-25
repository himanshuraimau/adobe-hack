# run.py

import os
import time
import torch
from transformers import pipeline, AutoTokenizer, AutoModelForQuestionAnswering

# --- Local Imports from your 'src' folder ---
from src.config import INPUT_DIR, OUTPUT_DIR
from src.pdf_parser import parse_pdf
from src.enhanced_document_analyzer import EnhancedDocumentAnalyzer
from src.json_builder import build_json

# --- Configuration for Offline Model Loading ---
# This path points to the folder where the model was saved.
# It's crucial that this folder is included with your submission.
LOCAL_MODEL_PATH = "./local_mobilebert"

def load_nlp_model():
    """
    Loads the MobileBERT model and tokenizer from the LOCAL FOLDER with optimizations.
    This function is designed to work entirely offline with performance improvements.
    """
    print(f"Loading NLP model from local path: '{LOCAL_MODEL_PATH}'...")

    # Step 1: Critical Check - Ensure the local model directory exists before proceeding.
    if not os.path.isdir(LOCAL_MODEL_PATH):
        print(f"\nFATAL ERROR: Local model directory not found at '{LOCAL_MODEL_PATH}'")
        print("Please ensure you have run the 'download_model.py' script with an internet connection")
        print("to download the model files into your project folder.\n")
        return None

    # Step 2: Load the model and tokenizer with optimizations
    try:
        # Check for a GPU for faster processing, otherwise default to CPU.
        device = -1 # Force CPU as per requirements
        
        print(f"Using device: CPU")
        
        # Load tokenizer with optimizations
        tokenizer = AutoTokenizer.from_pretrained(
            LOCAL_MODEL_PATH,
            use_fast=True,  # Use fast tokenizer if available
            local_files_only=True  # Ensure offline mode
        )
        
        # Load model with optimizations
        model = AutoModelForQuestionAnswering.from_pretrained(
            LOCAL_MODEL_PATH,
            local_files_only=True  # Ensure offline mode
        )
        
        # Initialize the pipeline for question-answering (matching the model type)
        try:
            nlp_pipeline = pipeline(
                'question-answering',
                model=model,
                tokenizer=tokenizer,
                device=device,
                max_length=256,
                truncation=True,
                padding=True,
            )
        except Exception as e:
            print(f"Warning: Could not initialize question-answering pipeline: {e}")
            print("Using fallback tokenizer-only approach...")
            # Create a simple wrapper that just uses the tokenizer
            nlp_pipeline = {
                'tokenizer': tokenizer,
                'model': model,
                'type': 'fallback'
            }
        
        print("NLP model loaded successfully from local files.")
        return nlp_pipeline
        
    except Exception as e:
        print(f"\nFATAL ERROR: Could not load the local NLP model. Reason: {e}")
        print("Please ensure the model files in the local directory are not corrupted.\n")
        return None


def process_single_pdf(pdf_path: str, output_path: str, nlp_pipeline) -> None:
    """
    Executes the optimized pipeline for a single PDF file: parse, analyze with NLP, and build JSON.
    Includes performance monitoring and error handling.
    
    Args:
        pdf_path: The path to the input PDF file.
        output_path: The path where the output JSON file should be saved.
        nlp_pipeline: The loaded Hugging Face pipeline for NLP analysis.
    """
    start_time = time.time()
    filename = os.path.basename(pdf_path)
    
    try:
        # Step 1: Parse the PDF to extract raw text blocks with properties (optimized for speed).
        print(f"  [1/3] Parsing '{filename}'...")
        parse_start = time.time()
        # Use heading_extraction preset for optimal performance
        extracted_data = parse_pdf(pdf_path, preset="heading_extraction")
        parse_time = time.time() - parse_start
        
        # Check if PDF has content
        text_blocks = extracted_data.get("text_blocks", [])
        if not text_blocks:
            print(f"  -> Warning: No text found in '{filename}'. Creating empty document.")
            analysis_result = {"title": "Empty Document", "outline": []}
        else:
            print(f"  -> Extracted {len(text_blocks)} text blocks in {parse_time:.2f}s")
            
            # Step 2: Analyze the document using the enhanced analyzer.
            print("  [2/3] Analyzing document structure...")
            analysis_start = time.time()
            enhanced_analyzer = EnhancedDocumentAnalyzer()
            analysis_result = enhanced_analyzer.analyze_document(extracted_data, nlp_pipeline)
            analysis_time = time.time() - analysis_start
            
            print(f"  -> Found title: '{analysis_result['title'][:50]}{'...' if len(analysis_result['title']) > 50 else ''}'")
            print(f"  -> Found {len(analysis_result['outline'])} headings in {analysis_time:.2f}s")

        # Step 3: Build and save the final JSON file.
        print(f"  [3/3] Building JSON output...")
        json_start = time.time()
        build_json(analysis_result, output_path)
        json_time = time.time() - json_start

        total_time = time.time() - start_time
        print(f"  -> Success! Completed in {total_time:.2f}s (Parse: {parse_time:.2f}s, Analysis: {analysis_time if 'analysis_time' in locals() else 0:.2f}s, JSON: {json_time:.2f}s)\n")

    except Exception as e:
        # Gracefully handle errors on a per-file basis
        error_time = time.time() - start_time
        print(f"  -> ERROR: Could not process '{filename}' after {error_time:.2f}s. Reason: {e}")
        print(f"     Creating fallback empty document JSON.\n")
        
        # Create a fallback JSON file
        try:
            fallback_result = {"title": f"Error processing {filename}", "outline": []}
            build_json(fallback_result, output_path)
        except Exception as fallback_error:
            print(f"     Could not create fallback JSON: {fallback_error}\n")


def main():
    """
    Main function to orchestrate the optimized processing of all PDFs in the input directory.
    """
    print("=== Optimized PDF Outline Extraction (Offline Mode) ===")
    
    # Load the NLP model ONCE at the start for maximum efficiency.
    # Note: While loaded, the new logic relies more on rules, making this less of a bottleneck.
    print("\n[INITIALIZATION] Loading NLP model...")
    model_load_start = time.time()
    nlp_pipeline = load_nlp_model()
    if nlp_pipeline is None:
        print("Failed to load model. Aborting.")
        return
    
    model_load_time = time.time() - model_load_start
    print(f"[INITIALIZATION] Model loaded successfully in {model_load_time:.2f} seconds\n")

    # Ensure input and output directories are set up correctly.
    if not os.path.isdir(INPUT_DIR):
        print(f"Error: Input directory '{INPUT_DIR}' not found. Aborting.")
        return
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        print(f"Created output directory: {OUTPUT_DIR}")

    # Find all PDF files in the input directory to be processed.
    pdf_files = [f for f in os.listdir(INPUT_DIR) if f.lower().endswith(".pdf")]

    if not pdf_files:
        print(f"No PDF files found in the '{INPUT_DIR}' directory.")
        print("=== Process Finished ===")
        return

    # Sort files for consistent processing order
    pdf_files.sort()
    
    print(f"[PROCESSING] Found {len(pdf_files)} PDF(s) to process:")
    for i, filename in enumerate(pdf_files, 1):
        print(f"  {i}. {filename}")
    print()

    total_start_time = time.time()
    successful_files = 0
    failed_files = 0

    # Process each PDF with progress tracking
    for i, pdf_filename in enumerate(pdf_files, 1):
        pdf_path = os.path.join(INPUT_DIR, pdf_filename)
        json_filename = os.path.splitext(pdf_filename)[0] + ".json"
        json_path = os.path.join(OUTPUT_DIR, json_filename)
        
        print(f"[{i}/{len(pdf_files)}] Processing '{pdf_filename}'...")
        
        try:
            process_single_pdf(pdf_path, json_path, nlp_pipeline)
            successful_files += 1
        except Exception as e:
            print(f"  -> CRITICAL ERROR: {e}\n")
            failed_files += 1

    total_time = time.time() - total_start_time
    
    print("\n=== PROCESSING SUMMARY ===")
    print(f"Total files processed: {len(pdf_files)}")
    print(f"Successful: {successful_files}")
    print(f"Failed: {failed_files}")
    print(f"Total processing time: {total_time:.2f} seconds")
    if pdf_files:
        print(f"Average time per file: {total_time/len(pdf_files):.2f} seconds")
    print("==========================")
    
    # Memory cleanup
    del nlp_pipeline
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()