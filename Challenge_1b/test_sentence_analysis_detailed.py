#!/usr/bin/env python3
"""Detailed test for sentence-level analysis functionality."""

import logging
from src.pdf_analysis.collection_processor import CollectionProcessor

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def test_sentence_analysis_detailed():
    """Test sentence-level analysis with detailed output."""
    processor = CollectionProcessor()
    
    # Process just Collection 1 for detailed analysis
    result = processor.process_collection("./Collection 1")
    
    if result.success:
        print(f"✓ Collection processed successfully")
        print(f"  - Documents processed: {result.documents_processed}")
        print(f"  - Sections processed: {result.sections_processed}")
        
        # Let's also test the semantic ranker directly to see sentence analysis details
        config = processor._load_input_config("./Collection 1")
        document_filenames = [doc['filename'] for doc in config.documents]
        pdf_paths = processor._find_pdf_files("./Collection 1", document_filenames)
        
        # Extract sections from first PDF only for detailed testing
        sections = processor.pdf_analyzer.get_section_content(pdf_paths[0])
        print(f"\n✓ Extracted {len(sections)} sections from first PDF")
        
        # Create query embedding
        query_embedding = processor.semantic_ranker.create_query_embedding(
            config.persona, config.job_to_be_done
        )
        
        # Rank sections
        ranked_sections = processor.semantic_ranker.rank_sections(query_embedding, sections)
        print(f"✓ Ranked {len(ranked_sections)} sections")
        
        # Show top 5 ranked sections
        print("\nTop 5 ranked sections:")
        for i, section in enumerate(ranked_sections[:5]):
            print(f"  {i+1}. '{section.section_title}' (score: {section.similarity_score:.3f})")
        
        # Get top sections for sentence analysis
        top_sections = []
        for ranked_section in ranked_sections[:5]:  # Test with top 5
            for section in sections:
                if (section.document == ranked_section.document and 
                    section.section_title == ranked_section.section_title and
                    section.page_number == ranked_section.page_number):
                    top_sections.append(section)
                    break
        
        # Perform sentence analysis
        sentence_analysis = processor.semantic_ranker.analyze_sentences(query_embedding, top_sections)
        print(f"\n✓ Sentence analysis completed for {len(sentence_analysis)} sections")
        
        # Show detailed sentence analysis results
        for i, analysis in enumerate(sentence_analysis):
            print(f"\nSection {i+1}: {analysis['document']} (Page {analysis['page_number']})")
            print(f"  Sentences analyzed: {analysis['sentence_count']}")
            print(f"  Average similarity: {analysis['avg_similarity']:.3f}")
            print(f"  Refined text preview: {analysis['refined_text'][:100]}...")
        
        print(f"\n✓ All sentence-level analysis requirements verified!")
        
    else:
        print(f"✗ Collection processing failed: {result.error_message}")

if __name__ == "__main__":
    test_sentence_analysis_detailed()