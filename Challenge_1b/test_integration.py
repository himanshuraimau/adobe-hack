#!/usr/bin/env python3
"""Integration test for SemanticRanker with PDFAnalyzer."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from pdf_analysis import PDFAnalyzer, SemanticRanker

def test_integration():
    """Test SemanticRanker integration with real PDF data."""
    print("Testing SemanticRanker integration...")
    
    # Initialize components
    pdf_analyzer = PDFAnalyzer()
    semantic_ranker = SemanticRanker()
    
    # Test with a real PDF from the collections
    test_pdf = "Collection 1/PDFs/South of France - Cuisine.pdf"
    
    if not os.path.exists(test_pdf):
        print(f"✗ Test PDF not found: {test_pdf}")
        return False
    
    try:
        # Extract sections from PDF
        print("1. Extracting sections from PDF...")
        sections = pdf_analyzer.get_section_content(test_pdf)
        print(f"✓ Extracted {len(sections)} sections")
        
        # Create query embedding
        print("2. Creating query embedding...")
        persona = {"role": "food enthusiast"}
        job = {"task": "discover traditional French cooking techniques"}
        
        query_embedding = semantic_ranker.create_query_embedding(persona, job)
        print(f"✓ Created query embedding with shape: {query_embedding.shape}")
        
        # Rank sections
        print("3. Ranking sections...")
        ranked_sections = semantic_ranker.rank_sections(query_embedding, sections)
        print(f"✓ Ranked {len(ranked_sections)} sections")
        
        # Show top 3 ranked sections
        print("\nTop 3 ranked sections:")
        for i, rs in enumerate(ranked_sections[:3]):
            print(f"  {rs.importance_rank}. '{rs.section_title}' (score: {rs.similarity_score:.3f})")
        
        # Perform sentence analysis on top sections
        print("\n4. Performing sentence analysis...")
        top_sections = [s for s in sections if any(rs.section_title == s.section_title and rs.importance_rank <= 3 for rs in ranked_sections)]
        sentence_analysis = semantic_ranker.analyze_sentences(query_embedding, top_sections[:3])
        
        print(f"✓ Completed sentence analysis for {len(sentence_analysis)} sections")
        
        # Show refined text samples
        print("\nRefined text samples:")
        for analysis in sentence_analysis[:2]:
            print(f"  Document: {analysis['document']}")
            print(f"  Refined text: {analysis['refined_text'][:150]}...")
            print()
        
        print("✓ Integration test completed successfully!")
        return True
        
    except Exception as e:
        print(f"✗ Integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_integration()
    sys.exit(0 if success else 1)