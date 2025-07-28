#!/usr/bin/env python3
"""Test script to verify SemanticRanker implementation."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from pdf_analysis import SemanticRanker, Section

def test_semantic_ranker():
    """Test the SemanticRanker functionality."""
    print("Testing SemanticRanker implementation...")
    
    # Create test data
    test_sections = [
        Section(
            document="test.pdf",
            section_title="Introduction to Cooking",
            content="This section covers basic cooking techniques and kitchen safety.",
            page_number=1,
            font_info={}
        ),
        Section(
            document="test.pdf", 
            section_title="Advanced Recipes",
            content="Complex dishes requiring professional techniques and specialized equipment.",
            page_number=2,
            font_info={}
        ),
        Section(
            document="test.pdf",
            section_title="Kitchen Equipment",
            content="Essential tools and appliances for home cooking and food preparation.",
            page_number=3,
            font_info={}
        )
    ]
    
    # Test persona and job
    test_persona = {"role": "home cook"}
    test_job = {"task": "learn basic cooking techniques"}
    
    # Initialize ranker
    ranker = SemanticRanker()
    
    # Test 1: Model loading
    print("1. Testing model loading...")
    try:
        ranker.load_model()
        print("✓ Model loaded successfully")
    except Exception as e:
        print(f"✗ Model loading failed: {e}")
        return False
    
    # Test 2: Query embedding creation
    print("2. Testing query embedding creation...")
    try:
        query_embedding = ranker.create_query_embedding(test_persona, test_job)
        print(f"✓ Query embedding created with shape: {query_embedding.shape}")
        assert len(query_embedding.shape) == 1, "Query embedding should be 1D"
        assert query_embedding.shape[0] > 0, "Query embedding should have dimensions"
    except Exception as e:
        print(f"✗ Query embedding creation failed: {e}")
        return False
    
    # Test 3: Section ranking
    print("3. Testing section ranking...")
    try:
        ranked_sections = ranker.rank_sections(query_embedding, test_sections)
        print(f"✓ Ranked {len(ranked_sections)} sections")
        
        # Verify ranking structure
        assert len(ranked_sections) == len(test_sections), "Should rank all sections"
        assert all(hasattr(rs, 'importance_rank') for rs in ranked_sections), "All sections should have ranks"
        assert all(hasattr(rs, 'similarity_score') for rs in ranked_sections), "All sections should have scores"
        
        # Verify ranks are sequential starting from 1
        ranks = [rs.importance_rank for rs in ranked_sections]
        expected_ranks = list(range(1, len(test_sections) + 1))
        assert sorted(ranks) == expected_ranks, f"Ranks should be {expected_ranks}, got {sorted(ranks)}"
        
        # Print ranking results
        for rs in ranked_sections:
            print(f"  Rank {rs.importance_rank}: '{rs.section_title}' (score: {rs.similarity_score:.3f})")
            
    except Exception as e:
        print(f"✗ Section ranking failed: {e}")
        return False
    
    # Test 4: Sentence analysis
    print("4. Testing sentence analysis...")
    try:
        # Use top 2 sections for sentence analysis
        top_sections = [s for s in test_sections if any(rs.section_title == s.section_title and rs.importance_rank <= 2 for rs in ranked_sections)]
        sentence_analysis = ranker.analyze_sentences(query_embedding, top_sections[:2])
        
        print(f"✓ Analyzed sentences for {len(sentence_analysis)} sections")
        
        # Verify analysis structure
        for analysis in sentence_analysis:
            assert 'document' in analysis, "Analysis should include document"
            assert 'refined_text' in analysis, "Analysis should include refined_text"
            assert 'page_number' in analysis, "Analysis should include page_number"
            assert len(analysis['refined_text']) > 0, "Refined text should not be empty"
            
        # Print analysis results
        for analysis in sentence_analysis:
            print(f"  Document: {analysis['document']}, Page: {analysis['page_number']}")
            print(f"  Refined text: {analysis['refined_text'][:100]}...")
            
    except Exception as e:
        print(f"✗ Sentence analysis failed: {e}")
        return False
    
    print("\n✓ All SemanticRanker tests passed!")
    return True

if __name__ == "__main__":
    success = test_semantic_ranker()
    sys.exit(0 if success else 1)