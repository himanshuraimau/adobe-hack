"""
Integration test for JSON builder with actual document processing.

This test verifies that the JSON builder integrates correctly with the
document processing pipeline and produces valid output.
"""

import json
import tempfile
from pathlib import Path

from .json_builder import JSONBuilder
from .models import DocumentStructure, Heading


def test_json_builder_with_sample_document():
    """Test JSON builder with a realistic document structure."""
    
    # Create a realistic document structure similar to what would be
    # extracted from a PDF document
    headings = [
        Heading(level="H1", text="Executive Summary", page=1, confidence=0.95),
        Heading(level="H2", text="Key Findings", page=1, confidence=0.88),
        Heading(level="H2", text="Recommendations", page=2, confidence=0.90),
        Heading(level="H1", text="Introduction", page=3, confidence=0.92),
        Heading(level="H2", text="Background", page=3, confidence=0.85),
        Heading(level="H3", text="Problem Statement", page=4, confidence=0.82),
        Heading(level="H3", text="Research Objectives", page=4, confidence=0.87),
        Heading(level="H1", text="Methodology", page=6, confidence=0.93),
        Heading(level="H2", text="Data Collection", page=6, confidence=0.89),
        Heading(level="H2", text="Analysis Framework", page=8, confidence=0.86),
        Heading(level="H1", text="Results", page=10, confidence=0.94),
        Heading(level="H2", text="Statistical Analysis", page=10, confidence=0.88),
        Heading(level="H3", text="Descriptive Statistics", page=11, confidence=0.84),
        Heading(level="H3", text="Inferential Statistics", page=13, confidence=0.83),
        Heading(level="H2", text="Qualitative Findings", page=15, confidence=0.87),
        Heading(level="H1", text="Discussion", page=18, confidence=0.91),
        Heading(level="H2", text="Implications", page=18, confidence=0.85),
        Heading(level="H2", text="Limitations", page=20, confidence=0.86),
        Heading(level="H1", text="Conclusion", page=22, confidence=0.96),
        Heading(level="H1", text="References", page=24, confidence=0.89),
    ]
    
    structure = DocumentStructure(
        title="Impact of Digital Transformation on Business Performance: A Comprehensive Analysis",
        headings=headings,
        metadata={
            "total_pages": 25,
            "language": "en",
            "document_type": "research_paper",
            "extraction_timestamp": "2024-01-15T10:30:00Z"
        }
    )
    
    # Test JSON generation
    json_builder = JSONBuilder()
    
    with tempfile.TemporaryDirectory() as temp_dir:
        output_path = Path(temp_dir) / "sample_document.json"
        
        # Process and write JSON
        result = json_builder.process_and_write(structure, str(output_path))
        
        # Verify the result structure
        assert result["title"] == "Impact of Digital Transformation on Business Performance: A Comprehensive Analysis"
        assert len(result["outline"]) == 20  # All headings should be included
        
        # Verify hierarchical structure is preserved
        assert result["outline"][0]["level"] == "H1"
        assert result["outline"][0]["text"] == "Executive Summary"
        assert result["outline"][0]["page"] == 1
        
        assert result["outline"][1]["level"] == "H2"
        assert result["outline"][1]["text"] == "Key Findings"
        assert result["outline"][1]["page"] == 1
        
        # Verify H3 levels are included
        h3_entries = [entry for entry in result["outline"] if entry["level"] == "H3"]
        assert len(h3_entries) == 4
        assert h3_entries[0]["text"] == "Problem Statement"
        
        # Verify file was written correctly
        assert output_path.exists()
        
        # Verify file content is valid JSON and matches result
        with open(output_path, 'r', encoding='utf-8') as f:
            file_content = json.load(f)
        
        assert file_content == result
        
        # Verify JSON is properly formatted (indented)
        with open(output_path, 'r', encoding='utf-8') as f:
            raw_content = f.read()
        
        assert "  " in raw_content  # Should be indented
        assert "\n" in raw_content  # Should have line breaks
        
        # Verify all required fields are present in each outline entry
        for entry in result["outline"]:
            assert "level" in entry
            assert "text" in entry
            assert "page" in entry
            assert entry["level"] in ["H1", "H2", "H3"]
            assert isinstance(entry["text"], str)
            assert len(entry["text"].strip()) > 0
            assert isinstance(entry["page"], int)
            assert entry["page"] >= 1


def test_json_builder_with_multilingual_content():
    """Test JSON builder with multilingual document content."""
    
    # Create document with multilingual headings
    headings = [
        Heading(level="H1", text="Introduction", page=1, confidence=0.95),
        Heading(level="H2", text="Introducción", page=2, confidence=0.90),  # Spanish
        Heading(level="H2", text="Einführung", page=3, confidence=0.88),    # German
        Heading(level="H1", text="Méthodologie", page=4, confidence=0.92),  # French
        Heading(level="H2", text="データ収集", page=5, confidence=0.85),      # Japanese
        Heading(level="H1", text="Заключение", page=6, confidence=0.89),    # Russian
    ]
    
    structure = DocumentStructure(
        title="Multilingual Document: Global Research Perspectives",
        headings=headings,
        metadata={"languages": ["en", "es", "de", "fr", "ja", "ru"]}
    )
    
    json_builder = JSONBuilder()
    
    with tempfile.TemporaryDirectory() as temp_dir:
        output_path = Path(temp_dir) / "multilingual_document.json"
        
        result = json_builder.process_and_write(structure, str(output_path))
        
        # Verify multilingual content is preserved
        assert result["title"] == "Multilingual Document: Global Research Perspectives"
        assert len(result["outline"]) == 6
        
        # Check specific multilingual entries
        spanish_entry = next(entry for entry in result["outline"] if entry["text"] == "Introducción")
        assert spanish_entry["level"] == "H2"
        assert spanish_entry["page"] == 2
        
        japanese_entry = next(entry for entry in result["outline"] if entry["text"] == "データ収集")
        assert japanese_entry["level"] == "H2"
        assert japanese_entry["page"] == 5
        
        # Verify file was written with proper UTF-8 encoding
        with open(output_path, 'r', encoding='utf-8') as f:
            file_content = json.load(f)
        
        assert file_content == result
        
        # Verify special characters are preserved
        raw_json = json.dumps(result, ensure_ascii=False, indent=2)
        assert "データ収集" in raw_json
        assert "Заключение" in raw_json


def test_json_builder_edge_cases():
    """Test JSON builder with various edge cases."""
    
    # Test with minimal document
    minimal_structure = DocumentStructure(
        title="Minimal Document",
        headings=[],
        metadata={}
    )
    
    json_builder = JSONBuilder()
    result = json_builder.build_json(minimal_structure)
    
    assert result["title"] == "Minimal Document"
    assert result["outline"] == []
    
    # Test with no title
    no_title_structure = DocumentStructure(
        title=None,
        headings=[Heading(level="H1", text="Only Heading", page=1, confidence=0.8)],
        metadata={}
    )
    
    result = json_builder.build_json(no_title_structure)
    
    assert result["title"] == "Untitled Document"
    assert len(result["outline"]) == 1
    
    # Test with title-level headings (should be excluded from outline)
    title_heading_structure = DocumentStructure(
        title="Document with Title Heading",
        headings=[
            Heading(level="title", text="Main Title", page=1, confidence=0.95),
            Heading(level="H1", text="First Section", page=1, confidence=0.9),
        ],
        metadata={}
    )
    
    result = json_builder.build_json(title_heading_structure)
    
    assert result["title"] == "Document with Title Heading"
    assert len(result["outline"]) == 1  # Only H1, not title-level heading
    assert result["outline"][0]["level"] == "H1"


if __name__ == "__main__":
    test_json_builder_with_sample_document()
    test_json_builder_with_multilingual_content()
    test_json_builder_edge_cases()
    print("All integration tests passed!")