#!/usr/bin/env python3
"""
Demonstration script for the JSON Builder functionality.

This script shows how to use the JSONBuilder class to convert document
structures into properly formatted JSON output.
"""

import json
from pathlib import Path
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from problem1a.json_builder import JSONBuilder
from problem1a.models import DocumentStructure, Heading


def demo_basic_usage():
    """Demonstrate basic JSON builder usage."""
    print("=== Basic JSON Builder Demo ===")
    
    # Create sample document structure
    headings = [
        Heading(level="H1", text="Introduction", page=1, confidence=0.95),
        Heading(level="H2", text="Background", page=1, confidence=0.88),
        Heading(level="H2", text="Objectives", page=2, confidence=0.90),
        Heading(level="H1", text="Methodology", page=3, confidence=0.92),
        Heading(level="H2", text="Data Collection", page=3, confidence=0.85),
        Heading(level="H3", text="Survey Design", page=4, confidence=0.82),
        Heading(level="H1", text="Results", page=6, confidence=0.94),
        Heading(level="H1", text="Conclusion", page=8, confidence=0.96),
    ]
    
    structure = DocumentStructure(
        title="Research Paper: AI in Document Processing",
        headings=headings,
        metadata={"pages": 10, "language": "en"}
    )
    
    # Create JSON builder and generate output
    json_builder = JSONBuilder()
    result = json_builder.build_json(structure)
    
    print("Generated JSON:")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    print()


def demo_edge_cases():
    """Demonstrate handling of edge cases."""
    print("=== Edge Cases Demo ===")
    
    # Case 1: No title
    print("1. Document with no title:")
    structure_no_title = DocumentStructure(
        title=None,
        headings=[Heading(level="H1", text="Only Heading", page=1, confidence=0.8)],
        metadata={}
    )
    
    json_builder = JSONBuilder()
    result = json_builder.build_json(structure_no_title)
    print(json.dumps(result, indent=2))
    print()
    
    # Case 2: No headings
    print("2. Document with no headings:")
    structure_no_headings = DocumentStructure(
        title="Empty Document",
        headings=[],
        metadata={}
    )
    
    result = json_builder.build_json(structure_no_headings)
    print(json.dumps(result, indent=2))
    print()
    
    # Case 3: Title-level headings (should be excluded)
    print("3. Document with title-level headings:")
    structure_with_title_heading = DocumentStructure(
        title="Document Title",
        headings=[
            Heading(level="title", text="Main Title", page=1, confidence=0.95),
            Heading(level="H1", text="First Section", page=1, confidence=0.9),
            Heading(level="H2", text="Subsection", page=2, confidence=0.85),
        ],
        metadata={}
    )
    
    result = json_builder.build_json(structure_with_title_heading)
    print(json.dumps(result, indent=2))
    print("Note: Title-level heading excluded from outline")
    print()


def demo_multilingual():
    """Demonstrate multilingual content handling."""
    print("=== Multilingual Content Demo ===")
    
    headings = [
        Heading(level="H1", text="Introduction", page=1, confidence=0.95),
        Heading(level="H2", text="Introducción", page=2, confidence=0.90),  # Spanish
        Heading(level="H2", text="Einführung", page=3, confidence=0.88),    # German
        Heading(level="H1", text="Méthodologie", page=4, confidence=0.92),  # French
        Heading(level="H2", text="データ収集", page=5, confidence=0.85),      # Japanese
        Heading(level="H1", text="Заключение", page=6, confidence=0.89),    # Russian
    ]
    
    structure = DocumentStructure(
        title="Multilingual Research: 多言語研究",
        headings=headings,
        metadata={"languages": ["en", "es", "de", "fr", "ja", "ru"]}
    )
    
    json_builder = JSONBuilder()
    result = json_builder.build_json(structure)
    
    print("Multilingual JSON output:")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    print()


def demo_file_output():
    """Demonstrate writing JSON to file."""
    print("=== File Output Demo ===")
    
    # Create sample structure
    headings = [
        Heading(level="H1", text="Chapter 1", page=1, confidence=0.95),
        Heading(level="H2", text="Section 1.1", page=1, confidence=0.88),
        Heading(level="H2", text="Section 1.2", page=3, confidence=0.90),
        Heading(level="H1", text="Chapter 2", page=5, confidence=0.92),
    ]
    
    structure = DocumentStructure(
        title="Sample Document for File Output",
        headings=headings,
        metadata={"demo": True}
    )
    
    # Write to file
    json_builder = JSONBuilder()
    output_path = Path("output") / "demo_output.json"
    
    result = json_builder.process_and_write(structure, str(output_path))
    
    print(f"JSON written to: {output_path}")
    print("File contents:")
    
    if output_path.exists():
        with open(output_path, 'r', encoding='utf-8') as f:
            content = f.read()
        print(content)
    else:
        print("File was not created successfully")
    
    print()


def demo_validation():
    """Demonstrate JSON validation."""
    print("=== Validation Demo ===")
    
    from problem1a.json_builder import OutputValidator
    
    validator = OutputValidator()
    
    # Valid JSON
    valid_json = {
        "title": "Valid Document",
        "outline": [
            {"level": "H1", "text": "Valid Heading", "page": 1}
        ]
    }
    
    print("1. Valid JSON:")
    print(f"Validation result: {validator.validate_output(valid_json)}")
    
    # Invalid JSON - missing title
    invalid_json = {
        "outline": [
            {"level": "H1", "text": "Heading", "page": 1}
        ]
    }
    
    print("2. Invalid JSON (missing title):")
    print(f"Validation result: {validator.validate_output(invalid_json)}")
    
    # Invalid JSON - bad level
    invalid_level_json = {
        "title": "Document",
        "outline": [
            {"level": "H4", "text": "Invalid Level", "page": 1}
        ]
    }
    
    print("3. Invalid JSON (bad heading level):")
    print(f"Validation result: {validator.validate_output(invalid_level_json)}")
    print()


if __name__ == "__main__":
    print("JSON Builder Demonstration")
    print("=" * 50)
    print()
    
    demo_basic_usage()
    demo_edge_cases()
    demo_multilingual()
    demo_file_output()
    demo_validation()
    
    print("Demo completed successfully!")