#!/usr/bin/env python3
"""
Dependency checker script for PDF Structure Extractor.

This script checks if all required dependencies are installed and provides
helpful installation instructions if any are missing.
"""

import sys
import importlib
from pathlib import Path

def check_dependency(module_name, package_name=None, description=""):
    """Check if a dependency is available."""
    if package_name is None:
        package_name = module_name
    
    try:
        importlib.import_module(module_name)
        return True, f"✅ {package_name} - {description}"
    except ImportError as e:
        return False, f"❌ {package_name} - {description} (Error: {e})"

def main():
    """Main dependency checking function."""
    print("🔍 PDF Structure Extractor - Dependency Check")
    print("=" * 50)
    
    # Define required dependencies
    dependencies = [
        ("fitz", "pymupdf>=1.26.3", "PDF text extraction and parsing"),
        ("torch", "torch>=2.7.1", "Machine learning framework"),
        ("transformers", "transformers>=4.54.0", "BERT model support"),
        ("numpy", "numpy>=2.3.2", "Numerical computing"),
        ("psutil", "psutil>=7.0.0", "System monitoring and profiling"),
        ("pytest", "pytest>=8.0.0", "Testing framework"),
    ]
    
    # Optional dependencies
    optional_dependencies = [
        ("black", "black>=23.0.0", "Code formatting (dev)"),
        ("flake8", "flake8>=6.0.0", "Code linting (dev)"),
        ("mypy", "mypy>=1.0.0", "Type checking (dev)"),
        ("pytest_cov", "pytest-cov>=4.0.0", "Test coverage (dev)"),
    ]
    
    print("\n📦 Required Dependencies:")
    print("-" * 30)
    
    all_required_ok = True
    results = []
    
    for module, package, desc in dependencies:
        ok, message = check_dependency(module, package, desc)
        results.append(message)
        if not ok:
            all_required_ok = False
    
    for result in results:
        print(result)
    
    print("\n🔧 Optional Dependencies (Development):")
    print("-" * 40)
    
    optional_results = []
    for module, package, desc in optional_dependencies:
        ok, message = check_dependency(module, package, desc)
        optional_results.append(message)
    
    for result in optional_results:
        print(result)
    
    # Check project structure
    print("\n📁 Project Structure:")
    print("-" * 20)
    
    project_root = Path(__file__).parent.parent
    structure_checks = [
        (project_root / "src" / "pdf_extractor", "Main package directory"),
        (project_root / "src" / "pdf_extractor" / "core", "Core modules"),
        (project_root / "src" / "pdf_extractor" / "models", "Data models"),
        (project_root / "src" / "pdf_extractor" / "config", "Configuration"),
        (project_root / "tests", "Test suite"),
        (project_root / "models" / "local_mobilebert", "MobileBERT model"),
        (project_root / "data" / "input", "Input directory"),
        (project_root / "data" / "output", "Output directory"),
    ]
    
    structure_ok = True
    for path, desc in structure_checks:
        if path.exists():
            print(f"✅ {desc}: {path}")
        else:
            print(f"❌ {desc}: {path} (missing)")
            structure_ok = False
    
    # Summary
    print("\n" + "=" * 50)
    print("📋 Summary:")
    
    if all_required_ok and structure_ok:
        print("🎉 All checks passed! Your environment is ready.")
        print("\n🚀 You can now run:")
        print("   python3 main.py data/input/ -o data/output/")
        return 0
    else:
        print("⚠️  Some issues detected.")
        
        if not all_required_ok:
            print("\n💡 To install missing dependencies:")
            print("   uv sync                    # Recommended")
            print("   # OR")
            print("   pip install -e .          # Alternative")
            print("   # OR install individually:")
            print("   pip install pymupdf torch transformers numpy psutil pytest")
        
        if not structure_ok:
            print("\n📁 Project structure issues detected.")
            print("   Please ensure you're running from the project root directory.")
            print("   Some directories may need to be created manually.")
        
        return 1

if __name__ == "__main__":
    sys.exit(main())