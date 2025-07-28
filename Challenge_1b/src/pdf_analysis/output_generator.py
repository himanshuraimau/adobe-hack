"""Output generation and JSON formatting module."""

import json
import os
import logging
from datetime import datetime
from typing import List, Dict, Any
from dataclasses import dataclass

from .semantic_ranker import RankedSection


@dataclass
class OutputFormat:
    """Represents the complete output structure."""
    metadata: Dict[str, Any]
    extracted_sections: List[Dict[str, Any]]
    subsection_analysis: List[Dict[str, Any]]


class OutputGenerator:
    """Formats analysis results into required JSON structure and saves output."""
    
    def __init__(self):
        self.output_filename = "challenge1b_output.json"
        self.logger = logging.getLogger(__name__)
    
    def generate_metadata(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Create metadata section from input configuration."""
        # Extract document filenames from input data
        input_documents = []
        if 'documents' in input_data:
            input_documents = [doc['filename'] for doc in input_data['documents']]
        
        # Extract persona role
        persona = ""
        if 'persona' in input_data and 'role' in input_data['persona']:
            persona = input_data['persona']['role']
        
        # Extract job to be done task
        job_to_be_done = ""
        if 'job_to_be_done' in input_data and 'task' in input_data['job_to_be_done']:
            job_to_be_done = input_data['job_to_be_done']['task']
        
        # Generate processing timestamp
        processing_timestamp = datetime.now().isoformat()
        
        metadata = {
            "input_documents": input_documents,
            "persona": persona,
            "job_to_be_done": job_to_be_done,
            "processing_timestamp": processing_timestamp
        }
        
        return metadata
    
    def format_extracted_sections(self, ranked_sections: List[RankedSection]) -> List[Dict[str, Any]]:
        """Format section rankings into extracted_sections structure."""
        extracted_sections = []
        
        for ranked_section in ranked_sections:
            section_data = {
                "document": ranked_section.document,
                "section_title": ranked_section.section_title,
                "importance_rank": ranked_section.importance_rank,
                "page_number": ranked_section.page_number
            }
            extracted_sections.append(section_data)
        
        return extracted_sections
    
    def format_subsection_analysis(self, sentence_analysis: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Format sentence analysis results into subsection_analysis structure."""
        subsection_analysis = []
        
        for analysis in sentence_analysis:
            subsection_data = {
                "document": analysis["document"],
                "refined_text": analysis["refined_text"],
                "page_number": analysis["page_number"]
            }
            subsection_analysis.append(subsection_data)
        
        return subsection_analysis
    
    def create_output_format(self, 
                           input_data: Dict[str, Any],
                           ranked_sections: List[RankedSection],
                           sentence_analysis: List[Dict[str, Any]]) -> OutputFormat:
        """Create complete output format from processing results."""
        
        metadata = self.generate_metadata(input_data)
        extracted_sections = self.format_extracted_sections(ranked_sections)
        subsection_analysis = self.format_subsection_analysis(sentence_analysis)
        
        return OutputFormat(
            metadata=metadata,
            extracted_sections=extracted_sections,
            subsection_analysis=subsection_analysis
        )
    
    def save_output(self, collection_path: str, output_data: OutputFormat) -> str:
        """Save JSON output to collection directory with comprehensive error handling."""
        if not collection_path:
            raise ValueError("Collection path cannot be empty")
        
        output_file_path = os.path.join(collection_path, self.output_filename)
        
        try:
            # Validate output data
            if not output_data:
                raise ValueError("Output data cannot be None")
            
            # Convert OutputFormat to dictionary for JSON serialization
            output_dict = {
                "metadata": output_data.metadata,
                "extracted_sections": output_data.extracted_sections,
                "subsection_analysis": output_data.subsection_analysis
            }
            
            # Validate JSON serializability
            try:
                json.dumps(output_dict)
            except (TypeError, ValueError) as e:
                raise RuntimeError(f"Output data is not JSON serializable: {e}")
            
            # Ensure the collection directory exists
            try:
                os.makedirs(collection_path, exist_ok=True)
            except PermissionError:
                raise PermissionError(f"Permission denied creating directory: {collection_path}")
            except OSError as e:
                raise RuntimeError(f"Error creating directory {collection_path}: {e}")
            
            # Check available disk space (basic check)
            try:
                import shutil
                free_space = shutil.disk_usage(collection_path).free
                estimated_size = len(json.dumps(output_dict)) * 2  # Rough estimate with formatting
                if free_space < estimated_size:
                    self.logger.warning(f"Low disk space: {free_space} bytes available, need ~{estimated_size}")
            except Exception:
                pass  # Continue without disk space check if it fails
            
            # Write JSON file with proper formatting
            try:
                with open(output_file_path, 'w', encoding='utf-8') as f:
                    json.dump(output_dict, f, indent=4, ensure_ascii=False)
                
                # Verify file was written correctly
                if not os.path.exists(output_file_path):
                    raise RuntimeError("Output file was not created")
                
                file_size = os.path.getsize(output_file_path)
                if file_size == 0:
                    raise RuntimeError("Output file is empty")
                
                self.logger.info(f"Output saved successfully: {output_file_path} ({file_size} bytes)")
                
            except PermissionError:
                raise PermissionError(f"Permission denied writing to: {output_file_path}")
            except OSError as e:
                raise RuntimeError(f"Error writing to file {output_file_path}: {e}")
            
            return output_file_path
            
        except Exception as e:
            self.logger.error(f"Failed to save output to {output_file_path}: {e}")
            raise RuntimeError(f"Failed to save output to {output_file_path}: {e}")
    
    def generate_and_save_output(self,
                               collection_path: str,
                               input_data: Dict[str, Any],
                               ranked_sections: List[RankedSection],
                               sentence_analysis: List[Dict[str, Any]]) -> str:
        """Complete workflow: generate output format and save to file."""
        
        # Create the complete output structure
        output_data = self.create_output_format(input_data, ranked_sections, sentence_analysis)
        
        # Save to JSON file
        output_file_path = self.save_output(collection_path, output_data)
        
        return output_file_path