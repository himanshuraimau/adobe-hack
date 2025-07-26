# src/json_builder.py

import json
from typing import Dict, Any

def build_json(analysis_result: Dict[str, Any], output_path: str) -> None:
    """
    Constructs the final JSON object from the analysis result and writes it to a file.

    This function takes the dictionary containing the title and the outline,
    formats it into a JSON string with indentation for readability, and saves
    it to the specified output path.

    Args:
        analysis_result: A dictionary with 'title' and 'outline' keys, as
                         produced by the document_analyzer.
        output_path: The full path where the output JSON file will be saved.

    Returns:
        None. The function writes directly to a file.
    """
    # Prepare the final dictionary structure as per the requirements.
    # The analysis_result already matches this, but this step ensures conformity.
    final_json_data = {
        "title": analysis_result.get("title", "Title Not Found"),
        "outline": analysis_result.get("outline", [])
    }

    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            # Use json.dump for writing to a file.
            # indent=2 makes the output JSON human-readable.
            # ensure_ascii=False allows for correct handling of multilingual
            # characters (like Japanese) without escaping them.
            json.dump(final_json_data, f, indent=2, ensure_ascii=False)
            
    except IOError as e:
        print(f"Error: Could not write JSON file to {output_path}. Reason: {e}")
    except TypeError as e:
        print(f"Error: Data for JSON serialization is not valid. Reason: {e}")