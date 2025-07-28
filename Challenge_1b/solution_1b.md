# Approach: Persona-Driven Document Intelligence (Multi-Collection Analysis)

This document outlines a robust, multi-stage approach to building a system that extracts and prioritizes document sections based on a specific persona and task, now extended to handle **multiple document collections**.

The solution is designed to operate entirely on a CPU, with no internet access, and adheres to a model size constraint of less than 1 GB.

## 1. System Architecture

Our proposed system follows a modular pipeline architecture, ensuring each step is optimized for performance and accuracy. The key stages are:

1.  **Collection Iteration:** The system will process each document collection sequentially.
2.  **Document Ingestion & Pre-processing:** PDFs within a collection are converted into a structured text format.
3.  **Layout-Aware Sectioning:** Text is segmented into meaningful sections based on visual and structural cues.
4.  **Semantic Representation (Embedding):** A lightweight sentence-transformer model converts the persona, job description, and document sections into numerical vectors (embeddings).
5.  **Relevance Scoring & Ranking:** We calculate the semantic similarity between the user's query (persona + job) and each document section within the current collection to determine its relevance.
6.  **Granular Information Extraction:** The top-ranked sections are further analyzed to extract the most critical sentences and details.
7.  **Output Generation:** The final, prioritized information for each collection is formatted into the required JSON structure.

## 2. Core Components & Model Selection

The success of this system hinges on selecting the right tools and models that fit within the given constraints.

*   **PDF Text & Layout Extraction:** We will use the `PyMuPDF` library. It is highly efficient for extracting not only text but also layout information like font size, font weight, and block positioning. This is crucial for accurately identifying headers and section boundaries without relying on complex visual models.

*   **Semantic Search & Ranking Model:** The core of our system is a sentence-transformer model. These models are ideal for understanding the semantic meaning of text. Given the constraints (CPU-only, ≤ 1 GB), we have selected the **`all-mpnet-base-v2`** model.
    *   **Why this model?**
        *   **Size:** It is significantly larger than `all-MiniLM-L6-v2`, with a size of approximately **410 MB** (for the ONNX version, PyTorch version is similar), which is well within the 1 GB limit and provides a more robust semantic understanding.
        *   **Performance:** While larger, it still offers excellent performance on CPU for semantic similarity tasks.
        *   **Effectiveness:** It excels at creating meaningful embeddings for sentences and paragraphs, making it highly effective for comparing the user's query to document content and providing better relevance ranking.

**Model Details (`all-mpnet-base-v2`):**
*   **Size:** ~410 MB (PyTorch model file, ONNX version is similar).
*   **Dimensions:** Outputs 768-dimensional embeddings.
*   **Performance:** Optimized for CPU inference.  For each `Collection`, it will read the `challenge1b_input.json` to get the list of PDFs, the persona, and the job-to-be-done.
    *   We will combine the persona and job into a single, coherent query string (e.g., "As an Investment Analyst, I need to analyze revenue trends, R&D investments, and market positioning").

2.  **Section & Sub-section Extraction (per Collection):**
    *   For each PDF within the current collection, `PyMuPDF` will be used to iterate through its pages and extract text blocks.
    *   We will implement a rule-based heuristic to identify section headers. A text block is likely a header if it has a larger font size, bold styling, or is followed by smaller-font body text.
    *   The content between two consecutive headers is treated as a single "section." This creates a structured representation of each document.

3.  **Embedding and Ranking (per Collection):**
    *   The `all-MiniLM-L6-v2` model will be loaded using the `sentence-transformers` library (loaded once globally).
    *   The model will generate a single embedding for the combined user query for the current collection.
    *   It will then generate an embedding for each extracted document section from the current collection.
    *   We will use **cosine similarity** to calculate the relevance score between the query embedding and each section embedding.
    *   Sections will be ranked in descending order of their similarity scores to determine the `importance_rank`.

4.  **Granular Information Extraction (Sub-section Analysis - per Collection):**
    *   The top 5-10 highest-ranked sections from the current collection will be selected for more detailed analysis.
    *   Each of these top sections will be split into individual sentences.
    *   The model will generate embeddings for every sentence within these top sections.
    *   We will again use cosine similarity to score each sentence against the user query embedding.
    *   The top-scoring sentences from across all the high-importance sections will be collected to form the `refined_text` for the `subsection_analysis` output.

5.  **Final Output Generation (per Collection):**
    *   The system will assemble the final JSON output for the current collection, populating the `metadata`, `extracted_sections`, and `subsection_analysis` fields according to the specified structure.
    *   This output will be saved to `challenge1b_output.json` within the respective `Collection` directory.

## 4. Detailed Implementation Plan

This section elaborates on the technical specifications and implementation steps for each component of the Persona-Driven Document Intelligence system, with specific considerations for multi-collection processing.

### 4.1. Environment Setup and Dependencies

The system will be built using Python 3.8+.

**Key Libraries:**
*   `PyMuPDF` (or `fitz`): For efficient PDF parsing and layout analysis.
*   `sentence-transformers`: For loading and using pre-trained sentence embedding models.
*   `scikit-learn`: For cosine similarity calculation.
*   `json`: For handling input/output data.
*   `os`, `glob`: For navigating file system and finding collection directories.

**Installation (within Dockerfile):**
```dockerfile
RUN pip install PyMuPDF sentence-transformers scikit-learn nltk
```

### 4.2. Document Ingestion and Pre-processing

**Objective:** Convert PDF documents into a structured, block-level text representation, preserving layout information.

**Steps:**
1.  **Load PDF:** Use `fitz.open(pdf_path)` to open each PDF document.
2.  **Iterate Pages:** Loop through each page of the document.
3.  **Extract Text Blocks with Layout:** For each page, use `page.get_text("dict")` or `page.get_text("blocks")` to extract text along with bounding box coordinates, font size, and font name. This is crucial for layout analysis.
4.  **Normalize Text:** Clean extracted text (e.g., remove extra whitespace, handle hyphens at line breaks).

**Example (Conceptual PyMuPDF usage):**
```python
import fitz # PyMuPDF

def extract_blocks(pdf_path):
    doc = fitz.open(pdf_path)
    all_blocks = []
    for page_num in range(doc.page_count):
        page = doc.load_page(page_num)
        blocks = page.get_text("dict")["blocks"]
        for b in blocks:
            if b["type"] == 0:  # text block
                for line in b["lines"]:
                    for span in line["spans"]:
                        all_blocks.append({
                            "text": span["text"],
                            "font_size": span["size"],
                            "font_name": span["font"],
                            "bbox": span["bbox"],
                            "page": page_num + 1
                        })
    return all_blocks
```

### 4.3. Layout-Aware Sectioning

**Objective:** Group extracted text blocks into logical sections based on visual cues (font size, position) to identify headers and their corresponding content.

**Steps:**
1.  **Identify Potential Headers:** Iterate through `all_blocks`. A block is considered a potential header if its font size is significantly larger than the average body text font size, or if its font is bold/distinctive, or if it's positioned at the beginning of a new logical content flow.
2.  **Section Delimitation:** Define sections by identifying header blocks. Content between two consecutive header blocks (or from a header to the end of the document/page) forms a section.
3.  **Assign Section Titles:** The text of the identified header block becomes the `section_title`.
4.  **Store Section Data:** Each section will include its `document` (filename), `page` numbers, `section_title`, and the concatenated `text` content.

**Heuristics for Header Detection:**
*   **Font Size:** Maintain a running average/median of font sizes. Blocks with font sizes > 1.2 * median_body_font_size are strong candidates.
*   **Vertical Position:** Blocks appearing at the top of a page or after significant vertical space.
*   **Font Weight/Style:** Check for 'Bold' in `font_name` (e.g., 'Times-Bold', 'Arial-BoldMT').

### 4.4. Semantic Representation (Embedding)

**Objective:** Convert textual data (persona+job query, document sections, sub-sections) into numerical embeddings using `all-MiniLM-L6-v2`.

**Steps:**
1.  **Load Model:** Initialize `SentenceTransformer('all-MiniLM-L6-v2')` once at the start of the program.
2.  **Encode Query:** Encode the combined `persona` and `job-to-be-done` string into a single query embedding.
3.  **Encode Sections:** Encode the `text` content of each identified document section into its respective embedding.
4.  **Encode Sentences (for sub-section analysis):** For the top-ranked sections, split their content into sentences (using `nltk.sent_tokenize` or a simple regex) and encode each sentence.

**Model Details (`all-MiniLM-L6-v2`):**
*   **Size:** ~80 MB (PyTorch model file).
*   **Dimensions:** Outputs 384-dimensional embeddings.
*   **Performance:** Optimized for CPU inference.

### 4.5. Relevance Scoring & Ranking

**Objective:** Calculate the similarity between the query and document sections/sentences, then rank them.

**Steps:**
1.  **Cosine Similarity:** Use `sklearn.metrics.pairwise.cosine_similarity` to compute the similarity score between the query embedding and each section/sentence embedding.
    *   `similarity = cosine_similarity(query_embedding.reshape(1, -1), section_embedding.reshape(1, -1))[0][0]`
2.  **Rank Sections:** Sort all extracted sections by their cosine similarity score in descending order. Assign `importance_rank` (1 for highest, 2 for second highest, etc.).

### 4.6. Granular Information Extraction (Sub-section Analysis)

**Objective:** From the most relevant sections, extract the most pertinent sentences.

**Steps:**
1.  **Select Top Sections:** Identify the top `N` sections (e.g., N=5 or 10, configurable) based on their `importance_rank`.
2.  **Sentence Tokenization:** Break down the text of these top sections into individual sentences.
3.  **Sentence Embedding & Scoring:** Embed each sentence and calculate its cosine similarity with the original query embedding.
4.  **Select Top Sentences:** Collect the top `M` sentences (e.g., M=3-5 per relevant section, or a global top M) based on their similarity scores. Concatenate these to form the `refined_text` for `sub-section_analysis`.

### 4.7. Output Generation

**Objective:** Format the extracted and ranked data into the specified JSON structure.

**Structure:**
```json
{
  "metadata": {
    "input_documents": ["list"],
    "persona": "User Persona",
    "job_to_be_done": "Task description"
  },
  "extracted_sections": [
    {
      "document": "source.pdf",
      "section_title": "Title",
      "importance_rank": 1,
      "page_number": 1
    }
  ],
  "subsection_analysis": [
    {
      "document": "source.pdf",
      "refined_text": "Content",
      "page_number": 1
    }
  ]
}
```

**Population:**
*   `metadata`: Populate with input parameters and current timestamp.
*   `extracted_sections`: Populate with data from the ranked sections. Note the `page_number` is now a single integer as per the updated output structure.
*   `sub_section_analysis`: Populate with the `refined_text` and corresponding `document` and `page_number` for the top sentences. Note the `page_number` is now a single integer.

## 5. Performance Optimization Strategies

Given the CPU-only and time constraints, several optimizations will be crucial:

1.  **Batch Processing:** When encoding sections or sentences, process them in batches rather than one by one. `sentence-transformers` supports batch encoding, which significantly speeds up inference on CPU.
2.  **Model Quantization (Optional but Recommended):** While `all-MiniLM-L6-v2` is already small, further quantization (e.g., to INT8) can reduce its size and speed up inference even more, potentially without significant loss in accuracy. This would involve using libraries like `Hugging Face Optimum` with `ONNX Runtime`.
3.  **Efficient PDF Parsing:** `PyMuPDF` is generally very fast. Avoid re-parsing the same PDF multiple times.
4.  **Heuristic Refinement:** The header detection heuristics should be robust but not overly complex to avoid adding significant processing overhead.
5.  **Early Exit for Sub-section Analysis:** Only perform detailed sentence-level analysis on the top `N` sections, not all sections, to save computation.
6.  **Profiling:** Use Python's `cProfile` module to identify performance bottlenecks during development and optimize critical sections of the code.

## 6. Dockerfile and Execution Instructions

**Dockerfile:**

```dockerfile
# Use a lightweight Python base image
FROM python:3.9-slim-buster

# Set working directory
WORKDIR /app

# Copy requirements file and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code
COPY . .

# Command to run the application (assuming a main.py script that handles multi-collection processing)
CMD ["python", "main.py"]
```

**`requirements.txt`:**

```
PymuPDF
sentence-transformers
scikit-learn
nltk
```

**Execution Instructions:**

1.  **Build Docker Image:**
    ```bash
    docker build -t document-intelligence .
    ```
2.  **Run Docker Container:**
    ```bash
    docker run -it --rm -v /path/to/your/Challenge_1b:/app/Challenge_1b document-intelligence
    ```
    *   Replace `/path/to/your/Challenge_1b` with the actual path to your `Challenge_1b` directory containing all collections.
    *   The `main.py` script inside the container should be designed to traverse the `Challenge_1b` directory, find each `Collection` folder, read its `challenge1b_input.json`, process the PDFs in its `PDFs` subfolder, and write the output to `challenge1b_output.json` within the same `Collection` folder.

This detailed plan provides a clear roadmap for implementing the Persona-Driven Document Intelligence system, adhering to all specified constraints and leveraging appropriate technologies.

