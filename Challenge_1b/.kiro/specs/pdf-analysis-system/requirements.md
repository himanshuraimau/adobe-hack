# Requirements Document

## Introduction

This system processes multiple PDF document collections and extracts relevant content based on specific personas and use cases. The system analyzes PDFs using semantic similarity to rank and extract the most relevant sections for each persona's job-to-be-done task, outputting structured JSON results.

## Requirements

### Requirement 1

**User Story:** As a system user, I want to process multiple document collections automatically, so that I can analyze different sets of PDFs for various personas and tasks.

#### Acceptance Criteria

1. WHEN the system starts THEN it SHALL discover all collection directories (Collection 1, Collection 2, Collection 3)
2. WHEN a collection is found THEN the system SHALL read its challenge1b_input.json file
3. WHEN processing collections THEN the system SHALL handle each collection independently

### Requirement 2

**User Story:** As a system user, I want PDFs to be parsed and sectioned intelligently, so that content can be analyzed at a granular level.

#### Acceptance Criteria

1. WHEN a PDF is processed THEN the system SHALL extract text with layout information using PyMuPDF
2. WHEN text blocks are extracted THEN the system SHALL identify section headers based on font size and styling
3. WHEN sections are identified THEN the system SHALL group content between headers into logical sections

### Requirement 3

**User Story:** As a system user, I want content to be ranked by relevance to the persona and task, so that the most important information is prioritized.

#### Acceptance Criteria

1. WHEN persona and job-to-be-done are provided THEN the system SHALL create a combined query embedding
2. WHEN document sections are extracted THEN the system SHALL generate embeddings for each section
3. WHEN embeddings are created THEN the system SHALL calculate cosine similarity scores between query and sections
4. WHEN similarity scores are calculated THEN the system SHALL rank sections by importance (1 = highest relevance)

### Requirement 4

**User Story:** As a system user, I want detailed analysis of the most relevant sections, so that I can get refined, sentence-level insights.

#### Acceptance Criteria

1. WHEN sections are ranked THEN the system SHALL select top 5-10 sections for detailed analysis
2. WHEN top sections are selected THEN the system SHALL split them into individual sentences
3. WHEN sentences are extracted THEN the system SHALL rank sentences by relevance to the query
4. WHEN sentence ranking is complete THEN the system SHALL compile top sentences into refined_text

### Requirement 5

**User Story:** As a system user, I want results formatted in the specified JSON structure, so that output is consistent and usable.

#### Acceptance Criteria

1. WHEN processing is complete THEN the system SHALL generate metadata with input documents, persona, and job description
2. WHEN sections are ranked THEN the system SHALL format extracted_sections with document, title, rank, and page number
3. WHEN sentence analysis is complete THEN the system SHALL format subsection_analysis with document, refined_text, and page number
4. WHEN JSON is formatted THEN the system SHALL save output to challenge1b_output.json in each collection directory

### Requirement 6

**User Story:** As a system user, I want the system to use lightweight models that work on CPU, so that it runs efficiently without GPU requirements.

#### Acceptance Criteria

1. WHEN the system initializes THEN it SHALL use sentence-transformers with all-MiniLM-L6-v2 model
2. WHEN embeddings are generated THEN the system SHALL process them on CPU only
3. WHEN model is loaded THEN it SHALL be under 1GB total size constraint