# === Universal Configuration for Heading Extraction ===

INPUT_DIR = "input"
OUTPUT_DIR = "output"

# --- Thresholds and Length Constraints ---
MIN_HEADING_LENGTH = 2
MAX_HEADING_LENGTH = 100
MAX_TEXT_LENGTH = 200  # For NLP or scoring purposes

# --- Title Exclusion Patterns ---
TITLE_EXCLUSION_PATTERNS = {
    'starts_with_number': r'^\d+\.',  # "1. Introduction"
    'starts_with_section_symbol': r'^[§¶]',  # Section symbols
    'excessive_punctuation': r'.*[-:]{4,}',  # More than 3 dashes/colons
    'date_formats': [
        r'\d{4}-\d{2}-\d{2}',  # YYYY-MM-DD
        r'\d{1,2}/\d{1,2}/\d{4}',  # MM/DD/YYYY
        r'^(january|february|march|april|may|june|july|august|september|october|november|december)\s+\d{1,2},?\s+\d{4}$'
    ],
    'urls_emails': [
        r'https?://',
        r'www\.',
        r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}'
    ],
    'copyright_notices': r'©|\(c\)|copyright|all rights reserved',
    'table_figure_captions': r'^(table|figure|fig\.?)\s*\d+:?|^(image|photo)\s+\d+',  # More specific patterns
    'starts_lowercase': r'^[a-z]',  # Unless common articles
    'common_articles': ['a', 'an', 'the', 'and', 'or', 'but', 'for', 'nor', 'so', 'yet']
}

# --- General Numbering Patterns (No document-specific prefixes) ---
NUMBERING_PATTERNS = [
    r'^\d+\.\s+',              # 1. Heading
    r'^\d+\.\d+\.\s+',        # 1.1. Heading
    r'^\d+\.\d+\.\d+\.\s+', # 1.1.1. Heading
    r'^[A-Z]\.\s+',             # A. Heading
    r'^(chapter|section|part)\s+\d+',
    r'^\([a-z]\)',              # (a) Heading
    r'^\([0-9]\)',              # (1) Heading
    r'^[IVXLC]+\.\s+',          # Roman numeral I. Heading
]

# --- Filters to Remove Noise ---
EXCLUDE_PATTERNS = [
    r'^(address|phone|email|date|time|location):',
    r'^(www\.|http)',
    r'\(see\s+(figure|table|appendix)',
    r'^[a-z]\)\s+.*',
    r'\d{1,2}\s+(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)',
    r'^-+$',
    r'^\([^)]*\)$',
    r'^\d{2,}$',
    r'(required|please|visit|attend|note|source|adapted from)',
    r'(developing|providing|supporting|ensuring|promoting)\s+.*',
    r'(the.*will|this.*will|it.*will).*',
    r'\w+\s+(who|that|which|can|should|would|could|will|shall|may|might)\s+\w+.*[.!?]$',
]

# --- Caption Patterns (exclude from headings) ---
CAPTION_PATTERNS = [
    r'^(figure|fig\.?|table|tbl\.?)\s+\d+',
    r'^(image|photo|chart|graph|diagram)\s+\d*',
    r'source:',
    r'note:',
    r'adapted from:'
]

# --- Heuristic Rules ---
HEADING_END_PUNCTUATION = ('.', ':', ';', ',')
TITLE_END_PUNCTUATION = ('.', ':', ';', ',')

# --- Merging Rules ---
MAX_HORIZONTAL_GAP_FOR_MERGE = 15
MAX_VERTICAL_GAP_FOR_HEADING_MERGE = 12

# --- Font-Size Heuristic ---
HEADING_FONT_SIZE_FACTOR = 1.05  # >5% larger than body
TITLE_FONT_SIZE_FACTOR = 1.2 # >20% larger than body

# --- Scoring Weights ---
SCORING_WEIGHTS = {
    "FONT_SIZE_MULTIPLIER": 15,
    "IS_BOLD_BONUS": 20,
    "NUMBERING_BONUS": 40,
    "VERTICAL_SPACE_BONUS": 20,
    "CENTERED_BONUS": 25,
    "ALL_CAPS_BONUS": 10,
    "DISTINCT_FONT_BONUS": 10,
    "LINE_LENGTH_PENALTY_FACTOR": 0.05,
    "ENDS_WITH_PUNCTUATION_PENALTY": 15,
    "TITLE_KEYWORD_BONUS": 50
}

MIN_HEADING_CONFIDENCE_SCORE = 5
MIN_TITLE_CONFIDENCE_SCORE = 10

# --- Title Detection Configuration ---
TITLE_QUALITY_INDICATORS = {
    'good_length_range': (5, 120),  # Min and max characters for title
    'max_words': 15,  # Maximum words in title
    'min_words': 1,   # Minimum words in title
    'position_bonus_range': 10,  # First N blocks get position bonus
    'font_size_multiplier_threshold': 1.15,  # Must be 15% larger than body
    'centering_tolerance': 0.2,  # 20% tolerance for center alignment
}

# --- Title Exclusion Patterns ---
TITLE_EXCLUSION_PATTERNS = {
    'starts_with_number': r'^\d+\.',  # "1. Introduction"
    'starts_with_section_symbol': r'^[§¶]',  # Section symbols
    'excessive_punctuation': r'.*[-:]{4,}',  # More than 3 dashes/colons
    'date_formats': [
        r'\d{4}-\d{2}-\d{2}',  # YYYY-MM-DD
        r'\d{1,2}/\d{1,2}/\d{4}',  # MM/DD/YYYY
        r'^(january|february|march|april|may|june|july|august|september|october|november|december)\s+\d{1,2},?\s+\d{4}$'
    ],
    'urls_emails': [
        r'https?://',
        r'www\.',
        r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}'
    ],
    'copyright_notices': r'©|\(c\)|copyright|all rights reserved',
    'table_figure_captions': r'^(table|figure|fig\.?|image|chart|graph)\s*\d*',
    'starts_lowercase': r'^[a-z]',  # Unless common articles
    'common_articles': ['a', 'an', 'the', 'and', 'or', 'but', 'for', 'nor', 'so', 'yet']
}

# --- Title Scoring Bonuses ---
TITLE_SCORING = {
    'position_bonus': 25,  # Bonus for appearing early in document
    'large_font_bonus': 30,  # Bonus for significantly larger font
    'bold_bonus': 20,  # Bonus for bold text
    'italic_bonus': 15,  # Bonus for italic text
    'title_case_bonus': 15,  # Bonus for Title Case
    'centered_bonus': 25,  # Bonus for centered text
    'title_keywords_bonus': 20,  # Bonus for title-like keywords
    'isolation_bonus': 10,  # Bonus for being isolated (surrounded by whitespace)
}

# --- Title Keywords ---
TITLE_KEYWORDS = [
    'introduction', 'overview', 'summary', 'report', 'analysis', 'study', 'research',
    'guide', 'manual', 'handbook', 'proposal', 'plan', 'strategy', 'framework',
    'methodology', 'approach', 'assessment', 'evaluation', 'review', 'survey'
]

# --- Hierarchy Configuration ---
HIERARCHY_CONFIG = {
    'font_size_tolerance': 0.5,  # Points tolerance for font size comparison
    'indentation_threshold': 10,  # Pixels threshold for indentation levels
    'dynamic_threshold_samples': 10,  # Number of samples for dynamic threshold calculation
    'min_font_size_difference': 1.0,  # Minimum font size difference between levels
    'consistency_weight': 0.3,  # Weight for style consistency analysis
}

# --- Numbering Pattern Priorities ---
NUMBERING_PATTERN_LEVELS = {
    r'^\d+\.\s+': 1,              # 1. -> H1
    r'^\d+\.\d+\.\s+': 2,         # 1.1. -> H2
    r'^\d+\.\d+\.\d+\.\s+': 3,    # 1.1.1. -> H3
    r'^[A-Z]\.\s+': 1,            # A. -> H1
    r'^[a-z]\.\s+': 2,            # a. -> H2
    r'^\([0-9]+\)': 2,            # (1) -> H2
    r'^\([a-z]\)': 3,             # (a) -> H3
    r'^[IVXLC]+\.\s+': 1,         # I. -> H1
    r'^(chapter|section|part)\s+\d+': 1,  # Chapter 1 -> H1
}

# --- Style Consistency Patterns ---
STYLE_CONSISTENCY_FEATURES = [
    'font_family',
    'font_size',
    'is_bold',
    'is_italic',
    'text_color',
    'indentation_level'
]

# --- Title Quality Indicators (Additional) ---
TITLE_QUALITY_INDICATORS['min_alphabetic_ratio'] = 0.5  # At least 50% alphabetic characters
TITLE_QUALITY_INDICATORS['max_special_char_ratio'] = 0.3  # Max 30% special characters

# --- Enhanced Exclude Patterns ---
ENHANCED_EXCLUDE_PATTERNS = [
    # Numbers and sections
    r'^\d+\.\s*$',  # Just "1."
    r'^[§¶]\s*\d+',  # Section symbols
    r'^\d+\.\d+\.\d+\s*$',  # Just numbering
    
    # Excessive punctuation
    r'.*[-:]{4,}',  # 4 or more dashes/colons
    r'^[-=_]{3,}$',  # Separator lines
    
    # URLs and emails
    r'https?://',
    r'www\.',
    r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}',
    
    # Copyright and legal
    r'©|\(c\)|copyright|all rights reserved',
    r'proprietary|confidential|internal use only',
    
    # Table/Figure captions - more specific patterns
    r'^(table|figure|fig\.?)\s*\d+:?',  # Table 1, Figure 2, etc.
    r'^(image|photo)\s+\d+',  # Image 1, Photo 2, etc.
    r'^chart\s+\d+',  # Chart 1, Chart 2 (but not "Chart Analysis")
    r'^(graph|diagram)\s+\d+',  # Graph 1, Diagram 2
    r'^source:|^note:|^adapted from:',
    
    # Form elements and signatures
    r'^\w+:\s*_+\s*$',  # Fields with underscores
    r'^\[\s*\]\s*',  # Checkboxes
    r'^\(\s*\)\s*',  # Radio buttons
    r'date.*signature',  # Signature lines
    r'signature.*date',  # Date and signature
    r'^\s*(name|date|signature|place).*\.(.*servant|officer|applicant)',  # Form signature areas
    
    # Form field patterns
    r'^(name|designation|service|date|whether|home\s+town)',  # Common form field starters
    r'^\w+\s+of\s+(the\s+)?(government\s+)?servant',  # "Name of the Government Servant"
    r'pay\s*\+\s*si\s*\+\s*npa',  # Government salary terms
    r'whether\s+(permanent|temporary|wife|husband)',  # Form questions
    r'is\s+to\s+be\s+availed',  # Form completion phrases
    r'amount\s+of\s+(advance|payment)\s+required',  # Financial form fields
    r'particulars\s+furnished\s+above',  # Form declaration language
    r'undertake\s+to\s+(produce|refund)',  # Legal undertaking language
    
    # Navigation and references
    r'^(see|refer to|continued on|page \d+)',
    r'^(next|previous|back|forward|home)',
]

# --- Local Context Correction Rules ---
HIERARCHY_CORRECTION_RULES = {
    'max_level_jump': 1,  # Don't jump more than 1 level (H1->H3 should be H1->H2)
    'sequence_analysis_window': 5,  # Analyze sequences of up to 5 headings
    'correction_confidence_threshold': 0.7,  # Threshold for applying corrections
}

# --- Document Structure Patterns ---
DOCUMENT_STRUCTURE_INDICATORS = {
    'toc_markers': ['table of contents', 'contents', 'index'],
    'section_prefixes': ['chapter', 'section', 'part', 'article', 'appendix'],
    'conclusion_markers': ['conclusion', 'summary', 'references', 'bibliography'],
    'header_footer_patterns': [
        r'page \d+ of \d+',
        r'^\d+$',  # Page numbers
        r'^(header|footer)$',
        r'confidential$',
        r'draft$'
    ]
}

# --- Classification Prompts ---
QA_PROMPTS = {
    "heading": """Given the text: "{text}", which is on page {page_num}, has a font size of {font_size}, is {bolded_italic_state}, and is preceded by the text "{preceding_text}", is this text a document heading, section title, or chapter title?  Explain your reasoning in one sentence and then answer "Yes" or "No".""",

    "caption": """Given the text: "{text}", which is on page {page_num}, is near a {image_table_context}, and has a font size of {font_size}, is this text a figure caption, table caption, or image description? Explain your reasoning in one sentence and then answer "Yes" or "No".""",

    "title": """Given the text: "{text}", which is on page {page_num}, has a font size of {font_size}, is {bolded_italic_state}, and is at the {top_bottom_position} of the document, is this text the main title or document title? Explain your reasoning in one sentence and then answer "Yes" or "No".""",

    "body": """Given the text: "{text}", which is on page {page_num}, has a font size of {font_size}, is {indented_state}, and is part of a longer paragraph, is this text regular paragraph content or body text? Explain your reasoning in one sentence and then answer "Yes" or "No"."""
}

# --- Confidence Thresholds for SLM Verification ---
CONFIDENCE_THRESHOLDS = {
    "title": 0.30,
    "heading": 0.15,
    "caption": 0.50,
    "body": 0.25
}

# --- Hierarchy Rules (Font + Numbering based) ---
HIERARCHY_RULES = {
    "font_size_difference_threshold": 2,
    "position_weight": 0.3,
    "style_consistency_bonus": 5,
    "numbering_level_mapping": {
        r'^\d+\.\s+': 'H1',
        r'^\d+\.\d+\.\s+': 'H2',
        r'^\d+\.\d+\.\d+\.\s+': 'H3'
    }
}

# --- Performance Settings ---
PERFORMANCE_SETTINGS = {
    "max_candidates_per_page": 50,
    "max_processing_time_per_file": 30,
    "min_confidence_for_hierarchy": 0.6,
    "enable_debug_output": False
}

# --- Title-specific Configuration ---
TITLE_SCORING_WEIGHTS = {
    "POSITION_BONUS": {
        "first": 50,
        "early": 30,
        "mid": 10,
        "late": -10
    },
    "FONT_SIZE_MULTIPLIER": 5,
    "BOLD_BONUS": 20,
    "LENGTH_BONUS": {
        "optimal_min": 2,
        "optimal_max": 8,
        "penalty_factor": 2
    },
    "CENTER_ALIGNMENT_BONUS": 15,
    "CAPITALIZATION_BONUS": 10,
    "KEYWORD_BONUS": 25,
    "PUNCTUATION_PENALTY": 10,
    "LONG_SENTENCE_PENALTY": 20
}

# --- Title Indicator Patterns ---
TITLE_INDICATOR_PATTERNS = [
    r'\b(manual|guide|handbook|report|study|analysis|proposal|plan|framework|strategy)\b',
    r'\b(white\s*paper|technical\s*document|specification|requirements)\b',
    r'\b(annual\s*report|quarterly\s*report|progress\s*report)\b',
    r'\b(user\s*guide|installation\s*guide|quick\s*start)\b',
    r'\b(policy|procedure|protocol|standard|guideline)\b'
]

# --- Non-Title Patterns (expanded) ---
NON_TITLE_PATTERNS = [
    # Fragment patterns
    r'quest for pr', r'r proposal', r'rfp:', r'request f',
    r'rsvp:', r'closed toed shoes', r'climbing',
    r'application form', r'grant of ltc',
    r'foundation level extensions',
    
    # Navigation/reference patterns
    r'^(page|p\.|pp\.)\s*\d+',
    r'^(see|refer to|as shown in)',
    r'(continued|cont\.d|see above|see below)',
    
    # Metadata patterns
    r'^\d{1,2}[/-]\d{1,2}[/-]\d{2,4}', # Dates
    r'^(version|ver\.?|v\.?)\s*\d+',
    r'^(draft|final|revised|updated)',
    
    # Contact/administrative patterns
    r'(email|phone|fax|address|contact)',
    r'(copyright|©|all rights reserved)',
    r'(confidential|proprietary|internal use)',
    
    # Form/template patterns
    r'(name:|title:|date:|signature:)',
    r'(check one|select all that apply)',
    r'(yes/no|true/false)',
    
    # Academic patterns
    r'¹|²|³|⁴|⁵',  # Superscript numbers used in affiliations
    r'university of|institute of|college of',
    r'department of|school of|faculty of',
    
    # Journal/publication patterns
    r'journal of|proceedings of|conference on',
    r'vol\.|volume|issue|pp\.',
    
    # Common fragments
    r'^(the|a|an)\s+\w+$', # Single articles with one word
    r'^\w{1,3}$', # Very short words
    r'^[\d\s\-\.]+$' # Only numbers, spaces, dashes, dots
]

# --- Bad Title Validation Patterns ---
BAD_TITLE_PATTERNS = [
    r'quest for pr', r'r proposal', r'request f', r'rsvp:', 
    r'closed toed shoes', r'climbing',
    r'^(page|chapter|section)\s+\d+',
    r'table of contents',
    r'^\d+$', # Just a number
    r'^[^\w\s]*$', # Only special characters
    r'¹|²|³|⁴|⁵',  # Superscript numbers
    r'university of|institute of|college of',
    r'department of|school of|faculty of',
    r'journal of|proceedings of|conference on',
    r'vol\.|volume|issue|pp\.',
]

# --- Title Quality Indicators ---
TITLE_QUALITY_INDICATORS = {
    "good_length_range": (3, 80),
    "optimal_word_count": (2, 12),
    "max_special_char_ratio": 0.3,
    "min_alphabetic_ratio": 0.5,
    "preferred_case_patterns": [
        "Title Case",
        "UPPERCASE",
        "Mixed Case"
    ]
}

# --- Heading Extraction Configuration ---
HEADING_EXTRACTION_CONFIG = {
    # Font size thresholds for heading detection
    "MIN_HEADING_FONT_MULTIPLIER": 1.05,  # 5% larger than body
    "H1_FONT_MULTIPLIER": 1.3,  # 30% larger than body for H1
    "H2_FONT_MULTIPLIER": 1.2,  # 20% larger than body for H2
    "H3_FONT_MULTIPLIER": 1.1,  # 10% larger than body for H3
    
    # Scoring weights for heading detection
    "SCORING_WEIGHTS": {
        "FONT_SIZE_BONUS": 3,
        "BOLD_BONUS": 25,
        "NUMBERING_BONUS": 60,
        "CAPITALIZATION_BONUS": 15,
        "ALL_CAPS_BONUS": 20,
        "HEADING_KEYWORDS_BONUS": 30,
        "PUNCTUATION_PENALTY": 30,
        "LENGTH_PENALTY": 10,
        "ISOLATION_BONUS": 15,  # Space above/below
        "CONSISTENCY_BONUS": 20,  # Similar to other headings
    },
    
    # Length constraints
    "MIN_HEADING_LENGTH": 2,
    "MAX_HEADING_LENGTH": 150,
    "OPTIMAL_WORD_COUNT_MIN": 1,
    "OPTIMAL_WORD_COUNT_MAX": 8,
    
    # Hierarchy assignment thresholds
    "MIN_HEADING_SCORE": 15,
    "H1_MIN_SCORE": 50,
    "H2_MIN_SCORE": 35,
    "H3_MIN_SCORE": 20,
}

# --- Hierarchical Numbering Patterns ---
HIERARCHICAL_NUMBERING_PATTERNS = {
    'H1': [
        r'^\d+\.\s+',  # 1. Introduction
        r'^[A-Z]\.\s+',  # A. Overview
        r'^(Chapter|CHAPTER)\s+\d+',  # Chapter 1
        r'^(Section|SECTION)\s+\d+',  # Section 1
        r'^[IVXLC]+\.\s+',  # I. Introduction (Roman numerals)
        r'^\(\d+\)\s*',  # (1) Introduction
        r'^\d+\s+[A-Z]',  # 1 INTRODUCTION (number + caps)
    ],
    'H2': [
        r'^\d+\.\d+\s+',  # 1.1 Background
        r'^\d+\.\d+\.\s+',  # 1.1. Background
        r'^[A-Z]\.\d+\s+',  # A.1 Section
        r'^\([a-z]\)\s*',  # (a) Subsection
        r'^\d+\.\d+\s*[A-Z]',  # 1.1 Title Case
        r'^[a-z]\.\s+',  # a. Subsection
    ],
    'H3': [
        r'^\d+\.\d+\.\d+\s+',  # 1.1.1 Details
        r'^\d+\.\d+\.\d+\.\s+',  # 1.1.1. Details
        r'^[a-z]\)\s*',  # a) Sub-point
        r'^\([i-v]+\)\s*',  # (i) Roman lower
        r'^\d+\.\d+\.\d+\s*[A-Z]',  # 1.1.1 Title Case
        r'^[a-z]\.[a-z]\s+',  # a.b Sub-subsection
    ]
}

# --- Heading Quality Indicators ---
HEADING_QUALITY_PATTERNS = {
    "STRONG_INDICATORS": [
        r'\b(introduction|overview|summary|conclusion|abstract|methodology)\b',
        r'\b(background|objectives|scope|purpose|goals|results)\b',
        r'\b(discussion|findings|recommendations|implementation)\b',
        r'\b(references|appendix|bibliography|acknowledgments)\b',
        r'\b(technical|specifications|requirements|standards)\b',
    ],
    "WEAK_INDICATORS": [
        r'\b(the|and|or|but|with|for|from|about|under|over)\b',
        r'\b(this|that|these|those|here|there|where|when)\b',
        r'\b(very|quite|rather|somewhat|really|actually)\b',
    ],
    "EXCLUDE_PATTERNS": [
        r'\b(www\.|http|email|phone|address|contact)\b',
        r'\b(page|figure|table|image|diagram)\s+\d+',
        r'\b(copyright|confidential|proprietary|draft)\b',
        r'\b(see|refer|please|visit|click|download)\b',
    ]
}

# --- Machine Learning Features for Hierarchy Assignment ---
ML_HIERARCHY_FEATURES = {
    "FONT_FEATURES": {
        "use_relative_size": True,
        "use_size_clustering": True,
        "cluster_tolerance": 1.5,  # Font size difference tolerance
    },
    "STYLE_FEATURES": {
        "bold_weight": 0.3,
        "italic_weight": 0.1,
        "font_family_weight": 0.2,
        "color_weight": 0.1,
    },
    "POSITION_FEATURES": {
        "vertical_spacing_weight": 0.2,
        "horizontal_alignment_weight": 0.15,
        "page_position_weight": 0.1,
    },
    "CONTENT_FEATURES": {
        "numbering_weight": 0.4,
        "capitalization_weight": 0.2,
        "keyword_weight": 0.25,
        "length_weight": 0.15,
    }
}

# --- Universal Pattern Lists for Title/Heading Extraction ---
NO_TITLE_PATTERNS = [
    r'application form', r'grant application', r'form', r'flyer', r'invitation', r'party', r'hope', r'ltc', r'topjump'
]
EXCLUDE_TITLE_PATTERNS = [
    r'chapter', r'table of contents', r'contents', r'preface', r'acknowledgments', r'praise', r'revision', r'history',
    r'ceo', r'former', r'head', r'involved', r'development', r'application', r'media6degrees', r'media6'
]
FORMAL_TITLE_PATTERNS = [
    r'(introduction|overview|summary|conclusion|abstract|contents|bibliography|references|appendix)'
]
HEADING_EMPHASIS_PATTERNS = [
    r'(introduction|overview|summary|conclusion|abstract|contents|bibliography|references|appendix)'
]
EXCLUDE_HEADING_PATTERNS = [
    r'chapter', r'table of contents', r'contents', r'preface', r'acknowledgments', r'praise', r'revision', r'history',
    r'ceo', r'former', r'head', r'involved', r'development', r'application', r'media6degrees', r'media6'
]
HIGH_PRIORITY_HEADING_PATTERNS = [
    r'^(introduction|overview|summary|conclusion|abstract|contents|bibliography|references|appendix)'
]
MAJOR_SECTION_PATTERNS = [
    r'(introduction|overview|summary|conclusion|abstract|contents|bibliography|references|appendix)'
]
MIN_ALL_CAPS_LENGTH = 4
MAX_ALL_CAPS_LENGTH = 40
MAX_NLP_TEXT_LENGTH = 200