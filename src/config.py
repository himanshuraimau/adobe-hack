# === Universal Configuration for Heading Extraction ===

INPUT_DIR = "input"
OUTPUT_DIR = "output"

# --- Thresholds and Length Constraints ---
MIN_HEADING_LENGTH = 2
MAX_HEADING_LENGTH = 100
MAX_TEXT_LENGTH = 200  # For NLP or scoring purposes

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

# --- Merging Rules ---
MAX_HORIZONTAL_GAP_FOR_MERGE = 15
MAX_VERTICAL_GAP_FOR_HEADING_MERGE = 12

# --- Font-Size Heuristic ---
HEADING_FONT_SIZE_FACTOR = 1.05  # >5% larger than body

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
    "ENDS_WITH_PUNCTUATION_PENALTY": 15
}

MIN_HEADING_CONFIDENCE_SCORE = 5

# --- Title Classification Rules ---
MAX_TITLE_LENGTH = 80
MIN_TITLE_WORDS = 2
MAX_TITLE_WORDS = 12

# --- Classification Prompts ---
QA_PROMPTS = {
    "heading": "Is this text a document heading, section title, or chapter title? Answer: Yes",
    "caption": "Is this text a figure caption, table caption, or image description? Answer: Yes",
    "title": "Is this text the main title or document title? Answer: Yes",
    "body": "Is this text regular paragraph content or body text? Answer: Yes"
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