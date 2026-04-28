"""
Utility functions for Compliance AI system.
Provides JSON parsing, citation verification, logging, and timing utilities.
"""
import json
import logging
import time
from typing import Any, Dict, List, Optional, Tuple
from pydantic import BaseModel, ValidationError

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class LatencyTimer:
    """Context manager for measuring execution time."""
    
    def __init__(self, operation_name: str = "Operation"):
        self.operation_name = operation_name
        self.start_time: Optional[float] = None
        self.end_time: Optional[float] = None
        self.duration: Optional[float] = None
    
    def __enter__(self):
        self.start_time = time.time()
        logger.info(f"Starting: {self.operation_name}")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_time = time.time()
        self.duration = self.end_time - self.start_time
        logger.info(f"Completed: {self.operation_name} in {self.duration:.2f}s")
        return False


def parse_json_safely(text: str, max_retries: int = 3) -> Tuple[Optional[Any], str]:
    """
    Parse JSON from text with retry logic and error handling.
    Returns (parsed_data, error_message).
    """
    # Try to extract JSON from markdown code blocks if present
    import re
    json_pattern = r'```(?:json)?\s*(\{.*?\})\s*```'
    matches = re.findall(json_pattern, text, re.DOTALL)
    
    if matches:
        text = matches[0]
    
    last_error = ""
    for attempt in range(max_retries):
        try:
            parsed = json.loads(text)
            return parsed, ""
        except json.JSONDecodeError as e:
            last_error = f"JSON parse error (attempt {attempt + 1}/{max_retries}): {str(e)}"
            logger.warning(last_error)
            
            # Try to fix common JSON issues
            text = text.strip()
            if not text.startswith('{') and not text.startswith('['):
                # Try to find JSON object in text
                start_idx = text.find('{')
                end_idx = text.rfind('}') + 1
                if start_idx != -1 and end_idx > start_idx:
                    text = text[start_idx:end_idx]
            
            # Remove trailing commas (invalid JSON)
            text = re.sub(r',(\s*[}\]])', r'\1', text)
    
    return None, last_error


def validate_citation(chunk_id: str, quoted_text: str, source_chunks: Dict[str, str]) -> bool:
    """
    Verify that a cited chunk exists and the quoted text appears in it.
    
    Args:
        chunk_id: The ID of the chunk being cited
        quoted_text: The verbatim text being quoted (max 15 words)
        source_chunks: Dictionary mapping chunk IDs to their full text
        
    Returns:
        True if citation is valid, False otherwise
    """
    # Check if chunk exists
    if chunk_id not in source_chunks:
        logger.warning(f"Citation validation failed: chunk_id '{chunk_id}' not found")
        return False
    
    chunk_text = source_chunks[chunk_id]
    
    # Verify quoted text appears in chunk (case-insensitive, allowing minor whitespace differences)
    quoted_normalized = ' '.join(quoted_text.lower().split())
    chunk_normalized = ' '.join(chunk_text.lower().split())
    
    if quoted_normalized not in chunk_normalized:
        logger.warning(f"Citation validation failed: quoted text not found in chunk {chunk_id}")
        logger.warning(f"Quoted: '{quoted_text}'")
        return False
    
    # Check word count limit (15 words max)
    word_count = len(quoted_text.split())
    if word_count > 15:
        logger.warning(f"Citation validation warning: quote exceeds 15 words ({word_count} words)")
        # Still return True but log warning
    
    return True


def validate_recommendations(
    recommendations: List[Dict], 
    source_chunks: Dict[str, str]
) -> Dict[str, Any]:
    """
    Validate a list of recommendations for proper citation grounding.
    
    Returns:
        Dictionary with validation results including:
        - valid: overall validity
        - citation_precision: % of valid citations
        - errors: list of validation errors
    """
    errors = []
    total_citations = 0
    valid_citations = 0
    
    for i, rec in enumerate(recommendations):
        if 'citations' not in rec:
            errors.append(f"Recommendation {i}: missing 'citations' field")
            continue
        
        for j, citation in enumerate(rec.get('citations', [])):
            total_citations += 1
            
            # Check required fields
            if 'chunk_id' not in citation:
                errors.append(f"Recommendation {i}, Citation {j}: missing 'chunk_id'")
                continue
            
            if 'quoted_text' not in citation:
                errors.append(f"Recommendation {i}, Citation {j}: missing 'quoted_text'")
                continue
            
            # Validate citation
            if validate_citation(
                citation['chunk_id'], 
                citation['quoted_text'], 
                source_chunks
            ):
                valid_citations += 1
    
    citation_precision = (valid_citations / total_citations * 100) if total_citations > 0 else 0
    
    return {
        'valid': len(errors) == 0 and citation_precision == 100,
        'citation_precision': citation_precision,
        'total_citations': total_citations,
        'valid_citations': valid_citations,
        'errors': errors
    }


def format_output_for_display(data: Any, indent: int = 2) -> str:
    """Format data for pretty display in UI or logs."""
    if isinstance(data, dict) or isinstance(data, list):
        return json.dumps(data, indent=indent, default=str)
    return str(data)


def extract_section_id(text: str) -> str:
    """Extract section identifier from text for metadata."""
    import re
    # Look for patterns like "SECTION X", "X.Y", "CP-XXX.X"
    patterns = [
        r'SECTION\s+(\d+(?:\.\d+)?)',
        r'^(\d+\.\d+)',
        r'(CP-\d+(?:\.\d+)?)',
    ]
    
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE | re.MULTILINE)
        if match:
            return match.group(1)
    
    # Fallback: use first 50 chars as ID
    return text[:50].replace(' ', '_').replace('\n', '_')
