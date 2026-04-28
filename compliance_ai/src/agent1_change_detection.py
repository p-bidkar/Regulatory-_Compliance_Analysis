"""
Agent 1: Change Detection Module
Identifies substantive changes between baseline and new regulations.
Uses chunk comparison + LLM analysis to filter boilerplate from real changes.
"""
import json
import re
from typing import Dict, List, Any, Optional, Tuple
from pydantic import BaseModel, Field

from config import MODEL_NAME, ANTHROPIC_API_KEY, MOCK_MODE
from src.utils import parse_json_safely, logger


class ChangedSection(BaseModel):
    """Schema for a detected change section."""
    section_id: str = Field(..., description="Identifier of the changed section")
    summary: str = Field(..., description="Brief summary of what changed")
    reason: str = Field(..., description="Why this is considered a substantive change")
    old_value: Optional[str] = Field(None, description="Previous value/text if applicable")
    new_value: Optional[str] = Field(None, description="New value/text if applicable")
    change_type: str = Field(default="modification", description="Type: modification, addition, deletion")


class ChangeDetectionResult(BaseModel):
    """Schema for Agent 1 output."""
    changed_sections: List[ChangedSection] = Field(
        default_factory=list, 
        description="List of sections with substantive changes"
    )
    analysis_summary: str = Field("", description="Overall summary of changes detected")
    confidence_score: float = Field(0.0, description="Confidence in change detection (0-1)")


class ChangeDetector:
    """
    Detects substantive changes between baseline and new regulations.
    
    Strategy:
    1. Chunk both documents by section
    2. Compute similarity scores between corresponding chunks
    3. Flag low-similarity sections for LLM review
    4. LLM summarizes only substantive changes (ignoring boilerplate)
    """
    
    def __init__(self, model_name: str = MODEL_NAME, api_key: str = ANTHROPIC_API_KEY):
        self.model_name = model_name
        self.api_key = api_key
        self.client = None
        
        if not MOCK_MODE and api_key:
            try:
                import anthropic
                self.client = anthropic.Anthropic(api_key=api_key)
            except ImportError:
                logger.warning("anthropic package not installed, using mock mode")
    
    def _chunk_by_section(self, text: str) -> Dict[str, str]:
        """Split document into sections for comparison."""
        sections = {}
        
        # Pattern to match section headers
        section_pattern = r'(SECTION\s+\d+[^:\n]*:|POLICY\s+SECTION\s+CP-\d+[^:\n]*:)'
        
        parts = re.split(section_pattern, text, flags=re.IGNORECASE)
        
        current_section_id = "preamble"
        current_content = []
        
        for part in parts:
            if re.match(section_pattern, part, re.IGNORECASE):
                # Save previous section
                if current_content:
                    sections[current_section_id] = ' '.join(current_content).strip()
                current_section_id = part.strip().rstrip(':')
                current_content = []
            elif part.strip():
                current_content.append(part.strip())
        
        # Save last section
        if current_content:
            sections[current_section_id] = ' '.join(current_content).strip()
        
        return sections
    
    def _compute_similarity(self, text1: str, text2: str) -> float:
        """
        Compute simple cosine similarity between two texts.
        For MVP, uses TF-IDF vectorization.
        """
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity
        
        try:
            vectorizer = TfidfVectorizer(stop_words='english')
            tfidf_matrix = vectorizer.fit_transform([text1, text2])
            similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
            return float(similarity)
        except Exception as e:
            logger.warning(f"Similarity computation failed: {e}")
            return 0.5  # Default moderate similarity
    
    def _identify_changed_sections(
        self, 
        baseline_sections: Dict[str, str], 
        new_sections: Dict[str, str],
        similarity_threshold: float = 0.7
    ) -> List[Tuple[str, str, float]]:
        """
        Identify sections that have changed significantly.
        
        Returns:
            List of tuples: (section_id, baseline_text, new_text, similarity_score)
        """
        changed = []
        
        all_section_ids = set(baseline_sections.keys()) | set(new_sections.keys())
        
        for section_id in all_section_ids:
            baseline_text = baseline_sections.get(section_id, "")
            new_text = new_sections.get(section_id, "")
            
            # Check if section is new or deleted
            if not baseline_text and new_text:
                changed.append((section_id, "", new_text, 0.0))  # New section
            elif baseline_text and not new_text:
                changed.append((section_id, baseline_text, "", 0.0))  # Deleted section
            else:
                # Compute similarity for existing sections
                similarity = self._compute_similarity(baseline_text, new_text)
                
                if similarity < similarity_threshold:
                    changed.append((section_id, baseline_text, new_text, similarity))
        
        # Sort by similarity (lowest first = most changed)
        changed.sort(key=lambda x: x[3])
        
        return changed
    
    def _analyze_changes_with_llm(
        self, 
        changed_sections: List[Tuple[str, str, str, float]]
    ) -> List[ChangedSection]:
        """Use LLM to analyze and summarize substantive changes."""
        
        if MOCK_MODE or not self.client:
            return self._mock_change_analysis(changed_sections)
        
        # Build prompt for LLM
        sections_context = ""
        for i, (section_id, old_text, new_text, similarity) in enumerate(changed_sections[:10]):
            sections_context += f"\n\n--- Section {i+1}: {section_id} ---\n"
            sections_context += f"Similarity Score: {similarity:.2f}\n"
            sections_context += f"PREVIOUS TEXT:\n{old_text[:500]}...\n" if len(old_text) > 500 else f"PREVIOUS TEXT:\n{old_text}\n"
            sections_context += f"NEW TEXT:\n{new_text[:500]}...\n" if len(new_text) > 500 else f"NEW TEXT:\n{new_text}\n"
        
        prompt = f"""You are a regulatory compliance expert analyzing changes between two versions of a financial regulation.

Your task is to identify SUBSTANTIVE changes that would require policy updates. Ignore minor wording changes, formatting differences, or boilerplate language.

Focus on changes that affect:
- Numerical thresholds (ratios, percentages, dollar amounts)
- Reporting requirements (frequency, deadlines, signatories)
- Scope/applicability (asset thresholds, institution types)
- New obligations or prohibitions
- Enforcement mechanisms or penalties

Here are the sections with detected differences:

{sections_context}

Return your analysis as a JSON array with this exact structure:
[
  {{
    "section_id": "string - the section identifier",
    "summary": "string - concise summary of what changed",
    "reason": "string - why this is substantively important",
    "old_value": "string or null - specific old value if applicable",
    "new_value": "string or null - specific new value if applicable",
    "change_type": "string - one of: modification, addition, deletion"
  }}
]

Only include genuinely substantive changes. Return an empty array if no substantive changes found."""

        try:
            response = self.client.messages.create(
                model=self.model_name,
                max_tokens=2000,
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )
            
            response_text = response.content[0].text
            
            # Parse JSON response
            parsed_data, error = parse_json_safely(response_text)
            
            if error or not parsed_data:
                logger.warning(f"LLM JSON parsing failed: {error}")
                return self._mock_change_analysis(changed_sections)
            
            # Validate and convert to ChangedSection objects
            if isinstance(parsed_data, list):
                changes = []
                for item in parsed_data:
                    try:
                        change = ChangedSection(**item)
                        changes.append(change)
                    except Exception as e:
                        logger.warning(f"Failed to validate change: {e}")
                return changes
            
            return []
            
        except Exception as e:
            logger.error(f"LLM analysis failed: {e}")
            return self._mock_change_analysis(changed_sections)
    
    def _mock_change_analysis(self, changed_sections: List[Tuple[str, str, str, float]]) -> List[ChangedSection]:
        """Generate mock change analysis for testing without API."""
        changes = []
        
        for section_id, old_text, new_text, similarity in changed_sections:
            if not new_text:
                continue  # Skip deletions in mock mode
            
            # Simple heuristic to extract key differences
            old_numbers = re.findall(r'\d+(?:\.\d+)?%|\$\d+(?:,\d+)*(?:\.\d+)?|\d+\s+(?:days?|months?|years?|billion|million)', old_text.lower())
            new_numbers = re.findall(r'\d+(?:\.\d+)?%|\$\d+(?:,\d+)*(?:\.\d+)?|\d+\s+(?:days?|months?|years?|billion|million)', new_text.lower())
            
            old_value = old_numbers[0] if old_numbers else None
            new_value = new_numbers[0] if new_numbers else None
            
            change_type = "addition" if not old_text else "modification"
            
            changes.append(ChangedSection(
                section_id=section_id,
                summary=f"Section {section_id} updated with new requirements",
                reason=f"Text similarity is {similarity:.0%}, indicating substantive changes",
                old_value=old_value,
                new_value=new_value,
                change_type=change_type
            ))
        
        return changes
    
    def detect_changes(
        self, 
        baseline_regulation: str, 
        new_regulation: str
    ) -> ChangeDetectionResult:
        """
        Main method to detect changes between baseline and new regulations.
        
        Args:
            baseline_regulation: Text of the baseline/original regulation
            new_regulation: Text of the new/updated regulation
            
        Returns:
            ChangeDetectionResult with identified changes
        """
        logger.info("Starting change detection analysis...")
        
        # Step 1: Chunk both documents by section
        baseline_sections = self._chunk_by_section(baseline_regulation)
        new_sections = self._chunk_by_section(new_regulation)
        
        logger.info(f"Baseline sections: {len(baseline_sections)}, New sections: {len(new_sections)}")
        
        # Step 2: Identify sections with significant differences
        changed_sections = self._identify_changed_sections(
            baseline_sections, 
            new_sections,
            similarity_threshold=0.8
        )
        
        logger.info(f"Found {len(changed_sections)} sections with potential changes")
        
        # Step 3: Use LLM to analyze and filter for substantive changes
        substantive_changes = self._analyze_changes_with_llm(changed_sections)
        
        logger.info(f"Identified {len(substantive_changes)} substantive changes")
        
        # Step 4: Calculate confidence score
        if changed_sections:
            avg_similarity = sum(cs[3] for cs in changed_sections) / len(changed_sections)
            confidence = min(1.0, (1.0 - avg_similarity) * 1.5)  # Scale to 0-1
        else:
            confidence = 0.0
        
        # Step 5: Generate summary
        if substantive_changes:
            summary = f"Detected {len(substantive_changes)} substantive changes across the regulation. "
            summary += f"Key areas affected: {', '.join(set(c.section_id[:50] for c in substantive_changes[:5]))}"
        else:
            summary = "No substantive changes detected. Changes appear to be primarily editorial or boilerplate."
        
        return ChangeDetectionResult(
            changed_sections=substantive_changes,
            analysis_summary=summary,
            confidence_score=confidence
        )


def run_change_detection(baseline_text: str, new_text: str) -> ChangeDetectionResult:
    """Convenience function to run change detection."""
    detector = ChangeDetector()
    return detector.detect_changes(baseline_text, new_text)
