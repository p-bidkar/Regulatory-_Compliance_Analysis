"""
Agent 3: Recommendation Generation Module
Generates actionable policy update recommendations with strict citation grounding.
Every recommendation must cite exact chunk IDs and quote ≤15 words verbatim.
"""
import json
from typing import Dict, List, Any, Optional
from pydantic import BaseModel, Field

from config import MODEL_NAME, ANTHROPIC_API_KEY, MOCK_MODE, LLM_PROVIDER
from src.utils import parse_json_safely, logger, validate_citation
from src.agent2_rag_retrieval import RetrievedPolicyChunk
from src.llm_client import get_llm_client


class Citation(BaseModel):
    """Schema for a citation to a policy chunk."""
    chunk_id: str = Field(..., description="ID of the cited policy chunk")
    quoted_text: str = Field(..., description="Verbatim quote from the chunk (max 15 words)")
    policy_section: str = Field("", description="Policy section identifier")


class Recommendation(BaseModel):
    """Schema for a single policy update recommendation."""
    recommendation: str = Field(..., description="Actionable recommendation text")
    citations: List[Citation] = Field(
        default_factory=list, 
        description="Citations supporting this recommendation"
    )
    risk_level: str = Field("medium", description="Risk level: high, medium, or low")
    affected_change: str = Field("", description="Which regulatory change this addresses")
    implementation_priority: int = Field(1, description="Priority 1-5, where 1 is highest")


class RecommendationResult(BaseModel):
    """Schema for Agent 3 output."""
    recommendations: List[Recommendation] = Field(
        default_factory=list,
        description="List of policy update recommendations"
    )
    summary: str = Field("", description="Executive summary of recommended actions")
    total_citations: int = Field(0, description="Total number of citations across all recommendations")


class RecommendationGenerator:
    """
    Agent 3: Generates actionable recommendations with strict citation grounding.
    
    Key requirements:
    - Every recommendation must cite specific policy chunks
    - Citations must include verbatim quotes (≤15 words)
    - Risk levels must be justified
    - Output must be structured JSON
    """
    
    def __init__(self, model_name: str = MODEL_NAME, api_key: str = ANTHROPIC_API_KEY):
        self.model_name = model_name
        self.api_key = api_key
        self.llm_client = get_llm_client()
    
    def _build_policy_context(self, retrieved_chunks: List[RetrievedPolicyChunk]) -> str:
        """Build formatted context from retrieved policy chunks."""
        context_parts = []
        
        for i, chunk in enumerate(retrieved_chunks[:10]):  # Limit context size
            context_parts.append(f"""
[POLICY CHUNK {i+1}]
ID: {chunk.chunk_id}
Section: {chunk.policy_section}
Relevance: {chunk.relevance_score:.2f}
---
{chunk.text[:800]}{'...' if len(chunk.text) > 800 else ''}
---
""")
        
        return '\n'.join(context_parts)
    
    def _build_changes_context(self, changes: List[Dict]) -> str:
        """Build formatted context from detected changes."""
        context_parts = []
        
        for i, change in enumerate(changes[:10]):
            context_parts.append(f"""
[CHANGE {i+1}]
Section: {change.get('section_id', 'Unknown')}
Summary: {change.get('summary', '')}
Type: {change.get('change_type', 'modification')}
Old: {change.get('old_value', 'N/A')}
New: {change.get('new_value', 'N/A')}
Reason: {change.get('reason', '')}
""")
        
        return '\n'.join(context_parts)
    
    def _generate_with_llm(
        self,
        changes: List[Dict],
        retrieved_chunks: List[RetrievedPolicyChunk]
    ) -> List[Recommendation]:
        """Generate recommendations using LLM with strict citation requirements."""
        
        policy_context = self._build_policy_context(retrieved_chunks)
        changes_context = self._build_changes_context(changes)
        
        prompt = f"""You are a compliance officer generating policy update recommendations based on regulatory changes.

Your task is to create ACTIONABLE recommendations for updating internal policies to comply with new regulations.

CRITICAL REQUIREMENTS:
1. Every recommendation MUST cite at least one specific policy chunk by ID
2. Each citation MUST include a verbatim quote of NO MORE THAN 15 words from the cited chunk
3. Quotes must be EXACT - word-for-word from the source
4. Assign risk levels based on: regulatory penalties, implementation complexity, timeline urgency
5. Be specific and actionable - avoid vague language

REGULATORY CHANGES DETECTED:
{changes_context}

RELEVANT POLICY CHUNKS:
{policy_context}

Generate your response as a JSON array with this EXACT structure:
[
  {{
    "recommendation": "string - specific, actionable recommendation",
    "citations": [
      {{
        "chunk_id": "string - exact chunk ID from above",
        "quoted_text": "string - verbatim quote, max 15 words",
        "policy_section": "string - policy section from the chunk"
      }}
    ],
    "risk_level": "string - high/medium/low",
    "affected_change": "string - which change this addresses",
    "implementation_priority": "integer 1-5 (1=highest)"
  }}
]

Guidelines for risk assessment:
- HIGH: New legal requirements, significant penalties, short deadlines, major process changes
- MEDIUM: Moderate updates needed, existing framework can accommodate, reasonable timeline
- LOW: Minor clarifications, already partially compliant, flexible deadline

Return ONLY the JSON array, no other text."""

        try:
            response_text = self.llm_client.generate(prompt)
            
            # Parse JSON response
            parsed_data, error = parse_json_safely(response_text)
            
            if error or not parsed_data:
                logger.warning(f"LLM JSON parsing failed: {error}")
                return self._generate_mock_recommendations(changes, retrieved_chunks)
            
            # Validate and convert to Recommendation objects
            if isinstance(parsed_data, list):
                recommendations = []
                for item in parsed_data:
                    try:
                        rec = Recommendation(**item)
                        
                        # Validate citations exist in provided chunks
                        chunk_ids = {c.chunk_id for c in retrieved_chunks}
                        valid_citations = []
                        for citation in rec.citations:
                            if citation.chunk_id in chunk_ids:
                                # Verify quote length
                                if len(citation.quoted_text.split()) <= 15:
                                    valid_citations.append(citation)
                                else:
                                    logger.warning(f"Citation quote too long: {len(citation.quoted_text.split())} words")
                            else:
                                logger.warning(f"Citation chunk_id not found: {citation.chunk_id}")
                        
                        rec.citations = valid_citations
                        recommendations.append(rec)
                        
                    except Exception as e:
                        logger.warning(f"Failed to validate recommendation: {e}")
                
                return recommendations
            
            return []
            
        except Exception as e:
            logger.error(f"LLM recommendation generation failed: {e}")
            return self._generate_mock_recommendations(changes, retrieved_chunks)
    
    def _generate_mock_recommendations(
        self,
        changes: List[Dict],
        retrieved_chunks: List[RetrievedPolicyChunk]
    ) -> List[Recommendation]:
        """Generate mock recommendations for testing without API."""
        recommendations = []
        
        if not retrieved_chunks:
            return recommendations
        
        for change in changes[:5]:  # Limit to first 5 changes
            # Find relevant chunks
            change_keywords = change.get('summary', '').lower().split()
            best_chunk = None
            best_score = 0
            
            for chunk in retrieved_chunks:
                chunk_text_lower = chunk.text.lower()
                overlap = sum(1 for kw in change_keywords if kw in chunk_text_lower and len(kw) > 3)
                if overlap > best_score:
                    best_score = overlap
                    best_chunk = chunk
            
            if not best_chunk:
                best_chunk = retrieved_chunks[0]  # Fallback to first chunk
            
            # Create a short quote from the chunk
            words = best_chunk.text.split()[:15]
            quote = ' '.join(words)
            
            # Determine risk level based on change type
            change_type = change.get('change_type', '')
            if change_type == 'addition' or 'penalty' in change.get('summary', '').lower():
                risk_level = 'high'
                priority = 1
            elif 'reporting' in change.get('summary', '').lower() or 'frequency' in change.get('summary', '').lower():
                risk_level = 'medium'
                priority = 2
            else:
                risk_level = 'medium'
                priority = 3
            
            recommendations.append(Recommendation(
                recommendation=f"Update {best_chunk.policy_section or 'relevant policy'} to address: {change.get('summary', 'regulatory change')}",
                citations=[Citation(
                    chunk_id=best_chunk.chunk_id,
                    quoted_text=quote,
                    policy_section=best_chunk.policy_section
                )],
                risk_level=risk_level,
                affected_change=change.get('section_id', 'Unknown'),
                implementation_priority=priority
            ))
        
        return recommendations
    
    def generate_recommendations(
        self,
        changes: List[Dict],
        retrieved_chunks: List[RetrievedPolicyChunk]
    ) -> RecommendationResult:
        """
        Generate policy update recommendations.
        
        Args:
            changes: List of detected changes from Agent 1 (as dicts)
            retrieved_chunks: List of relevant policy chunks from Agent 2
            
        Returns:
            RecommendationResult with structured recommendations
        """
        logger.info(f"Generating recommendations for {len(changes)} changes using {len(retrieved_chunks)} policy chunks")
        
        if not changes:
            return RecommendationResult(
                recommendations=[],
                summary="No substantive regulatory changes detected. No policy updates required at this time.",
                total_citations=0
            )
        
        # Convert Pydantic models to dicts for processing
        changes_dicts = [c.dict() if hasattr(c, 'dict') else c for c in changes]
        
        # Generate recommendations
        recommendations = self._generate_with_llm(changes_dicts, retrieved_chunks)
        
        # Calculate total citations
        total_citations = sum(len(rec.citations) for rec in recommendations)
        
        # Generate summary
        if recommendations:
            high_risk = sum(1 for r in recommendations if r.risk_level == 'high')
            medium_risk = sum(1 for r in recommendations if r.risk_level == 'medium')
            low_risk = sum(1 for r in recommendations if r.risk_level == 'low')
            
            summary = f"Generated {len(recommendations)} policy update recommendations. "
            summary += f"Risk distribution: {high_risk} high, {medium_risk} medium, {low_risk} low priority. "
            summary += f"Total citations: {total_citations}."
        else:
            summary = "No recommendations generated. Review input data and retry."
        
        return RecommendationResult(
            recommendations=recommendations,
            summary=summary,
            total_citations=total_citations
        )
    
    def validate_recommendations(
        self, 
        recommendations: List[Recommendation],
        source_chunks: Dict[str, str]
    ) -> Dict[str, Any]:
        """Validate that all citations are properly grounded."""
        from src.utils import validate_recommendations
        
        recs_dicts = [r.dict() for r in recommendations]
        return validate_recommendations(recs_dicts, source_chunks)


def create_recommendation_generator() -> RecommendationGenerator:
    """Factory function to create a recommendation generator."""
    return RecommendationGenerator()
