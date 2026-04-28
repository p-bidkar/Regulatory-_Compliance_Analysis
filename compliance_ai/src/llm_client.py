"""
Multi-provider LLM client supporting Anthropic, NVIDIA, and OpenAI.
Provides a unified interface for all agents in the compliance system.
"""
import json
import logging
from typing import Any, Dict, List, Optional
from tenacity import retry, stop_after_attempt, wait_exponential

from config import (
    ANTHROPIC_API_KEY,
    NVIDIA_API_KEY,
    OPENAI_API_KEY,
    LLM_PROVIDER,
    MODEL_NAME,
    NVIDIA_BASE_URL,
    MOCK_MODE,
    MAX_RETRIES,
)

logger = logging.getLogger(__name__)


class LLMClient:
    """
    Unified LLM client supporting multiple providers.
    
    Usage:
        client = LLMClient()
        response = client.generate(prompt, system_prompt="You are helpful...")
    """
    
    def __init__(self, provider: Optional[str] = None):
        self.provider = provider or LLM_PROVIDER
        self.model_name = MODEL_NAME
        self.mock_mode = MOCK_MODE
        
        # Initialize provider-specific clients
        self._anthropic_client = None
        self._openai_client = None
        
        if self.provider == "anthropic" and not self.mock_mode:
            try:
                import anthropic
                self._anthropic_client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
                logger.info(f"Initialized Anthropic client with model: {self.model_name}")
            except ImportError:
                logger.warning("anthropic package not installed, falling back to mock mode")
                self.mock_mode = True
                
        elif self.provider in ["nvidia", "openai"] and not self.mock_mode:
            try:
                from openai import OpenAI
                if self.provider == "nvidia":
                    self._openai_client = OpenAI(
                        api_key=NVIDIA_API_KEY,
                        base_url=NVIDIA_BASE_URL
                    )
                    logger.info(f"Initialized NVIDIA client with model: {self.model_name}")
                else:  # openai
                    self._openai_client = OpenAI(api_key=OPENAI_API_KEY)
                    logger.info(f"Initialized OpenAI client with model: {self.model_name}")
            except ImportError:
                logger.warning("openai package not installed, falling back to mock mode")
                self.mock_mode = True
        
        if self.mock_mode:
            logger.info("Running in MOCK MODE - using heuristic responses")
    
    @retry(
        stop=stop_after_attempt(MAX_RETRIES),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        reraise=True
    )
    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.1,
        max_tokens: int = 2048,
        response_format: Optional[str] = None  # "json" for JSON mode
    ) -> str:
        """
        Generate text using the configured LLM provider.
        
        Args:
            prompt: User prompt
            system_prompt: Optional system instruction
            temperature: Sampling temperature (lower = more deterministic)
            max_tokens: Maximum tokens to generate
            response_format: "json" for JSON mode (OpenAI/NVIDIA only)
            
        Returns:
            Generated text response
        """
        if self.mock_mode:
            return self._mock_generate(prompt, system_prompt)
        
        if self.provider == "anthropic":
            return self._generate_anthropic(
                prompt, system_prompt, temperature, max_tokens
            )
        elif self.provider in ["nvidia", "openai"]:
            return self._generate_openai_compatible(
                prompt, system_prompt, temperature, max_tokens, response_format
            )
        else:
            raise ValueError(f"Unsupported provider: {self.provider}")
    
    def _generate_anthropic(
        self,
        prompt: str,
        system_prompt: Optional[str],
        temperature: float,
        max_tokens: int
    ) -> str:
        """Generate using Anthropic Claude API."""
        messages = [{"role": "user", "content": prompt}]
        
        kwargs = {
            "model": self.model_name,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": messages,
        }
        
        if system_prompt:
            kwargs["system"] = system_prompt
        
        response = self._anthropic_client.messages.create(**kwargs)
        return response.content[0].text
    
    def _generate_openai_compatible(
        self,
        prompt: str,
        system_prompt: Optional[str],
        temperature: float,
        max_tokens: int,
        response_format: Optional[str]
    ) -> str:
        """Generate using OpenAI-compatible API (NVIDIA or OpenAI)."""
        messages = []
        
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        
        messages.append({"role": "user", "content": prompt})
        
        kwargs = {
            "model": self.model_name,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        
        # Enable JSON mode for structured output (NVIDIA/OpenAI)
        if response_format == "json":
            kwargs["response_format"] = {"type": "json_object"}
        
        response = self._openai_client.chat.completions.create(**kwargs)
        return response.choices[0].message.content
    
    def _mock_generate(self, prompt: str, system_prompt: Optional[str]) -> str:
        """
        Generate mock responses for testing without API keys.
        Uses simple heuristics to return plausible structured output.
        """
        logger.debug("Using mock generation")
        
        # Detect expected output format from prompt
        prompt_lower = prompt.lower()
        
        if "changed_sections" in prompt_lower or "substantive changes" in prompt_lower:
            # Mock change detection output
            return json.dumps({
                "changed_sections": [
                    {
                        "section_id": "SEC-001",
                        "summary": "Updated capital requirements from 8% to 10% for tier-1 assets",
                        "reason": "Regulatory enhancement to strengthen financial stability"
                    },
                    {
                        "section_id": "SEC-003",
                        "summary": "New reporting frequency requirement: quarterly instead of annual",
                        "reason": "Increased oversight for risk monitoring"
                    }
                ]
            })
        
        elif "recommendation" in prompt_lower and "citations" in prompt_lower:
            # Mock recommendation output
            return json.dumps({
                "recommendations": [
                    {
                        "recommendation": "Update capital adequacy policy to reflect new 10% minimum requirement",
                        "citations": [
                            {
                                "chunk_id": "policy_chunk_1",
                                "quoted_text": "minimum capital requirement",
                                "policy_section": "Capital Adequacy Framework"
                            }
                        ],
                        "risk_level": "high"
                    },
                    {
                        "recommendation": "Revise reporting procedures to accommodate quarterly submissions",
                        "citations": [
                            {
                                "chunk_id": "policy_chunk_3",
                                "quoted_text": "regulatory reporting timeline",
                                "policy_section": "Reporting Requirements"
                            }
                        ],
                        "risk_level": "medium"
                    }
                ]
            })
        
        elif "retrieve" in prompt_lower or "relevant" in prompt_lower:
            # Mock retrieval output
            return json.dumps({
                "relevant_chunks": [
                    {
                        "chunk_id": "policy_chunk_1",
                        "content": "The institution shall maintain a minimum tier-1 capital ratio of 8%...",
                        "metadata": {"section": "Capital Adequacy", "relevance_score": 0.92}
                    },
                    {
                        "chunk_id": "policy_chunk_2",
                        "content": "Annual regulatory reports must be submitted within 90 days...",
                        "metadata": {"section": "Reporting", "relevance_score": 0.85}
                    }
                ]
            })
        
        # Default mock response
        return json.dumps({
            "status": "mock_response",
            "prompt_preview": prompt[:100] + "..."
        })
    
    def generate_structured(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        output_schema: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """
        Generate structured JSON output with validation.
        
        Args:
            prompt: User prompt
            system_prompt: Optional system instruction
            output_schema: Expected JSON schema (for documentation/validation)
            
        Returns:
            Parsed JSON dictionary
        """
        response_text = self.generate(
            prompt,
            system_prompt,
            response_format="json"
        )
        
        # Parse JSON response
        try:
            # Remove markdown code blocks if present
            import re
            json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', response_text, re.DOTALL)
            if json_match:
                response_text = json_match.group(1)
            
            return json.loads(response_text)
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON response: {e}")
            logger.error(f"Response text: {response_text[:500]}")
            raise ValueError(f"Invalid JSON response from LLM: {e}")


def get_llm_client() -> LLMClient:
    """Factory function to get configured LLM client."""
    return LLMClient()
