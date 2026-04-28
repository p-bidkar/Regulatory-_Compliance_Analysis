"""
Configuration module for Compliance AI system.
Loads environment variables and provides default settings.
"""
import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Base directory
BASE_DIR = Path(__file__).parent

# API Configuration
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY", "")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")  # Fallback or alternative

# LLM Provider Selection: "anthropic", "nvidia", "openai"
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "anthropic")

# Model Configuration - provider-specific defaults
if LLM_PROVIDER == "nvidia":
    MODEL_NAME = os.getenv("MODEL_NAME", "meta/llama-3.1-405b-instruct")
    EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "nvidia/nv-embedqa-e5-v5")
elif LLM_PROVIDER == "openai":
    MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4o")
    EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-large")
else:  # anthropic (default)
    MODEL_NAME = os.getenv("MODEL_NAME", "claude-3-5-sonnet-20241022")
    EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-large")

# NVIDIA Base URL (always defined for import compatibility)
NVIDIA_BASE_URL = os.getenv("NVIDIA_BASE_URL", "https://integrate.api.nvidia.com/v1")

# Mock mode for testing without API key
MOCK_MODE = os.getenv("MOCK_MODE", "false").lower() == "true"

# Retrieval Settings
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "512"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "50"))
TOP_K_RETRIEVAL = int(os.getenv("TOP_K_RETRIEVAL", "5"))

# Vector DB Settings
VECTOR_DB_PATH = os.getenv("VECTOR_DB_PATH", str(BASE_DIR / "data" / "vector_db"))

# Data paths
DATA_DIR = BASE_DIR / "data"
TEST_REGULATION_PATH = DATA_DIR / "test_regulation.txt"
TEST_POLICIES_PATH = DATA_DIR / "test_policies.txt"

# Logging
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

# Retry Configuration
MAX_RETRIES = int(os.getenv("MAX_RETRIES", "3"))
RETRY_DELAY = float(os.getenv("RETRY_DELAY", "1.0"))

# Validation
def validate_config():
    """Validate configuration and return warnings if needed."""
    warnings = []
    
    # Check API key based on selected provider
    if LLM_PROVIDER == "nvidia" and not NVIDIA_API_KEY and not MOCK_MODE:
        warnings.append("Warning: NVIDIA_API_KEY not set. Set MOCK_MODE=true for testing.")
    elif LLM_PROVIDER == "openai" and not OPENAI_API_KEY and not MOCK_MODE:
        warnings.append("Warning: OPENAI_API_KEY not set. Set MOCK_MODE=true for testing.")
    elif LLM_PROVIDER == "anthropic" and not ANTHROPIC_API_KEY and not MOCK_MODE:
        warnings.append("Warning: ANTHROPIC_API_KEY not set. Set MOCK_MODE=true for testing.")
    
    return warnings
