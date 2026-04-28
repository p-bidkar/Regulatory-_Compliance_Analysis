# Compliance AI - Multi-Agent Regulatory Document Analysis System

## Overview

This MVP implements a multi-agent AI system that automates regulatory compliance document analysis for financial institutions. The system processes new regulatory documents through three specialized agents coordinated via LangGraph:

1. **Agent 1 (Change Detection)**: Identifies substantive changes vs boilerplate in new regulations
2. **Agent 2 (RAG Policy Matching)**: Retrieves relevant internal policy sections using hybrid search + re-ranking
3. **Agent 3 (Recommendation Generation)**: Generates actionable policy update recommendations with strict citation grounding

## Project Structure

```
compliance_ai/
├── .env.example              # Environment variable template
├── requirements.txt          # Python dependencies
├── config.py                 # Configuration loader
├── data/
│   ├── test_regulation.txt   # Sample regulation documents
│   └── test_policies.txt     # Sample company policies
├── src/
│   ├── ingestion.py          # Document parsing, chunking, vector storage
│   ├── agent1_change_detection.py    # Change detection agent
│   ├── agent2_rag_retrieval.py       # RAG retrieval agent
│   ├── agent3_recommendation.py      # Recommendation generation agent
│   ├── orchestrator.py       # LangGraph workflow coordination
│   └── utils.py              # Utility functions
├── app.py                    # Streamlit UI
├── eval.py                   # Evaluation script
└── README.md                 # This file
```

## Quick Start

### 1. Installation

```bash
cd compliance_ai

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configuration

```bash
# Copy environment template
cp .env.example .env

# Edit .env and configure your preferred LLM provider:

# Option A: NVIDIA API (recommended for cost-effective Llama 3.1, Mistral)
LLM_PROVIDER=nvidia
NVIDIA_API_KEY=nvapi-your_key_here
MODEL_NAME=meta/llama-3.1-405b-instruct
EMBEDDING_MODEL=nvidia/nv-embedqa-e5-v5

# Option B: Anthropic API (Claude models)
LLM_PROVIDER=anthropic
ANTHROPIC_API_KEY=your_key_here
MODEL_NAME=claude-3-5-sonnet-20241022

# Option C: OpenAI API
LLM_PROVIDER=openai
OPENAI_API_KEY=your_key_here
MODEL_NAME=gpt-4o

# For testing without API key, set:
MOCK_MODE=true
```

### 3. Run the Streamlit UI

```bash
streamlit run app.py
```

The UI will open at `http://localhost:8501`. You can:
- Upload regulation and policy documents
- Click "Load Test Data" to use sample documents
- Run the analysis and view results

### 4. Run Evaluation

```bash
python eval.py
```

This runs the complete pipeline on test data and outputs:
- Retrieval Accuracy (Recall, Precision, F1)
- Citation Precision
- Latency metrics

## Architecture

### Workflow (LangGraph StateGraph)

```
┌─────────────────┐
│  Input State    │
│  - new_reg      │
│  - baseline_reg │
│  - policy_docs  │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Agent 1:        │
│ Change Detection│
└────────┬────────┘
         │
    ┌────┴────┐
    │ Changes │
    │ == []?  │
    └────┬────┘
         │
    Yes  │  No
    ┌────┴────┐
    │         │
    ▼         ▼
┌───────┐  ┌─────────────────┐
│ Final │  │ Agent 2:        │
│ Output│  │ Policy Retrieval│
└───┬───┘  └────────┬────────┘
    │               │
    │          ┌────┴────┐
    │          │         │
    │          ▼         │
    │    ┌─────────────────┐
    │    │ Agent 3:        │
    │    │ Recommendations │
    │    └────────┬────────┘
    │             │
    └─────────────┼──────┘
                  │
                  ▼
         ┌─────────────────┐
         │ Final Output    │
         │ + Validation    │
         └─────────────────┘
```

### Key Design Decisions

1. **Multi-Provider LLM Support**: Unified interface supporting NVIDIA (Llama 3.1, Mistral), Anthropic (Claude), and OpenAI (GPT-4) via a single configuration switch.

2. **Semantic Chunking**: Documents are split by section headers first, then into overlapping chunks for better context preservation.

3. **Hybrid Retrieval**: Combines BM25 keyword search with TF-IDF semantic similarity, then re-ranks with weighted scoring.

4. **Strict Citation Grounding**: Every recommendation must cite specific chunk IDs with verbatim quotes (≤15 words). Citations are programmatically validated.

5. **Mock Mode Fallback**: When no API key is provided, the system generates heuristic-based mock responses for testing.

6. **Pydantic Validation**: All LLM outputs are validated against strict schemas before being used downstream.

## Configuration Options

| Variable | Default | Description |
|----------|---------|-------------|
| `LLM_PROVIDER` | `anthropic` | Provider: "nvidia", "anthropic", or "openai" |
| `NVIDIA_API_KEY` | - | NVIDIA API key (required if LLM_PROVIDER=nvidia) |
| `ANTHROPIC_API_KEY` | - | Anthropic API key (required if LLM_PROVIDER=anthropic) |
| `OPENAI_API_KEY` | - | OpenAI API key (required if LLM_PROVIDER=openai) |
| `MODEL_NAME` | varies | Model name (provider-specific defaults) |
| `EMBEDDING_MODEL` | varies | Embedding model (provider-specific) |
| `CHUNK_SIZE` | `512` | Characters per chunk |
| `CHUNK_OVERLAP` | `50` | Overlap between chunks |
| `TOP_K_RETRIEVAL` | `5` | Policy chunks to retrieve |
| `MOCK_MODE` | `false` | Enable mock responses for testing |

### NVIDIA-Specific Models

When using `LLM_PROVIDER=nvidia`, recommended models include:

| Model | Use Case | MODEL_NAME value |
|-------|----------|------------------|
| Llama 3.1 405B | Best quality, complex reasoning | `meta/llama-3.1-405b-instruct` |
| Llama 3.1 70B | Good balance of speed/quality | `meta/llama-3.1-70b-instruct` |
| Mistral Large 2 | Strong instruction following | `mistralai/mistral-large-2-instruct` |
| NV-EmbedQA-E5 | High-quality embeddings | `nvidia/nv-embedqa-e5-v5` |

## Evaluation Metrics

The `eval.py` script computes:

### Retrieval Accuracy
- **Recall**: % of ground-truth policy sections found in top-k results
- **Precision**: % of retrieved sections that are relevant
- **F1 Score**: Harmonic mean of precision and recall

### Citation Precision
- **Citation Precision**: % of citations that exactly match source chunks
- Validates chunk_id exists and quoted text appears verbatim

### Latency
- Total pipeline execution time
- Per-step breakdown (ingestion, change detection, retrieval, recommendation)

## Known Limitations (MVP)

1. **Embeddings**: Uses TF-IDF as proxy for semantic similarity. Production should use actual embeddings (OpenAI, sentence-transformers).

2. **PDF Parsing**: Currently supports plain text only. PDF support requires `pypdf` integration.

3. **Vector DB**: Uses ChromaDB in persistent mode but without true vector embeddings in mock mode.

4. **Ground Truth**: Evaluation uses manually defined ground truth mappings. Real evaluation would need expert-labeled data.

5. **Error Recovery**: Basic retry logic for JSON parsing. Production needs more robust error handling.

6. **Scalability**: In-memory processing. Production needs async processing and job queues.

## Next-Step Extensions

### Short-term (Week 2-4)
- [ ] Add real embedding models (OpenAI/Sentence Transformers)
- [ ] Implement PDF parsing with `pypdf`
- [ ] Add cross-encoder re-ranking (ms-marco, bge-reranker)
- [ ] Expand test dataset with more regulation/policy pairs
- [ ] Add unit tests for each agent

### Medium-term (Week 5-8)
- [ ] Implement human-in-the-loop review UI
- [ ] Add export functionality (PDF, Word reports)
- [ ] Integrate with regulatory APIs (SEC, FDIC, Federal Reserve)
- [ ] Add conversation history for iterative refinement
- [ ] Implement caching for repeated queries

### Long-term (Week 9-12)
- [ ] Multi-tenant architecture with user management
- [ ] Fine-tune embeddings on compliance domain data
- [ ] Add audit trail and version control for recommendations
- [ ] Implement real-time monitoring dashboard
- [ ] Add support for multiple jurisdictions (EU, UK, APAC)

## Tech Stack

- **Orchestration**: LangGraph (stateful workflows)
- **Document Processing**: LlamaIndex (chunking, indexing)
- **Vector Store**: ChromaDB (in-memory/persistent)
- **LLM Providers**: 
  - NVIDIA API (Llama 3.1, Mistral Large)
  - Anthropic Claude API
  - OpenAI API (GPT-4)
- **Unified Client**: Custom `src/llm_client.py` with automatic provider switching
- **UI**: Streamlit
- **Evaluation**: Custom Python metrics

## License

MIT License - See LICENSE file for details.

## Contributing

This is a semester project. Please coordinate with the team before making changes.

---

**Built for the Financial Compliance Automation course - Fall 2024**
