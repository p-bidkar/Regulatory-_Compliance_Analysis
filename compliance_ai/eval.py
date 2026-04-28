"""
Evaluation Script for Compliance AI System.
Runs the pipeline on test data and computes metrics:
- Retrieval Accuracy
- Citation Precision
- Latency per regulation
"""
import json
import time
from pathlib import Path
from typing import Dict, List, Any, Tuple
from datetime import datetime

# Add parent directory to path
import sys
sys.path.insert(0, str(Path(__file__).parent))

from config import MOCK_MODE
from src.ingestion import DocumentIngester
from src.agent1_change_detection import ChangeDetector
from src.agent2_rag_retrieval import create_policy_retriever
from src.agent3_recommendation import RecommendationGenerator
from src.utils import validate_citation, logger


# Ground truth data for evaluation
# Format: {change_description: [expected_policy_section_ids]}
GROUND_TRUTH = {
    "capital ratio increase": ["CP-001", "CP-001.2"],
    "leverage ratio change": ["CP-001", "CP-001.2"],
    "reporting frequency monthly": ["CP-001.4"],
    "LCR requirement 110%": ["CP-002", "CP-002.2"],
    "NSFR threshold 50 billion": ["CP-002", "CP-002.3"],
    "board members expertise": ["CP-003", "CP-003.1"],
    "stress testing climate": ["CP-004", "CP-004.1"],
    "record retention 10 years": ["CP-005", "CP-005.4"],
    "penalties increased": ["CP-006", "CP-006.3"],
    "climate risk disclosure": ["CP-007", "CP-007.1"],
}


def load_test_data() -> Tuple[str, str, str]:
    """Load test regulation and policy documents."""
    # Load from text files directly
    data_dir = Path(__file__).parent / "data"
    
    with open(data_dir / "test_regulation.txt", 'r') as f:
        content = f.read()
        import re
        baseline_match = re.search(r'BASELINE_REGULATION\s*=\s*"""(.*?)"""', content, re.DOTALL)
        new_match = re.search(r'NEW_REGULATION\s*=\s*"""(.*?)"""', content, re.DOTALL)
        
        BASELINE_REGULATION = baseline_match.group(1).strip() if baseline_match else ""
        NEW_REGULATION = new_match.group(1).strip() if new_match else ""
    
    with open(data_dir / "test_policies.txt", 'r') as f:
        policy_content = f.read()
        policy_match = re.search(r'COMPANY_POLICIES\s*=\s*"""(.*?)"""', policy_content, re.DOTALL)
        COMPANY_POLICIES = policy_match.group(1).strip() if policy_match else ""
    
    return BASELINE_REGULATION, NEW_REGULATION, COMPANY_POLICIES


def run_evaluation_pipeline(
    baseline_reg: str,
    new_reg: str,
    policy_docs: str
) -> Dict[str, Any]:
    """
    Run the complete evaluation pipeline.
    
    Returns:
        Dictionary containing results and metrics
    """
    start_time = time.time()
    results = {
        'timestamp': datetime.now().isoformat(),
        'mock_mode': MOCK_MODE,
        'steps': {}
    }
    
    # Step 1: Document Ingestion
    logger.info("=" * 50)
    logger.info("STEP 1: Document Ingestion")
    step_start = time.time()
    
    ingester = DocumentIngester()
    ingester.clear_collection()
    
    reg_chunks = ingester.ingest_text(new_reg, "test_regulation", "regulation")
    policy_chunks = ingester.ingest_text(policy_docs, "test_policies", "policy")
    
    results['steps']['ingestion'] = {
        'duration_seconds': time.time() - step_start,
        'regulation_chunks': len(reg_chunks),
        'policy_chunks': len(policy_chunks)
    }
    logger.info(f"Ingested {len(reg_chunks)} regulation chunks, {len(policy_chunks)} policy chunks")
    
    # Step 2: Change Detection
    logger.info("=" * 50)
    logger.info("STEP 2: Change Detection")
    step_start = time.time()
    
    detector = ChangeDetector()
    change_result = detector.detect_changes(baseline_reg, new_reg)
    
    changes = change_result.changed_sections
    results['steps']['change_detection'] = {
        'duration_seconds': time.time() - step_start,
        'num_changes_detected': len(changes),
        'confidence_score': change_result.confidence_score,
        'changes': [c.dict() for c in changes]
    }
    logger.info(f"Detected {len(changes)} substantive changes")
    
    # Step 3: Policy Retrieval
    logger.info("=" * 50)
    logger.info("STEP 3: Policy Retrieval")
    step_start = time.time()
    
    retriever = create_policy_retriever(policy_chunks)
    change_summaries = [c.summary for c in changes]
    retrieval_result = retriever.retrieve_for_changes(change_summaries)
    
    retrieved_chunks = retrieval_result.retrieved_chunks
    results['steps']['retrieval'] = {
        'duration_seconds': time.time() - step_start,
        'num_retrieved': len(retrieved_chunks),
        'method': retrieval_result.retrieval_method,
        'retrieved_chunk_ids': [c.chunk_id for c in retrieved_chunks]
    }
    logger.info(f"Retrieved {len(retrieved_chunks)} policy chunks")
    
    # Step 4: Recommendation Generation
    logger.info("=" * 50)
    logger.info("STEP 4: Recommendation Generation")
    step_start = time.time()
    
    generator = RecommendationGenerator()
    rec_result = generator.generate_recommendations(
        changes=[c.dict() for c in changes],
        retrieved_chunks=retrieved_chunks
    )
    
    recommendations = rec_result.recommendations
    results['steps']['recommendations'] = {
        'duration_seconds': time.time() - step_start,
        'num_recommendations': len(recommendations),
        'total_citations': rec_result.total_citations,
        'summary': rec_result.summary
    }
    logger.info(f"Generated {len(recommendations)} recommendations with {rec_result.total_citations} citations")
    
    # Total latency
    total_latency = time.time() - start_time
    results['total_latency_seconds'] = total_latency
    logger.info(f"Total pipeline latency: {total_latency:.2f}s")
    
    return results, ingester, changes, retrieved_chunks, recommendations


def compute_retrieval_accuracy(
    detected_changes: List,
    retrieved_chunks: List,
    ground_truth: Dict[str, List[str]]
) -> Dict[str, float]:
    """
    Compute retrieval accuracy metrics.
    
    Measures what percentage of ground-truth policy sections
    appear in the top-k retrieved results.
    """
    # Extract policy section IDs from retrieved chunks
    retrieved_sections = set()
    for chunk in retrieved_chunks:
        section = chunk.policy_section or chunk.metadata.get('section_id', '')
        if section:
            retrieved_sections.add(section.split('.')[0])  # Get base section (e.g., CP-001)
    
    # Map changes to expected sections
    total_expected = 0
    total_found = 0
    
    for change in detected_changes:
        summary_lower = change.summary.lower() if hasattr(change, 'summary') else change.get('summary', '').lower()
        
        # Find matching ground truth entries
        for gt_key, expected_sections in ground_truth.items():
            if gt_key.lower() in summary_lower:
                total_expected += len(expected_sections)
                
                # Check how many were retrieved
                for expected in expected_sections:
                    base_expected = expected.split('.')[0]
                    if base_expected in retrieved_sections:
                        total_found += 1
    
    # Calculate metrics
    recall = total_found / total_expected if total_expected > 0 else 0.0
    
    # Also compute precision (what fraction of retrieved are relevant)
    total_retrieved = len(retrieved_chunks)
    precision = total_found / total_retrieved if total_retrieved > 0 else 0.0
    
    # F1 score
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return {
        'retrieval_recall': round(recall, 4),
        'retrieval_precision': round(precision, 4),
        'retrieval_f1': round(f1, 4),
        'expected_sections': total_expected,
        'found_sections': total_found
    }


def compute_citation_precision(
    recommendations: List,
    source_chunks: Dict[str, str]
) -> Dict[str, Any]:
    """
    Compute citation precision metrics.
    
    Verifies that each citation:
    1. References an existing chunk
    2. Contains accurate quoted text from that chunk
    """
    total_citations = 0
    valid_citations = 0
    errors = []
    
    for i, rec in enumerate(recommendations):
        rec_citations = rec.citations if hasattr(rec, 'citations') else rec.get('citations', [])
        
        for j, citation in enumerate(rec_citations):
            total_citations += 1
            
            chunk_id = citation.chunk_id if hasattr(citation, 'chunk_id') else citation.get('chunk_id', '')
            quoted_text = citation.quoted_text if hasattr(citation, 'quoted_text') else citation.get('quoted_text', '')
            
            is_valid = validate_citation(chunk_id, quoted_text, source_chunks)
            
            if is_valid:
                valid_citations += 1
            else:
                errors.append({
                    'recommendation_index': i,
                    'citation_index': j,
                    'chunk_id': chunk_id,
                    'reason': 'Invalid citation'
                })
    
    precision = valid_citations / total_citations if total_citations > 0 else 0.0
    
    return {
        'citation_precision': round(precision, 4),
        'total_citations': total_citations,
        'valid_citations': valid_citations,
        'invalid_citations': total_citations - valid_citations,
        'errors': errors
    }


def run_full_evaluation():
    """Run complete evaluation and save results."""
    logger.info("=" * 60)
    logger.info("COMPLIANCE AI SYSTEM - EVALUATION RUN")
    logger.info("=" * 60)
    
    # Load test data
    baseline_reg, new_reg, policy_docs = load_test_data()
    
    # Run pipeline
    results, ingester, changes, retrieved_chunks, recommendations = run_evaluation_pipeline(
        baseline_reg, new_reg, policy_docs
    )
    
    # Get source chunks for validation
    source_chunks = ingester.get_all_chunks()
    
    # Compute metrics
    logger.info("=" * 50)
    logger.info("STEP 5: Computing Metrics")
    
    retrieval_metrics = compute_retrieval_accuracy(changes, retrieved_chunks, GROUND_TRUTH)
    citation_metrics = compute_citation_precision(recommendations, source_chunks)
    
    # Compile final results
    final_results = {
        **results,
        'evaluation_metrics': {
            'retrieval': retrieval_metrics,
            'citations': citation_metrics,
            'latency': {
                'total_seconds': results['total_latency_seconds'],
                'per_step': {
                    step: data.get('duration_seconds', 0) 
                    for step, data in results['steps'].items()
                }
            }
        },
        'summary': {
            'changes_detected': len(changes),
            'policies_retrieved': len(retrieved_chunks),
            'recommendations_generated': len(recommendations),
            'retrieval_f1': retrieval_metrics['retrieval_f1'],
            'citation_precision': citation_metrics['citation_precision'],
            'total_latency_seconds': results['total_latency_seconds']
        }
    }
    
    # Print results table
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS SUMMARY")
    print("=" * 60)
    print(f"{'Metric':<35} {'Value':>20}")
    print("-" * 60)
    print(f"{'Changes Detected':<35} {len(changes):>20}")
    print(f"{'Policies Retrieved':<35} {len(retrieved_chunks):>20}")
    print(f"{'Recommendations Generated':<35} {len(recommendations):>20}")
    print("-" * 60)
    print(f"{'Retrieval Recall':<35} {retrieval_metrics['retrieval_recall']:>20.2%}")
    print(f"{'Retrieval Precision':<35} {retrieval_metrics['retrieval_precision']:>20.2%}")
    print(f"{'Retrieval F1 Score':<35} {retrieval_metrics['retrieval_f1']:>20.2%}")
    print("-" * 60)
    print(f"{'Citation Precision':<35} {citation_metrics['citation_precision']:>20.2%}")
    print(f"{'Total Citations':<35} {citation_metrics['total_citations']:>20}")
    print("-" * 60)
    print(f"{'Total Latency (seconds)':<35} {results['total_latency_seconds']:>20.2f}")
    print(f"{'Latency per Regulation':<35} {results['total_latency_seconds']:>20.2f}s")
    print("=" * 60 + "\n")
    
    # Save results to file
    output_path = Path(__file__).parent / "eval_results.json"
    with open(output_path, 'w') as f:
        json.dump(final_results, f, indent=2, default=str)
    
    logger.info(f"Results saved to {output_path}")
    
    return final_results


if __name__ == "__main__":
    run_full_evaluation()
