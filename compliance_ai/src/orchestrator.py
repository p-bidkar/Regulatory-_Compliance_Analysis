"""
Orchestrator Module - LangGraph-based multi-agent coordination.
Coordinates the 3 agents through a stateful workflow with error handling.
"""
from typing import TypedDict, List, Dict, Any, Optional, Annotated
from pathlib import Path
from typing_extensions import NotRequired

from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages

from config import MOCK_MODE, LLM_PROVIDER
from src.utils import logger, LatencyTimer, format_output_for_display
from src.agent1_change_detection import ChangeDetector, ChangeDetectionResult
from src.agent2_rag_retrieval import PolicyRetrieverAgent, RetrievalResult, create_policy_retriever
from src.agent3_recommendation import RecommendationGenerator, RecommendationResult


# Define the state schema for our workflow
class ComplianceState(TypedDict):
    """State container for the compliance analysis workflow."""
    
    # Inputs
    new_reg: str
    baseline_reg: str
    policy_docs: List[Dict]  # Policy chunks
    
    # Intermediate results
    changes: List[Dict]  # Detected changes from Agent 1
    retrieved_policies: List[Dict]  # Retrieved policies from Agent 2
    recommendations: List[Dict]  # Recommendations from Agent 3
    
    # Metadata and logs
    logs: List[str]
    errors: List[str]
    metrics: Dict[str, Any]
    
    # Final output
    final_output: Dict[str, Any]


class ComplianceOrchestrator:
    """
    Orchestrates the multi-agent compliance analysis workflow using LangGraph.
    
    Workflow:
    1. detect_changes → Agent 1 identifies substantive regulatory changes
    2. retrieve_policies → Agent 2 finds relevant internal policies
    3. generate_recommendations → Agent 3 creates actionable recommendations
    4. final_output → Compile and validate results
    """
    
    def __init__(self):
        self.change_detector = ChangeDetector()
        self.policy_retriever: Optional[PolicyRetrieverAgent] = None
        self.recommendation_generator = RecommendationGenerator()
        
        # Build the LangGraph workflow
        self.graph = self._build_graph()
    
    def _log(self, state: ComplianceState, message: str):
        """Add a log message to the state."""
        if 'logs' not in state:
            state['logs'] = []
        state['logs'].append(message)
        logger.info(message)
    
    def _error(self, state: ComplianceState, message: str):
        """Add an error message to the state."""
        if 'errors' not in state:
            state['errors'] = []
        state['errors'].append(message)
        logger.error(message)
    
    def detect_changes_node(self, state: ComplianceState) -> ComplianceState:
        """Node: Agent 1 - Detect regulatory changes."""
        self._log(state, "Starting change detection (Agent 1)...")
        
        with LatencyTimer("Change Detection"):
            try:
                result: ChangeDetectionResult = self.change_detector.detect_changes(
                    baseline_regulation=state['baseline_reg'],
                    new_regulation=state['new_reg']
                )
                
                # Convert Pydantic model to dict for state
                state['changes'] = [c.dict() for c in result.changed_sections]
                state['metrics']['change_detection'] = {
                    'num_changes': len(result.changed_sections),
                    'confidence': result.confidence_score,
                    'summary': result.analysis_summary
                }
                
                self._log(state, f"Detected {len(result.changed_sections)} substantive changes")
                
            except Exception as e:
                self._error(state, f"Change detection failed: {str(e)}")
                state['changes'] = []
        
        return state
    
    def should_skip_retrieval(self, state: ComplianceState) -> str:
        """Conditional edge: Skip to end if no changes detected."""
        if not state.get('changes') or len(state['changes']) == 0:
            self._log(state, "No changes detected, skipping retrieval and recommendation steps")
            return "final_output"
        return "retrieve_policies"
    
    def retrieve_policies_node(self, state: ComplianceState) -> ComplianceState:
        """Node: Agent 2 - Retrieve relevant policies."""
        self._log(state, "Starting policy retrieval (Agent 2)...")
        
        with LatencyTimer("Policy Retrieval"):
            try:
                # Initialize retriever if not already done
                if self.policy_retriever is None:
                    self.policy_retriever = create_policy_retriever(state['policy_docs'])
                
                # Extract change summaries for queries
                change_summaries = [c.get('summary', '') for c in state['changes']]
                
                result: RetrievalResult = self.policy_retriever.retrieve_for_changes(
                    change_summaries=change_summaries
                )
                
                # Store retrieved policies
                state['retrieved_policies'] = [r.dict() for r in result.retrieved_chunks]
                state['metrics']['retrieval'] = {
                    'num_retrieved': len(result.retrieved_chunks),
                    'method': result.retrieval_method,
                    'query_summary': result.query_summary
                }
                
                self._log(state, f"Retrieved {len(result.retrieved_chunks)} relevant policy chunks")
                
            except Exception as e:
                self._error(state, f"Policy retrieval failed: {str(e)}")
                state['retrieved_policies'] = []
        
        return state
    
    def generate_recommendations_node(self, state: ComplianceState) -> ComplianceState:
        """Node: Agent 3 - Generate recommendations."""
        self._log(state, "Starting recommendation generation (Agent 3)...")
        
        with LatencyTimer("Recommendation Generation"):
            try:
                # Convert retrieved policies back to Pydantic models
                from src.agent2_rag_retrieval import RetrievedPolicyChunk
                retrieved_chunks = [
                    RetrievedPolicyChunk(**r) for r in state['retrieved_policies']
                ]
                
                result: RecommendationResult = self.recommendation_generator.generate_recommendations(
                    changes=state['changes'],
                    retrieved_chunks=retrieved_chunks
                )
                
                # Store recommendations
                state['recommendations'] = [r.dict() for r in result.recommendations]
                state['metrics']['recommendations'] = {
                    'num_recommendations': len(result.recommendations),
                    'total_citations': result.total_citations,
                    'summary': result.summary
                }
                
                self._log(state, f"Generated {len(result.recommendations)} recommendations")
                
                # Validate citations
                source_chunks = {c['id']: c['text'] for c in state['policy_docs']}
                validation = self.recommendation_generator.validate_recommendations(
                    result.recommendations,
                    source_chunks
                )
                state['metrics']['citation_validation'] = validation
                
            except Exception as e:
                self._error(state, f"Recommendation generation failed: {str(e)}")
                state['recommendations'] = []
        
        return state
    
    def final_output_node(self, state: ComplianceState) -> ComplianceState:
        """Node: Compile final output."""
        self._log(state, "Compiling final output...")
        
        # Build comprehensive output
        state['final_output'] = {
            'summary': {
                'changes_detected': len(state.get('changes', [])),
                'policies_retrieved': len(state.get('retrieved_policies', [])),
                'recommendations_generated': len(state.get('recommendations', [])),
                'overall_status': 'success' if not state.get('errors') else 'partial_failure'
            },
            'changes': state.get('changes', []),
            'retrieved_policies': state.get('retrieved_policies', []),
            'recommendations': state.get('recommendations', []),
            'metrics': state.get('metrics', {}),
            'logs': state.get('logs', []),
            'errors': state.get('errors', [])
        }
        
        self._log(state, "Workflow completed successfully")
        
        return state
    
    def _build_graph(self) -> StateGraph:
        """Build the LangGraph workflow."""
        
        # Create the graph
        workflow = StateGraph(ComplianceState)
        
        # Add nodes
        workflow.add_node("detect_changes", self.detect_changes_node)
        workflow.add_node("retrieve_policies", self.retrieve_policies_node)
        workflow.add_node("generate_recommendations", self.generate_recommendations_node)
        workflow.add_node("final_output", self.final_output_node)
        
        # Set entry point
        workflow.set_entry_point("detect_changes")
        
        # Add edges with conditional routing
        workflow.add_conditional_edges(
            "detect_changes",
            self.should_skip_retrieval,
            {
                "retrieve_policies": "retrieve_policies",
                "final_output": "final_output"
            }
        )
        
        workflow.add_edge("retrieve_policies", "generate_recommendations")
        workflow.add_edge("generate_recommendations", "final_output")
        
        # Compile the graph
        return workflow.compile()
    
    def run(
        self,
        new_regulation: str,
        baseline_regulation: str,
        policy_chunks: List[Dict]
    ) -> Dict[str, Any]:
        """
        Run the complete compliance analysis workflow.
        
        Args:
            new_regulation: Text of the new regulation
            baseline_regulation: Text of the baseline regulation
            policy_chunks: List of policy document chunks
            
        Returns:
            Dictionary containing the complete analysis results
        """
        logger.info("Starting compliance analysis workflow...")
        
        # Initialize state
        initial_state: ComplianceState = {
            'new_reg': new_regulation,
            'baseline_reg': baseline_regulation,
            'policy_docs': policy_chunks,
            'changes': [],
            'retrieved_policies': [],
            'recommendations': [],
            'logs': [],
            'errors': [],
            'metrics': {},
            'final_output': {}
        }
        
        # Run the workflow
        final_state = self.graph.invoke(initial_state)
        
        logger.info(f"Workflow completed. Logs: {len(final_state.get('logs', []))}")
        
        return final_state.get('final_output', {})
    
    def run_with_mock_data(self) -> Dict[str, Any]:
        """Run with mock data for testing."""
        # Load test data from text files
        data_dir = Path(__file__).parent.parent / "data"
        
        with open(data_dir / "test_regulation.txt", 'r') as f:
            content = f.read()
            # Extract BASELINE_REGULATION and NEW_REGULATION from file
            import re
            baseline_match = re.search(r'BASELINE_REGULATION\s*=\s*"""(.*?)"""', content, re.DOTALL)
            new_match = re.search(r'NEW_REGULATION\s*=\s*"""(.*?)"""', content, re.DOTALL)
            
            BASELINE_REGULATION = baseline_match.group(1).strip() if baseline_match else ""
            NEW_REGULATION = new_match.group(1).strip() if new_match else ""
        
        with open(data_dir / "test_policies.txt", 'r') as f:
            policy_content = f.read()
            policy_match = re.search(r'COMPANY_POLICIES\s*=\s*"""(.*?)"""', policy_content, re.DOTALL)
            COMPANY_POLICIES = policy_match.group(1).strip() if policy_match else ""
        
        # Create simple chunks for mock data
        policy_chunks = [
            {'id': f'mock_chunk_{i}', 'text': chunk, 'metadata': {}}
            for i, chunk in enumerate(COMPANY_POLICIES.split('\n\n')[:20])
            if len(chunk.strip()) > 50
        ]
        
        return self.run(
            new_regulation=NEW_REGULATION,
            baseline_regulation=BASELINE_REGULATION,
            policy_chunks=policy_chunks
        )


def create_orchestrator() -> ComplianceOrchestrator:
    """Factory function to create an orchestrator."""
    return ComplianceOrchestrator()


# Convenience function for running the pipeline
def run_compliance_analysis(
    new_reg: str,
    baseline_reg: str,
    policy_chunks: List[Dict]
) -> Dict[str, Any]:
    """
    Run the complete compliance analysis pipeline.
    
    This is the main entry point for the multi-agent system.
    """
    orchestrator = ComplianceOrchestrator()
    return orchestrator.run(new_reg, baseline_reg, policy_chunks)
