"""
Streamlit UI for Compliance AI System.
Provides a simple interface for uploading documents and running analysis.
"""
import streamlit as st
import sys
import os
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from config import MOCK_MODE, validate_config
from src.ingestion import DocumentIngester
from src.orchestrator import ComplianceOrchestrator
from src.utils import format_output_for_display


def initialize_session_state():
    """Initialize Streamlit session state variables."""
    if 'analysis_result' not in st.session_state:
        st.session_state.analysis_result = None
    if 'logs' not in st.session_state:
        st.session_state.logs = []
    if 'is_running' not in st.session_state:
        st.session_state.is_running = False


def add_log(message: str):
    """Add a message to the logs."""
    st.session_state.logs.append(message)


def run_analysis(new_reg_text: str, baseline_reg_text: str, policy_text: str):
    """Run the compliance analysis pipeline."""
    try:
        add_log("Initializing document ingestion...")
        
        # Create ingester and process documents
        ingester = DocumentIngester()
        ingester.clear_collection()
        
        add_log("Chunking regulation documents...")
        reg_chunks = ingester.ingest_text(new_reg_text, "new_regulation", "regulation")
        baseline_chunks = ingester.ingest_text(baseline_reg_text, "baseline_regulation", "regulation")
        
        add_log(f"Created {len(reg_chunks)} regulation chunks")
        
        add_log("Chunking policy documents...")
        policy_chunks = ingester.ingest_text(policy_text, "company_policies", "policy")
        add_log(f"Created {len(policy_chunks)} policy chunks")
        
        add_log("Starting multi-agent analysis workflow...")
        
        # Run orchestrator
        orchestrator = ComplianceOrchestrator()
        result = orchestrator.run(
            new_regulation=new_reg_text,
            baseline_regulation=baseline_reg_text,
            policy_chunks=policy_chunks
        )
        
        return result
        
    except Exception as e:
        add_log(f"ERROR: {str(e)}")
        st.error(f"Analysis failed: {str(e)}")
        return None


def display_results(result: dict):
    """Display analysis results in the UI."""
    if not result:
        return
    
    st.header("📊 Analysis Results")
    
    # Summary metrics
    summary = result.get('summary', {})
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            "Changes Detected", 
            summary.get('changes_detected', 0),
            delta=None
        )
    
    with col2:
        st.metric(
            "Policies Retrieved", 
            summary.get('policies_retrieved', 0),
            delta=None
        )
    
    with col3:
        st.metric(
            "Recommendations", 
            summary.get('recommendations_generated', 0),
            delta=None
        )
    
    # Status
    status = summary.get('overall_status', 'unknown')
    if status == 'success':
        st.success("✅ Analysis completed successfully")
    elif status == 'partial_failure':
        st.warning("⚠️ Analysis completed with some errors")
    else:
        st.info("ℹ️ Analysis status unknown")
    
    # Changes section
    changes = result.get('changes', [])
    if changes:
        with st.expander(f"📝 Detected Changes ({len(changes)})", expanded=True):
            for i, change in enumerate(changes):
                st.subheader(f"Change {i+1}: {change.get('section_id', 'Unknown')}")
                st.write(f"**Summary:** {change.get('summary', 'N/A')}")
                st.write(f"**Type:** {change.get('change_type', 'N/A')}")
                st.write(f"**Old Value:** {change.get('old_value', 'N/A')}")
                st.write(f"**New Value:** {change.get('new_value', 'N/A')}")
                st.write(f"**Reason:** {change.get('reason', 'N/A')}")
                st.divider()
    
    # Recommendations section
    recommendations = result.get('recommendations', [])
    if recommendations:
        with st.expander(f"🎯 Recommendations ({len(recommendations)})", expanded=True):
            for i, rec in enumerate(recommendations):
                risk_level = rec.get('risk_level', 'medium')
                risk_emoji = {"high": "🔴", "medium": "🟡", "low": "🟢"}.get(risk_level, "⚪")
                
                st.subheader(f"{risk_emoji} Recommendation {i+1}")
                st.write(f"**Priority:** {rec.get('implementation_priority', 'N/A')} (1=highest)")
                st.write(f"**Risk Level:** {risk_level.upper()}")
                st.write(f"**Affected Change:** {rec.get('affected_change', 'N/A')}")
                st.info(f"**Action:** {rec.get('recommendation', 'N/A')}")
                
                # Citations
                citations = rec.get('citations', [])
                if citations:
                    st.write("**Citations:**")
                    for j, citation in enumerate(citations):
                        with st.container(border=True):
                            st.caption(f"Citation {j+1}: `{citation.get('chunk_id', 'Unknown')}`")
                            st.write(f"*Section:* {citation.get('policy_section', 'N/A')}")
                            st.quote(f"\"{citation.get('quoted_text', 'N/A')}\"")
                
                st.divider()
    
    # Retrieved policies
    retrieved = result.get('retrieved_policies', [])
    if retrieved:
        with st.expander(f"📚 Retrieved Policy Chunks ({len(retrieved)})"):
            for i, chunk in enumerate(retrieved):
                st.text_area(
                    f"Chunk {i+1}: {chunk.get('chunk_id', 'Unknown')}",
                    chunk.get('text', ''),
                    height=150,
                    key=f"policy_chunk_{i}"
                )
                st.write(f"Relevance Score: {chunk.get('relevance_score', 0):.4f}")
                st.divider()
    
    # Logs
    logs = result.get('logs', [])
    if logs:
        with st.expander("📋 Processing Logs"):
            for log in logs:
                st.text(log)
    
    # Errors
    errors = result.get('errors', [])
    if errors:
        with st.expander("⚠️ Errors"):
            for error in errors:
                st.error(error)
    
    # Metrics
    metrics = result.get('metrics', {})
    if metrics:
        with st.expander("📈 Metrics"):
            st.json(metrics)


def main():
    """Main Streamlit application."""
    st.set_page_config(
        page_title="Compliance AI - Regulatory Analysis",
        page_icon="⚖️",
        layout="wide"
    )
    
    st.title("⚖️ Compliance AI - Regulatory Document Analysis")
    st.markdown("""
    Multi-agent AI system for automating regulatory compliance document analysis.
    Upload a new regulation and baseline to identify changes and generate policy update recommendations.
    """)
    
    # Show configuration warnings
    warnings = validate_config()
    for warning in warnings:
        st.warning(warning)
    
    if MOCK_MODE:
        st.info("🧪 Running in MOCK MODE - Using simulated responses for testing")
    
    initialize_session_state()
    
    # Sidebar for file uploads
    with st.sidebar:
        st.header("📁 Document Upload")
        
        # File uploaders
        new_reg_file = st.file_uploader(
            "New Regulation",
            type=['txt', 'pdf'],
            help="Upload the new/updated regulation document"
        )
        
        baseline_reg_file = st.file_uploader(
            "Baseline Regulation",
            type=['txt', 'pdf'],
            help="Upload the previous version of the regulation"
        )
        
        policy_file = st.file_uploader(
            "Company Policies",
            type=['txt', 'pdf'],
            help="Upload internal policy documents"
        )
        
        st.divider()
        
        # Use test data button
        if st.button("🧪 Load Test Data", use_container_width=True):
            # Load test data from files
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
            
            st.session_state.new_reg_text = NEW_REGULATION
            st.session_state.baseline_reg_text = BASELINE_REGULATION
            st.session_state.policy_text = COMPANY_POLICIES
            
            st.success("Test data loaded!")
        
        st.divider()
        
        # Quick instructions
        with st.expander("📖 Instructions"):
            st.markdown("""
            1. Upload three documents:
               - **New Regulation**: The updated regulatory text
               - **Baseline Regulation**: The previous version
               - **Company Policies**: Your internal policies
            
            2. Or click "Load Test Data" to use sample documents
            
            3. Click "Run Analysis" to start the multi-agent workflow
            
            4. Review results:
               - Detected changes
               - Retrieved policy sections
               - Actionable recommendations with citations
            """)
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Text areas for direct input (alternative to file upload)
        if 'new_reg_text' not in st.session_state:
            st.session_state.new_reg_text = ""
        if 'baseline_reg_text' not in st.session_state:
            st.session_state.baseline_reg_text = ""
        if 'policy_text' not in st.session_state:
            st.session_state.policy_text = ""
        
        new_reg_text = st.text_area(
            "New Regulation Text",
            value=st.session_state.new_reg_text,
            height=200,
            placeholder="Paste new regulation text here or upload a file..."
        )
        
        baseline_reg_text = st.text_area(
            "Baseline Regulation Text",
            value=st.session_state.baseline_reg_text,
            height=200,
            placeholder="Paste baseline regulation text here or upload a file..."
        )
        
        policy_text = st.text_area(
            "Company Policy Text",
            value=st.session_state.policy_text,
            height=200,
            placeholder="Paste company policy text here or upload a file..."
        )
    
    with col2:
        # File reading logic
        if new_reg_file:
            new_reg_text = new_reg_file.read().decode('utf-8')
            st.session_state.new_reg_text = new_reg_text
            st.success(f"Loaded: {new_reg_file.name}")
        
        if baseline_reg_file:
            baseline_reg_text = baseline_reg_file.read().decode('utf-8')
            st.session_state.baseline_reg_text = baseline_reg_text
            st.success(f"Loaded: {baseline_reg_file.name}")
        
        if policy_file:
            policy_text = policy_file.read().decode('utf-8')
            st.session_state.policy_text = policy_text
            st.success(f"Loaded: {policy_file.name}")
    
    # Run button
    st.divider()
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        run_button = st.button(
            "🚀 Run Compliance Analysis",
            type="primary",
            use_container_width=True,
            disabled=st.session_state.is_running
        )
    
    if run_button:
        # Validate inputs
        if not new_reg_text or not baseline_reg_text or not policy_text:
            st.error("Please provide all three documents (upload files or paste text)")
        else:
            st.session_state.is_running = True
            st.session_state.logs = []
            
            with st.spinner("Running analysis... This may take a few minutes."):
                # Create progress container
                progress_container = st.empty()
                
                # Run analysis
                result = run_analysis(new_reg_text, baseline_reg_text, policy_text)
                
                st.session_state.analysis_result = result
                st.session_state.is_running = False
                
                if result:
                    st.success("Analysis completed!")
                    display_results(result)
    
    # Display previous results if available
    if st.session_state.analysis_result and not run_button:
        display_results(st.session_state.analysis_result)
    
    # Footer
    st.divider()
    st.caption("Compliance AI MVP v1.0 | Built with LangGraph, LlamaIndex, and Streamlit")


if __name__ == "__main__":
    main()
