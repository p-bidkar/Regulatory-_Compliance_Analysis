"""
Agent 2: RAG Policy Retrieval Module
Retrieves relevant internal policy sections affected by detected regulatory changes.
Uses hybrid search (BM25 + dense embeddings) with cross-encoder re-ranking.
"""
import re
from typing import Dict, List, Any, Optional, Tuple
from pydantic import BaseModel, Field

from config import TOP_K_RETRIEVAL
from src.utils import logger


class RetrievedPolicyChunk(BaseModel):
    """Schema for a retrieved policy chunk."""
    chunk_id: str = Field(..., description="Unique identifier of the chunk")
    text: str = Field(..., description="The actual policy text")
    relevance_score: float = Field(..., description="Combined relevance score (0-1)")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Chunk metadata")
    policy_section: str = Field("", description="Policy section identifier")


class RetrievalResult(BaseModel):
    """Schema for Agent 2 output."""
    retrieved_chunks: List[RetrievedPolicyChunk] = Field(
        default_factory=list,
        description="Top-k most relevant policy chunks"
    )
    query_summary: str = Field("", description="Summary of what was searched")
    retrieval_method: str = Field("hybrid", description="Search method used")


class HybridRetriever:
    """
    Hybrid retriever combining BM25 keyword search with semantic similarity.
    Includes cross-encoder style re-ranking for improved relevance.
    """
    
    def __init__(self, top_k: int = TOP_K_RETRIEVAL):
        self.top_k = top_k
        self.bm25_index = None
        self.policy_chunks: List[Dict] = []
        self.chunk_texts: List[str] = []
    
    def build_bm25_index(self, documents: List[str]):
        """Build BM25 index from document texts."""
        try:
            from rank_bm25 import BM25Okapi
            
            # Tokenize documents (simple whitespace + punctuation splitting)
            tokenized_docs = [self._tokenize(doc) for doc in documents]
            self.bm25_index = BM25Okapi(tokenized_docs)
            logger.info(f"Built BM25 index with {len(documents)} documents")
        except ImportError:
            logger.warning("rank-bm25 not installed, falling back to keyword matching")
            self.bm25_index = None
    
    def _tokenize(self, text: str) -> List[str]:
        """Simple tokenizer for BM25."""
        # Lowercase and split on non-alphanumeric
        return re.findall(r'\b[a-z]+\b', text.lower())
    
    def set_policy_chunks(self, chunks: List[Dict]):
        """Set the policy chunks to search over."""
        self.policy_chunks = chunks
        self.chunk_texts = [chunk['text'] for chunk in chunks]
        self.build_bm25_index(self.chunk_texts)
    
    def _bm25_search(self, query: str, top_n: int = 10) -> List[Tuple[int, float]]:
        """Search using BM25 algorithm."""
        if not self.bm25_index or not self.chunk_texts:
            return []
        
        query_tokens = self._tokenize(query)
        scores = self.bm25_index.get_scores(query_tokens)
        
        # Get top-n indices with scores
        indexed_scores = list(enumerate(scores))
        indexed_scores.sort(key=lambda x: x[1], reverse=True)
        
        return indexed_scores[:top_n]
    
    def _keyword_match_score(self, query: str, text: str) -> float:
        """Compute simple keyword overlap score."""
        query_words = set(self._tokenize(query))
        text_words = set(self._tokenize(text))
        
        if not query_words:
            return 0.0
        
        overlap = query_words & text_words
        return len(overlap) / len(query_words)
    
    def _semantic_similarity(self, query: str, text: str) -> float:
        """
        Compute semantic similarity between query and text.
        For MVP, uses TF-IDF cosine similarity as proxy.
        In production, would use sentence transformers or API embeddings.
        """
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
            from sklearn.metrics.pairwise import cosine_similarity
            
            vectorizer = TfidfVectorizer(stop_words='english')
            tfidf_matrix = vectorizer.fit_transform([query, text])
            similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
            return float(similarity)
        except Exception as e:
            logger.warning(f"Semantic similarity failed: {e}")
            return self._keyword_match_score(query, text)
    
    def _rerank_with_cross_encoder(
        self, 
        query: str, 
        candidates: List[Tuple[int, float, str]]
    ) -> List[Tuple[int, float]]:
        """
        Re-rank candidates using a more sophisticated scoring approach.
        For MVP, uses weighted combination of BM25 and semantic similarity.
        """
        reranked = []
        
        for idx, bm25_score, text in candidates:
            # Compute semantic similarity
            semantic_score = self._semantic_similarity(query, text)
            
            # Normalize BM25 score to 0-1 range (approximate)
            bm25_normalized = min(1.0, bm25_score / 10.0)
            
            # Weighted combination (favor semantic for regulatory matching)
            combined_score = 0.4 * bm25_normalized + 0.6 * semantic_score
            
            reranked.append((idx, combined_score))
        
        # Sort by combined score
        reranked.sort(key=lambda x: x[1], reverse=True)
        
        return reranked
    
    def retrieve(
        self, 
        query: str, 
        filter_metadata: Optional[Dict] = None,
        top_k: Optional[int] = None
    ) -> List[RetrievedPolicyChunk]:
        """
        Retrieve top-k policy chunks relevant to the query.
        
        Args:
            query: Search query (typically a change summary)
            filter_metadata: Optional metadata filters (e.g., {'doc_type': 'policy'})
            top_k: Number of results (overrides instance default if provided)
            
        Returns:
            List of RetrievedPolicyChunk objects
        """
        k = top_k if top_k else self.top_k
        
        if not self.policy_chunks:
            logger.warning("No policy chunks available for retrieval")
            return []
        
        # Apply metadata filter if specified
        filtered_chunks = self.policy_chunks
        filtered_texts = self.chunk_texts
        
        if filter_metadata:
            filtered = []
            filtered_txts = []
            for i, chunk in enumerate(self.policy_chunks):
                match = True
                for key, value in filter_metadata.items():
                    if chunk.get('metadata', {}).get(key) != value:
                        match = False
                        break
                if match:
                    filtered.append(chunk)
                    filtered_txts.append(self.chunk_texts[i])
            filtered_chunks = filtered
            filtered_texts = filtered_txts
        
        if not filtered_chunks:
            return []
        
        # Step 1: BM25 search
        bm25_results = self._bm25_search(query, top_n=min(20, len(filtered_texts)))
        
        # Step 2: Prepare candidates for re-ranking
        candidates = []
        for idx, score in bm25_results:
            text = filtered_texts[idx]
            candidates.append((idx, score, text))
        
        # Also add exact keyword matches that BM25 might have missed
        for i, text in enumerate(filtered_texts):
            if self._keyword_match_score(query, text) > 0.3:
                if not any(c[0] == i for c in candidates):
                    candidates.append((i, 0.5, text))
        
        # Step 3: Re-rank candidates
        reranked = self._rerank_with_cross_encoder(query, candidates)
        
        # Step 4: Build result objects
        results = []
        for idx, score in reranked[:k]:
            chunk = filtered_chunks[idx]
            
            # Extract policy section from metadata or text
            policy_section = chunk.get('metadata', {}).get('section_id', '')
            if not policy_section:
                # Try to extract from text
                match = re.search(r'(CP-\d+(?:\.\d+)?)', chunk['text'])
                if match:
                    policy_section = match.group(1)
            
            results.append(RetrievedPolicyChunk(
                chunk_id=chunk['id'],
                text=chunk['text'],
                relevance_score=round(score, 4),
                metadata=chunk.get('metadata', {}),
                policy_section=policy_section
            ))
        
        return results
    
    def batch_retrieve(
        self, 
        queries: List[str], 
        deduplicate: bool = True
    ) -> List[RetrievedPolicyChunk]:
        """
        Retrieve for multiple queries, optionally deduplicating results.
        
        Args:
            queries: List of search queries
            deduplicate: If True, remove duplicate chunks across queries
            
        Returns:
            Combined list of retrieved chunks
        """
        all_results = []
        seen_ids = set()
        
        for query in queries:
            results = self.retrieve(query)
            
            if deduplicate:
                for result in results:
                    if result.chunk_id not in seen_ids:
                        all_results.append(result)
                        seen_ids.add(result.chunk_id)
            else:
                all_results.extend(results)
        
        # Sort by relevance score
        all_results.sort(key=lambda x: x.relevance_score, reverse=True)
        
        return all_results[:self.top_k * 2]  # Return more for batch queries


class PolicyRetrieverAgent:
    """
    Agent 2: Retrieves relevant policy sections for detected changes.
    """
    
    def __init__(self, top_k: int = TOP_K_RETRIEVAL):
        self.retriever = HybridRetriever(top_k=top_k)
        self.initialized = False
    
    def initialize_with_policies(self, policy_chunks: List[Dict]):
        """Initialize the retriever with policy document chunks."""
        self.retriever.set_policy_chunks(policy_chunks)
        self.initialized = True
        logger.info(f"Policy retriever initialized with {len(policy_chunks)} chunks")
    
    def retrieve_for_changes(
        self, 
        change_summaries: List[str],
        additional_context: Optional[List[str]] = None
    ) -> RetrievalResult:
        """
        Retrieve relevant policies for a list of detected changes.
        
        Args:
            change_summaries: List of change summaries from Agent 1
            additional_context: Optional additional context for each change
            
        Returns:
            RetrievalResult with matched policy chunks
        """
        if not self.initialized:
            logger.error("Policy retriever not initialized. Call initialize_with_policies first.")
            return RetrievalResult(
                retrieved_chunks=[],
                query_summary="Error: retriever not initialized",
                retrieval_method="none"
            )
        
        # Build enhanced queries from change summaries
        queries = []
        for summary in change_summaries:
            # Enhance query with compliance-related terms
            enhanced_query = f"{summary} compliance policy requirement regulation"
            queries.append(enhanced_query)
        
        # Add additional context as separate queries if provided
        if additional_context:
            queries.extend(additional_context)
        
        # Perform batch retrieval
        retrieved_chunks = self.retriever.batch_retrieve(queries, deduplicate=True)
        
        # Generate query summary
        query_summary = f"Searched for policies related to {len(queries)} change descriptions"
        
        return RetrievalResult(
            retrieved_chunks=retrieved_chunks,
            query_summary=query_summary,
            retrieval_method="hybrid_bm25_semantic"
        )
    
    def retrieve_single(self, query: str) -> RetrievalResult:
        """Retrieve for a single query."""
        if not self.initialized:
            return RetrievalResult(retrieved_chunks=[], query_summary="Error: not initialized")
        
        chunks = self.retriever.retrieve(query)
        
        return RetrievalResult(
            retrieved_chunks=chunks,
            query_summary=f"Single query: {query[:100]}...",
            retrieval_method="hybrid"
        )


def create_policy_retriever(policy_chunks: List[Dict], top_k: int = TOP_K_RETRIEVAL) -> PolicyRetrieverAgent:
    """Factory function to create and initialize a policy retriever."""
    agent = PolicyRetrieverAgent(top_k=top_k)
    agent.initialize_with_policies(policy_chunks)
    return agent
