"""
Document ingestion module for Compliance AI system.
Handles parsing, chunking, embedding, and vector storage of documents.
"""
import hashlib
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple

from llama_index.core import Document, Settings
from llama_index.core.node_parser import SemanticSplitterNodeParser, SentenceSplitter
from llama_index.core.storage import StorageContext
from llama_index.core.vector_stores.types import VectorStoreQuery
from llama_index.vector_stores.chroma import ChromaVectorStore
import chromadb

from config import CHUNK_SIZE, CHUNK_OVERLAP, VECTOR_DB_PATH, EMBEDDING_MODEL


class DocumentIngester:
    """
    Handles document ingestion with semantic-aware chunking and vector storage.
    Uses ChromaDB as the in-memory vector store for MVP.
    """
    
    def __init__(self, collection_name: str = "compliance_docs"):
        """Initialize the ingester with ChromaDB backend."""
        # Initialize ChromaDB client (in-memory for MVP)
        self.chroma_client = chromadb.PersistentClient(path=str(VECTOR_DB_PATH))
        
        # Get or create collection
        self.collection = self.chroma_client.get_or_create_collection(
            name=collection_name,
            metadata={"description": "Compliance document chunks"}
        )
        
        # Configure embedding model (using OpenAI-compatible for MVP)
        # In production, this would use actual embedding API
        Settings.embed_model = None  # We'll compute embeddings manually for MVP
        
        # Chunk size and overlap from config
        self.chunk_size = CHUNK_SIZE
        self.chunk_overlap = CHUNK_OVERLAP
    
    def _compute_chunk_id(self, text: str, doc_id: str, index: int) -> str:
        """Generate a unique chunk ID based on content and position."""
        content_hash = hashlib.md5(text[:100].encode()).hexdigest()[:8]
        return f"{doc_id}_chunk{index}_{content_hash}"
    
    def _chunk_document(
        self, 
        text: str, 
        doc_id: str, 
        doc_type: str = "regulation"
    ) -> List[Dict[str, Any]]:
        """
        Split document into semantically meaningful chunks with metadata.
        
        Strategy:
        1. First split by major sections (SECTION headers)
        2. Then split each section into smaller chunks respecting paragraph boundaries
        3. Add metadata including section info and chunk position
        """
        chunks = []
        
        # Split by section headers first
        section_pattern = r'(SECTION\s+\d+[^:]*:|POLICY\s+SECTION\s+CP-\d+[^:]*:)'
        import re
        
        # Find all section breaks
        sections = re.split(section_pattern, text, flags=re.IGNORECASE)
        
        # Reconstruct sections with their headers
        current_section = ""
        for i, part in enumerate(sections):
            if re.match(section_pattern, part, re.IGNORECASE):
                current_section = part + "\n"
            elif part.strip():
                current_section += part + "\n"
                
                # If we have a complete section or it's getting long, create chunks
                if len(current_section) > self.chunk_size * 2 or i == len(sections) - 1:
                    # Further split large sections
                    subsections = self._split_into_chunks(
                        current_section.strip(), 
                        doc_id, 
                        doc_type,
                        len(chunks)
                    )
                    chunks.extend(subsections)
                    current_section = ""
        
        # Handle any remaining text
        if current_section.strip():
            subsections = self._split_into_chunks(
                current_section.strip(), 
                doc_id, 
                doc_type,
                len(chunks)
            )
            chunks.extend(subsections)
        
        return chunks
    
    def _split_into_chunks(
        self, 
        text: str, 
        doc_id: str, 
        doc_type: str,
        start_index: int
    ) -> List[Dict[str, Any]]:
        """Split text into overlapping chunks with metadata."""
        chunks = []
        
        # Simple sentence-based splitting with overlap
        sentences = text.replace('\n', ' ').split('. ')
        current_chunk = []
        current_length = 0
        
        for sentence in sentences:
            sentence = sentence.strip() + '. '
            sentence_length = len(sentence)
            
            if current_length + sentence_length <= self.chunk_size:
                current_chunk.append(sentence)
                current_length += sentence_length
            else:
                # Save current chunk if not empty
                if current_chunk:
                    chunk_text = ''.join(current_chunk).strip()
                    chunk_id = self._compute_chunk_id(chunk_text, doc_id, start_index + len(chunks))
                    
                    # Extract section info from text
                    section_id = self._extract_section_info(chunk_text)
                    
                    chunks.append({
                        'id': chunk_id,
                        'text': chunk_text,
                        'metadata': {
                            'doc_id': doc_id,
                            'doc_type': doc_type,
                            'section_id': section_id,
                            'chunk_index': start_index + len(chunks),
                            'char_length': len(chunk_text)
                        }
                    })
                
                # Start new chunk with overlap
                overlap_sentences = current_chunk[-max(1, len(current_chunk) // 3):]
                current_chunk = overlap_sentences + [sentence]
                current_length = sum(len(s) for s in current_chunk)
        
        # Don't forget the last chunk
        if current_chunk:
            chunk_text = ''.join(current_chunk).strip()
            chunk_id = self._compute_chunk_id(chunk_text, doc_id, start_index + len(chunks))
            section_id = self._extract_section_info(chunk_text)
            
            chunks.append({
                'id': chunk_id,
                'text': chunk_text,
                'metadata': {
                    'doc_id': doc_id,
                    'doc_type': doc_type,
                    'section_id': section_id,
                    'chunk_index': start_index + len(chunks),
                    'char_length': len(chunk_text)
                }
            })
        
        return chunks
    
    def _extract_section_info(self, text: str) -> str:
        """Extract section identifier from chunk text."""
        import re
        
        # Look for section patterns
        patterns = [
            r'SECTION\s+(\d+(?:\.\d+)?)',
            r'(\d+\.\d+)',
            r'(CP-\d+(?:\.\d+)?)',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(1)
        
        return "unknown"
    
    def ingest_text(
        self, 
        text: str, 
        doc_id: str, 
        doc_type: str = "regulation"
    ) -> List[Dict[str, Any]]:
        """
        Ingest text document into vector store.
        
        Args:
            text: Document text content
            doc_id: Unique identifier for the document
            doc_type: Type of document ('regulation' or 'policy')
            
        Returns:
            List of chunk dictionaries
        """
        # Chunk the document
        chunks = self._chunk_document(text, doc_id, doc_type)
        
        # Store in ChromaDB
        if chunks:
            ids = [chunk['id'] for chunk in chunks]
            texts = [chunk['text'] for chunk in chunks]
            metadatas = [chunk['metadata'] for chunk in chunks]
            
            # For MVP, we'll store without embeddings (or use simple hash-based)
            # In production, this would call embedding API
            self.collection.add(
                ids=ids,
                documents=texts,
                metadatas=metadatas
            )
        
        return chunks
    
    def query_similar(
        self, 
        query_text: str, 
        top_k: int = 5,
        filter_metadata: Optional[Dict] = None
    ) -> List[Tuple[str, str, float, Dict]]:
        """
        Query for similar chunks using semantic search.
        
        Args:
            query_text: Search query text
            top_k: Number of results to return
            filter_metadata: Optional metadata filters
            
        Returns:
            List of tuples: (chunk_id, text, score, metadata)
        """
        # For MVP, use keyword-based search with simple scoring
        # In production, this would use dense embeddings
        
        query_results = self.collection.query(
            query_texts=[query_text],
            n_results=top_k,
            where=filter_metadata
        )
        
        results = []
        if query_results and query_results['ids']:
            for i, chunk_id in enumerate(query_results['ids'][0]):
                text = query_results['documents'][0][i] if query_results['documents'] else ""
                metadata = query_results['metadatas'][0][i] if query_results['metadatas'] else {}
                # Distance score (lower is better in ChromaDB, convert to similarity)
                distance = query_results['distances'][0][i] if query_results.get('distances') else 0
                score = 1.0 / (1.0 + distance) if distance else 1.0
                
                results.append((chunk_id, text, score, metadata))
        
        return results
    
    def get_chunk_by_id(self, chunk_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve a specific chunk by its ID."""
        result = self.collection.get(ids=[chunk_id])
        
        if result and result['ids'] and len(result['ids']) > 0:
            return {
                'id': result['ids'][0],
                'text': result['documents'][0] if result['documents'] else "",
                'metadata': result['metadatas'][0] if result['metadatas'] else {}
            }
        return None
    
    def get_all_chunks(self) -> Dict[str, str]:
        """Get all chunks as a dictionary mapping IDs to text."""
        result = self.collection.get()
        
        chunks_dict = {}
        if result and result['ids']:
            for i, chunk_id in enumerate(result['ids']):
                text = result['documents'][i] if result['documents'] else ""
                chunks_dict[chunk_id] = text
        
        return chunks_dict
    
    def clear_collection(self):
        """Clear all data from the collection (for testing)."""
        self.chroma_client.delete_collection(self.collection.name)
        self.collection = self.chroma_client.create_collection(
            name=self.collection.name,
            metadata={"description": "Compliance document chunks"}
        )


def load_and_ingest_documents(
    regulation_text: str,
    policy_text: str,
    reg_doc_id: str = "regulation_v1",
    policy_doc_id: str = "company_policies_v1"
) -> Tuple[DocumentIngester, List[Dict], List[Dict]]:
    """
    Load and ingest both regulation and policy documents.
    
    Returns:
        Tuple of (ingester, regulation_chunks, policy_chunks)
    """
    ingester = DocumentIngester()
    
    # Clear existing data for fresh start
    ingester.clear_collection()
    
    # Ingest documents
    reg_chunks = ingester.ingest_text(regulation_text, reg_doc_id, "regulation")
    policy_chunks = ingester.ingest_text(policy_text, policy_doc_id, "policy")
    
    return ingester, reg_chunks, policy_chunks
