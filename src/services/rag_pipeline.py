"""
Retrieval-Augmented Generation (RAG) pipeline for contextual anomaly detection.

This module implements a RAG system using vector embeddings of historical
transaction patterns to provide context for anomaly detection and reduce
false positives.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
import logging
from collections import defaultdict
import pickle
import json

try:
    import chromadb
    from chromadb.config import Settings
    CHROMA_AVAILABLE = True
except ImportError:
    CHROMA_AVAILABLE = False

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except (ImportError, AttributeError):
    # Handle import errors or compatibility issues
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    SentenceTransformer = None

logger = logging.getLogger(__name__)


class TransactionEmbeddingGenerator:
    """
    Generate embeddings for transaction patterns using sentence transformers.
    
    This class creates semantic embeddings of transaction data for similarity
    search and contextual analysis.
    """
    
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        """
        Initialize the embedding generator.
        
        Args:
            model_name: Name of the sentence transformer model to use
        """
        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            logger.warning("Sentence transformers not available. Install with: pip install sentence-transformers")
            self.enabled = False
            return
        
        self.model = SentenceTransformer(model_name)
        self.enabled = True
        
        logger.info(f"Embedding generator initialized with model: {model_name}")
    
    def transaction_to_text(self, transaction: Dict) -> str:
        """
        Convert transaction data to text representation for embedding.
        
        Args:
            transaction: Transaction dictionary
            
        Returns:
            Text representation of the transaction
        """
        # Create natural language description
        parts = []
        
        # Basic info
        amount = transaction.get('amount', 0)
        txn_type = transaction.get('type', 'UNKNOWN')
        parts.append(f"Transaction type {txn_type} with amount {amount:.2f}")
        
        # Account info
        origin = transaction.get('nameOrig', 'unknown')
        dest = transaction.get('nameDest', 'unknown')
        parts.append(f"from account {origin} to account {dest}")
        
        # Balance changes
        old_bal_orig = transaction.get('oldbalanceOrg', 0)
        new_bal_orig = transaction.get('newbalanceOrig', 0)
        parts.append(f"origin balance changed from {old_bal_orig:.2f} to {new_bal_orig:.2f}")
        
        old_bal_dest = transaction.get('oldbalanceDest', 0)
        new_bal_dest = transaction.get('newbalanceDest', 0)
        parts.append(f"destination balance changed from {old_bal_dest:.2f} to {new_bal_dest:.2f}")
        
        # Fraud status if available
        if 'isFraud' in transaction:
            fraud_status = "fraudulent" if transaction['isFraud'] == 1 else "legitimate"
            parts.append(f"historically {fraud_status}")
        
        return ". ".join(parts)
    
    def generate_embedding(self, transaction: Dict) -> np.ndarray:
        """
        Generate embedding for a single transaction.
        
        Args:
            transaction: Transaction dictionary
            
        Returns:
            Embedding vector as numpy array
        """
        if not self.enabled:
            return np.zeros(384)  # Default dimension for MiniLM
        
        text = self.transaction_to_text(transaction)
        embedding = self.model.encode(text, convert_to_numpy=True)
        
        return embedding
    
    def generate_batch_embeddings(self, transactions: List[Dict]) -> np.ndarray:
        """
        Generate embeddings for multiple transactions.
        
        Args:
            transactions: List of transaction dictionaries
            
        Returns:
            Array of embedding vectors
        """
        if not self.enabled:
            return np.zeros((len(transactions), 384))
        
        texts = [self.transaction_to_text(txn) for txn in transactions]
        embeddings = self.model.encode(texts, convert_to_numpy=True, show_progress_bar=True)
        
        return embeddings


class VectorDatabase:
    """
    Vector database for storing and retrieving transaction embeddings.
    
    Uses ChromaDB for efficient similarity search over transaction patterns.
    """
    
    def __init__(self,
                 persist_directory: str = "./chroma_db",
                 collection_name: str = "transactions"):
        """
        Initialize the vector database.
        
        Args:
            persist_directory: Directory to persist the database
            collection_name: Name of the collection to use
        """
        if not CHROMA_AVAILABLE:
            logger.warning("ChromaDB not available. Install with: pip install chromadb")
            self.enabled = False
            return
        
        self.persist_directory = persist_directory
        self.collection_name = collection_name
        
        # Initialize ChromaDB client
        self.client = chromadb.Client(Settings(
            chroma_db_impl="duckdb+parquet",
            persist_directory=persist_directory
        ))
        
        # Get or create collection
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
        )
        
        self.enabled = True
        
        logger.info(f"Vector database initialized with collection: {collection_name}")
    
    def add_transactions(self,
                        transactions: List[Dict],
                        embeddings: np.ndarray,
                        metadata: Optional[List[Dict]] = None) -> None:
        """
        Add transactions to the vector database.
        
        Args:
            transactions: List of transaction dictionaries
            embeddings: Array of embedding vectors
            metadata: Optional list of metadata dictionaries
        """
        if not self.enabled:
            return
        
        # Prepare data for ChromaDB
        ids = [f"txn_{i}" for i in range(len(transactions))]
        embeddings_list = embeddings.tolist()
        
        # Prepare metadata
        if metadata is None:
            metadata = []
            for txn in transactions:
                meta = {
                    'amount': float(txn.get('amount', 0)),
                    'type': str(txn.get('type', 'UNKNOWN')),
                    'isFraud': int(txn.get('isFraud', 0))
                }
                metadata.append(meta)
        
        # Prepare documents (text representations)
        documents = [
            f"Amount: {txn.get('amount', 0)}, Type: {txn.get('type', 'UNKNOWN')}"
            for txn in transactions
        ]
        
        # Add to collection in batches
        batch_size = 1000
        for i in range(0, len(transactions), batch_size):
            end_idx = min(i + batch_size, len(transactions))
            
            self.collection.add(
                ids=ids[i:end_idx],
                embeddings=embeddings_list[i:end_idx],
                metadatas=metadata[i:end_idx],
                documents=documents[i:end_idx]
            )
        
        logger.info(f"Added {len(transactions)} transactions to vector database")
    
    def search_similar_transactions(self,
                                   query_embedding: np.ndarray,
                                   n_results: int = 10,
                                   filter_metadata: Optional[Dict] = None) -> Dict:
        """
        Search for similar transactions.
        
        Args:
            query_embedding: Query embedding vector
            n_results: Number of results to return
            filter_metadata: Optional metadata filter
            
        Returns:
            Dictionary with search results
        """
        if not self.enabled:
            return {'ids': [], 'distances': [], 'metadatas': [], 'documents': []}
        
        results = self.collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=n_results,
            where=filter_metadata
        )
        
        return results
    
    def get_fraud_context(self,
                         query_embedding: np.ndarray,
                         n_results: int = 20) -> Tuple[int, int, float]:
        """
        Get fraud context from similar historical transactions.
        
        Args:
            query_embedding: Query embedding vector
            n_results: Number of similar transactions to retrieve
            
        Returns:
            Tuple of (fraud_count, legitimate_count, fraud_rate)
        """
        if not self.enabled:
            return 0, 0, 0.0
        
        results = self.search_similar_transactions(query_embedding, n_results)
        
        if not results['metadatas'] or len(results['metadatas'][0]) == 0:
            return 0, 0, 0.0
        
        fraud_count = sum(
            meta.get('isFraud', 0) for meta in results['metadatas'][0]
        )
        legitimate_count = n_results - fraud_count
        fraud_rate = fraud_count / n_results if n_results > 0 else 0.0
        
        return fraud_count, legitimate_count, fraud_rate
    
    def persist(self) -> None:
        """Persist the database to disk."""
        if self.enabled:
            self.client.persist()
            logger.info("Vector database persisted to disk")


class RAGAnomalyDetector:
    """
    Retrieval-Augmented Generation based anomaly detector.
    
    Uses historical transaction context to improve anomaly detection accuracy
    and reduce false positives.
    """
    
    def __init__(self,
                 embedding_model: str = 'all-MiniLM-L6-v2',
                 vector_db_path: str = "./chroma_db",
                 similarity_threshold: float = 0.7):
        """
        Initialize the RAG anomaly detector.
        
        Args:
            embedding_model: Sentence transformer model name
            vector_db_path: Path to vector database
            similarity_threshold: Similarity threshold for context matching
        """
        self.embedding_generator = TransactionEmbeddingGenerator(embedding_model)
        self.vector_db = VectorDatabase(persist_directory=vector_db_path)
        self.similarity_threshold = similarity_threshold
        
        self.enabled = self.embedding_generator.enabled and self.vector_db.enabled
        
        if self.enabled:
            logger.info("RAG anomaly detector initialized")
        else:
            logger.warning("RAG anomaly detector disabled due to missing dependencies")
    
    def index_historical_transactions(self,
                                     df: pd.DataFrame,
                                     batch_size: int = 1000) -> None:
        """
        Index historical transactions for retrieval.
        
        Args:
            df: DataFrame with historical transaction data
            batch_size: Batch size for processing
        """
        if not self.enabled:
            logger.warning("RAG detector not enabled")
            return
        
        logger.info(f"Indexing {len(df)} historical transactions")
        
        # Convert to list of dicts
        transactions = df.to_dict('records')
        
        # Process in batches
        for i in range(0, len(transactions), batch_size):
            end_idx = min(i + batch_size, len(transactions))
            batch = transactions[i:end_idx]
            
            # Generate embeddings
            embeddings = self.embedding_generator.generate_batch_embeddings(batch)
            
            # Add to vector database
            self.vector_db.add_transactions(batch, embeddings)
        
        # Persist database
        self.vector_db.persist()
        
        logger.info("Historical transactions indexed successfully")
    
    def detect_with_context(self,
                           transaction: Dict,
                           base_risk_score: float,
                           n_context: int = 20) -> Tuple[float, Dict]:
        """
        Detect anomalies using contextual information from similar transactions.
        
        Args:
            transaction: Transaction to evaluate
            base_risk_score: Risk score from other detection methods
            n_context: Number of similar transactions to retrieve
            
        Returns:
            Tuple of (adjusted_risk_score, context_info)
        """
        if not self.enabled:
            return base_risk_score, {}
        
        # Generate embedding for query transaction
        query_embedding = self.embedding_generator.generate_embedding(transaction)
        
        # Retrieve similar historical transactions
        results = self.vector_db.search_similar_transactions(
            query_embedding,
            n_results=n_context
        )
        
        if not results['metadatas'] or len(results['metadatas'][0]) == 0:
            return base_risk_score, {'context_available': False}
        
        # Analyze historical context
        fraud_count, legitimate_count, fraud_rate = self.vector_db.get_fraud_context(
            query_embedding,
            n_results=n_context
        )
        
        # Calculate average similarity
        distances = results['distances'][0] if results['distances'] else []
        avg_similarity = 1 - np.mean(distances) if distances else 0.0
        
        # Adjust risk score based on context
        adjusted_score = self._adjust_risk_score(
            base_risk_score,
            fraud_rate,
            avg_similarity
        )
        
        context_info = {
            'context_available': True,
            'similar_fraud_count': fraud_count,
            'similar_legitimate_count': legitimate_count,
            'historical_fraud_rate': fraud_rate,
            'avg_similarity': avg_similarity,
            'context_size': len(results['metadatas'][0]),
            'adjustment_factor': adjusted_score / base_risk_score if base_risk_score > 0 else 1.0
        }
        
        return adjusted_score, context_info
    
    def _adjust_risk_score(self,
                          base_score: float,
                          historical_fraud_rate: float,
                          similarity: float) -> float:
        """
        Adjust risk score based on historical context.
        
        Args:
            base_score: Original risk score
            historical_fraud_rate: Fraud rate in similar transactions
            similarity: Average similarity to historical transactions
            
        Returns:
            Adjusted risk score
        """
        # Only adjust if similarity is high enough
        if similarity < self.similarity_threshold:
            return base_score
        
        # Calculate adjustment factor based on historical fraud rate
        if historical_fraud_rate > 0.5:
            # High historical fraud rate -> increase risk score
            adjustment = 1.0 + (historical_fraud_rate - 0.5) * 0.5
        elif historical_fraud_rate < 0.1:
            # Low historical fraud rate -> decrease risk score
            adjustment = 0.7 + historical_fraud_rate * 3.0
        else:
            # Moderate fraud rate -> small adjustment
            adjustment = 0.9 + historical_fraud_rate
        
        # Weight adjustment by similarity
        final_adjustment = 1.0 + (adjustment - 1.0) * similarity
        
        # Apply adjustment
        adjusted_score = base_score * final_adjustment
        
        # Clamp to valid range
        adjusted_score = max(0.0, min(10.0, adjusted_score))
        
        return adjusted_score
    
    def batch_detect_with_context(self,
                                  df: pd.DataFrame,
                                  base_risk_scores: pd.Series,
                                  n_context: int = 20) -> Tuple[pd.Series, pd.DataFrame]:
        """
        Detect anomalies for multiple transactions using context.
        
        Args:
            df: DataFrame with transactions
            base_risk_scores: Series with base risk scores
            n_context: Number of similar transactions to retrieve
            
        Returns:
            Tuple of (adjusted_scores Series, context_info DataFrame)
        """
        if not self.enabled:
            return base_risk_scores, pd.DataFrame()
        
        logger.info(f"Processing {len(df)} transactions with contextual detection")
        
        adjusted_scores = []
        context_infos = []
        
        transactions = df.to_dict('records')
        
        for i, (txn, base_score) in enumerate(zip(transactions, base_risk_scores)):
            adjusted_score, context_info = self.detect_with_context(
                txn,
                base_score,
                n_context
            )
            
            adjusted_scores.append(adjusted_score)
            context_infos.append(context_info)
            
            if (i + 1) % 100 == 0:
                logger.info(f"Processed {i + 1}/{len(transactions)} transactions")
        
        adjusted_scores_series = pd.Series(adjusted_scores, index=df.index)
        context_df = pd.DataFrame(context_infos, index=df.index)
        
        logger.info("Contextual detection complete")
        
        return adjusted_scores_series, context_df
    
    def explain_context(self,
                       transaction: Dict,
                       n_examples: int = 5) -> List[Dict]:
        """
        Get example similar transactions for explanation.
        
        Args:
            transaction: Transaction to explain
            n_examples: Number of examples to return
            
        Returns:
            List of similar transaction dictionaries with metadata
        """
        if not self.enabled:
            return []
        
        # Generate embedding
        query_embedding = self.embedding_generator.generate_embedding(transaction)
        
        # Search for similar transactions
        results = self.vector_db.search_similar_transactions(
            query_embedding,
            n_results=n_examples
        )
        
        if not results['metadatas'] or len(results['metadatas'][0]) == 0:
            return []
        
        # Build explanation examples
        examples = []
        for i in range(len(results['metadatas'][0])):
            example = {
                'similarity': 1 - results['distances'][0][i],
                'amount': results['metadatas'][0][i].get('amount', 0),
                'type': results['metadatas'][0][i].get('type', 'UNKNOWN'),
                'was_fraud': bool(results['metadatas'][0][i].get('isFraud', 0)),
                'description': results['documents'][0][i] if results['documents'] else ''
            }
            examples.append(example)
        
        return examples
