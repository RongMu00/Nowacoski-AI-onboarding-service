"""
MongoDB Vector Store with SSL/TLS fixes for MongoDB Atlas
"""

import logging
import os
from typing import Any, Dict, List, Optional
from urllib.parse import quote_plus

import numpy as np
from pymongo import MongoClient
from pymongo.errors import ServerSelectionTimeoutError


class MongoVectorStore:
    """MongoDB-based vector store for caching embeddings and documents"""

    def __init__(self, embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"):
        """
        Initialize MongoDB Vector Store with SSL/TLS fixes for Atlas

        Args:
            embedding_model: Name of the embedding model to use
        """
        self.logger = logging.getLogger("vector_store")

        # Get MongoDB connection details from environment
        mongo_uri = os.getenv("MONGODB_URI")

        if not mongo_uri:
            # Fallback: Build from separate credentials
            username = os.getenv("MONGODB_USERNAME", "nowacoski")
            password = os.getenv("MONGODB_PASSWORD", "")
            cluster = os.getenv("MONGODB_CLUSTER", "localhost:27017")

            # URL encode credentials (handle special characters)
            encoded_username = quote_plus(username)
            encoded_password = quote_plus(password) if password else ""

            # Build URI based on cluster type
            if "mongodb+srv" in cluster or "@" in cluster:
                mongo_uri = f"mongodb+srv://{encoded_username}:{encoded_password}@{cluster}/nowacoski?retryWrites=true&w=majority"
            else:
                mongo_uri = f"mongodb://{encoded_username}:{encoded_password}@{cluster}/nowacoski"

        # Database and collection names
        self.db_name = os.getenv("DB_NAME", "nowacoski")
        self.collection_name = os.getenv("COLLECTION_NAME", "onboarding_content")

        # Initialize MongoDB connection with SSL handling
        self._init_mongo_connection(mongo_uri)

        # Initialize embedding model
        self._init_embeddings(embedding_model)

    def _init_mongo_connection(self, mongo_uri: str):
        """Initialize MongoDB connection with multiple SSL/TLS strategies"""

        self.client = None
        self.db = None
        self.collection = None

        # Connection strategies (try each in order)
        strategies = [
            {
                "name": "Default SSL",
                "kwargs": {
                    "serverSelectionTimeoutMS": 30000,
                    "connectTimeoutMS": 30000,
                }
            },
            {
                "name": "SSL with tlsInsecure=True",
                "kwargs": {
                    "serverSelectionTimeoutMS": 30000,
                    "connectTimeoutMS": 30000,
                    "tlsInsecure": True,  # Disable cert verification (temporary)
                }
            },
        ]

        for strategy in strategies:
            try:
                self.logger.info(f"🔄 Attempting MongoDB connection: {strategy['name']}...")

                # Try to connect
                client = MongoClient(mongo_uri, **strategy["kwargs"])

                # Test connection
                client.admin.command('ping')

                # Connection successful!
                self.client = client
                self.db = client[self.db_name]
                self.collection = self.db[self.collection_name]

                if strategy["kwargs"].get("tlsInsecure"):
                    self.logger.warning("⚠️  Connected with tlsInsecure=True (not recommended for production)")
                else:
                    self.logger.info("✅ Connected with standard SSL")

                self.logger.info(f"✅ MongoDB Vector Store Connected!")
                self._ensure_indexes()
                return

            except (ServerSelectionTimeoutError, Exception) as e:
                error_msg = str(e)[:80]
                self.logger.debug(f"   ❌ {strategy['name']} failed: {error_msg}")
                continue

        # All strategies failed
        self.logger.error("❌ Could not connect to MongoDB with any strategy")
        self.logger.error("   Please verify:")
        self.logger.error("   • MONGODB_URI is correct in .env")
        self.logger.error("   • MongoDB Atlas cluster is running")
        self.logger.error("   • Network IP is whitelisted")
        self.collection = None

    def _init_embeddings(self, embedding_model: str):
        """Initialize embedding model"""

        if not self.collection:
            self.logger.error("❌ Cannot initialize embeddings - MongoDB not connected")
            self.embeddings_model = None
            self.embedding_dim = None
            return

        try:
            from sentence_transformers import SentenceTransformer

            self.logger.info(f"📦 Loading embedding model: {embedding_model}...")
            self.embeddings_model = SentenceTransformer(embedding_model)
            self.embedding_dim = self.embeddings_model.get_sentence_embedding_dimension()

            self.logger.info(f"✅ Embedding model ready (dimension: {self.embedding_dim})")

        except ImportError:
            self.logger.error("❌ sentence-transformers not installed")
            self.logger.error("   Run: pip install sentence-transformers")
            self.embeddings_model = None
            self.embedding_dim = None

        except Exception as e:
            self.logger.error(f"❌ Failed to load embedding model: {e}")
            self.embeddings_model = None
            self.embedding_dim = None

    def _ensure_indexes(self):
        """Create necessary indexes for efficient querying"""

        if not self.collection:
            return

        try:
            # Create text index on content for simple_search fallback
            self.collection.create_index([("content", "text")], name="content_text_idx")
            self.logger.info("✅ Text index created on 'content'")
        except Exception as e:
            # Index may already exist — that's fine
            self.logger.debug(f"Index creation note: {e}")

        try:
            # Create compound index for folder_id + source lookups
            self.collection.create_index(
                [("metadata.folder_id", 1)],
                name="folder_id_idx"
            )
        except Exception as e:
            self.logger.debug(f"Folder index note: {e}")

    def store_document(
        self,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
        source: Optional[str] = None,
        folder_id: Optional[str] = None
    ) -> str:
        """
        Store a document with its embedding

        Args:
            content: Document content to store
            metadata: Optional metadata dictionary
            source: Source identifier (e.g., URL, file path)
            folder_id: Optional folder/group identifier for organizing documents

        Returns:
            Document ID
        """

        if not self.collection or not self.embeddings_model:
            self.logger.warning("⚠️  Vector store not available - document not stored")
            return None

        try:
            # Generate embedding
            embedding = self.embeddings_model.encode(content).tolist()

            # Merge folder_id into metadata
            doc_metadata = metadata or {}
            if folder_id:
                doc_metadata["folder_id"] = folder_id

            # Create document
            doc = {
                "content": content,
                "embedding": embedding,
                "metadata": doc_metadata,
                "source": source,
            }

            # Store in MongoDB
            result = self.collection.insert_one(doc)

            return str(result.inserted_id)

        except Exception as e:
            self.logger.error(f"❌ Failed to store document: {e}")
            return None

    def search_similar(
        self,
        query: str,
        top_k: int = 5,
        threshold: float = 0.3,
        folder_id: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Search for similar documents using cosine similarity

        Args:
            query: Search query
            top_k: Number of results to return
            threshold: Similarity threshold (0-1)
            folder_id: Optional folder to limit search scope

        Returns:
            List of similar documents with scores
        """

        if not self.collection or not self.embeddings_model:
            self.logger.warning("⚠️  Vector store not available - returning empty results")
            return []

        try:
            # Generate query embedding
            query_embedding = np.array(self.embeddings_model.encode(query))

            # Build MongoDB filter
            mongo_filter = {}
            if folder_id:
                mongo_filter["metadata.folder_id"] = folder_id

            # Fetch documents with embeddings from MongoDB
            cursor = self.collection.find(
                mongo_filter,
                {"content": 1, "metadata": 1, "source": 1, "embedding": 1}
            )

            # Compute cosine similarity in Python
            scored_results = []
            for doc in cursor:
                doc_embedding = doc.get("embedding")
                if not doc_embedding:
                    continue

                doc_vec = np.array(doc_embedding)

                # Cosine similarity = dot(a, b) / (||a|| * ||b||)
                dot_product = np.dot(query_embedding, doc_vec)
                query_norm = np.linalg.norm(query_embedding)
                doc_norm = np.linalg.norm(doc_vec)

                if query_norm == 0 or doc_norm == 0:
                    similarity = 0.0
                else:
                    similarity = float(dot_product / (query_norm * doc_norm))

                if similarity >= threshold:
                    scored_results.append({
                        "id": str(doc.get("_id")),
                        "content": doc.get("content", ""),
                        "metadata": doc.get("metadata", {}),
                        "source": doc.get("source"),
                        "similarity": similarity
                    })

            # Sort by similarity descending and take top_k
            scored_results.sort(key=lambda x: x["similarity"], reverse=True)
            return scored_results[:top_k]

        except Exception as e:
            self.logger.error(f"❌ Search failed: {e}")
            return []

    def simple_search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Simple text search (fallback if vector search fails)

        Args:
            query: Search query
            top_k: Number of results to return

        Returns:
            List of matching documents
        """

        if not self.collection:
            return []

        try:
            results = list(self.collection.find(
                {"$text": {"$search": query}},
                {"score": {"$meta": "textScore"}}
            ).sort([("score", {"$meta": "textScore"})]).limit(top_k))

            formatted_results = [
                {
                    "id": str(r.get("_id")),
                    "content": r.get("content", ""),
                    "metadata": r.get("metadata", {}),
                    "source": r.get("source"),
                    "score": r.get("score", 0)
                }
                for r in results
            ]

            return formatted_results

        except Exception as e:
            self.logger.debug(f"Text search not available: {e}")
            return []

    def get_documents_by_folder(
        self,
        folder_id: str,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """Get documents from a specific folder"""

        if not self.collection:
            return []

        try:
            results = list(self.collection.find(
                {"metadata.folder_id": folder_id},
                {"embedding": 0}  # Exclude embeddings for performance
            ).limit(limit))

            formatted_results = [
                {
                    "id": str(r.get("_id")),
                    "content": r.get("content", ""),
                    "metadata": r.get("metadata", {}),
                    "source": r.get("source")
                }
                for r in results
            ]

            return formatted_results

        except Exception as e:
            self.logger.error(f"❌ Failed to get documents by folder: {e}")
            return []

    def delete_folder_documents(self, folder_id: str) -> int:
        """Delete all documents in a specific folder

        Args:
            folder_id: Folder identifier to delete documents from

        Returns:
            Number of documents deleted
        """

        if not self.collection:
            return 0

        try:
            result = self.collection.delete_many({"metadata.folder_id": folder_id})
            deleted = result.deleted_count
            self.logger.info(f"🗑️  Deleted {deleted} documents from folder '{folder_id}'")
            return deleted
        except Exception as e:
            self.logger.error(f"❌ Failed to delete folder documents: {e}")
            return 0

    def get_stats(self) -> Dict[str, Any]:
        """Get vector store statistics"""

        if not self.collection:
            return {
                "status": "disconnected",
                "embedding_dimension": None,
                "document_count": 0,
                "ready": False
            }

        try:
            count = self.collection.count_documents({})

            return {
                "status": "connected",
                "embedding_dimension": self.embedding_dim,
                "document_count": count,
                "ready": True,
                "database": self.db_name,
                "collection": self.collection_name
            }

        except Exception as e:
            self.logger.error(f"Failed to get stats: {e}")
            return {
                "status": "error",
                "error": str(e)
            }

    def health_check(self) -> bool:
        """Check if vector store is healthy"""

        if not self.client:
            return False

        try:
            self.client.admin.command('ping')
            return True
        except Exception:
            return False

    def clear_all(self) -> bool:
        """Clear all documents (use with caution!)"""

        if not self.collection:
            return False

        try:
            self.collection.delete_many({})
            self.logger.warning("⚠️  All documents cleared from vector store")
            return True
        except Exception as e:
            self.logger.error(f"Failed to clear collection: {e}")
            return False

    def close(self):
        """Close MongoDB connection"""

        if self.client:
            self.client.close()
            self.logger.info("MongoDB connection closed")


# Singleton instance for easier access
_vector_store = None


def get_vector_store() -> Optional[MongoVectorStore]:
    """Get or create the vector store singleton"""

    global _vector_store

    if _vector_store is None:
        _vector_store = MongoVectorStore()

    return _vector_store


if __name__ == "__main__":
    # Test the vector store
    from dotenv import load_dotenv
    load_dotenv()

    print("Testing MongoDB Vector Store...")
    print("=" * 60)

    store = MongoVectorStore()
    stats = store.get_stats()

    print(f"Status: {stats.get('status')}")
    print(f"Connected: {stats.get('ready')}")
    print(f"Document count: {stats.get('document_count')}")
    print(f"Embedding dimension: {stats.get('embedding_dimension')}")

    if stats.get('ready'):
        print("✅ Vector Store Ready!")
    else:
        print("❌ Vector Store Not Ready")
