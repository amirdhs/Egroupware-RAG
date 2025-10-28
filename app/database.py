"""
Qdrant Vector Database Service
Handles all vector storage and retrieval operations
"""

import logging
import hashlib
import uuid
import json
from pathlib import Path
from typing import List, Dict, Any, Optional
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance, VectorParams, PointStruct, Filter,
    FieldCondition, MatchValue, SearchRequest
)

logger = logging.getLogger(__name__)


class QdrantDatabase:
    """Qdrant vector database client with user isolation"""

    def __init__(self, config: Dict[str, Any], embedding_dimension: int):
        self.config = config
        self.embedding_dimension = embedding_dimension

        # Initialize Qdrant client
        qdrant_config = config.get('qdrant', {})
        mode = qdrant_config.get('mode', 'memory')

        if mode == 'memory':
            logger.info("Initializing Qdrant in memory mode")
            self.client = QdrantClient(":memory:")
        elif mode == 'disk':
            path = qdrant_config.get('path', './qdrant_storage')
            logger.info(f"Initializing Qdrant with disk storage: {path}")
            self.client = QdrantClient(path=path)
        elif mode == 'server':
            host = qdrant_config.get('host', 'localhost')
            port = qdrant_config.get('port', 6333)
            logger.info(f"Connecting to Qdrant server: {host}:{port}")
            self.client = QdrantClient(host=host, port=port)
        else:
            raise ValueError(f"Unsupported Qdrant mode: {mode}")

        self.collection_name = "egroupware_rag"
        self.current_user_id = None
        # Avoid repeated validation/recreation within the same process
        self._collection_validated = False

        # Create collection if it doesn't exist
        self._ensure_collection()

    def _ensure_collection(self):
        """Ensure the collection exists with proper configuration"""
        try:
            # Use a small local meta file to remember the collection vector size
            qdrant_config = self.config.get('qdrant', {})
            path = qdrant_config.get('path', './qdrant_storage')
            meta_path = Path(path) / 'collection_meta.json'

            if meta_path.exists():
                try:
                    meta = json.loads(meta_path.read_text())
                    existing_meta_size = meta.get('vector_size')
                    if existing_meta_size is not None and int(existing_meta_size) == int(self.embedding_dimension):
                        logger.info(f"✅ Collection metadata matches vector size {existing_meta_size}; skipping recreation")
                        self._collection_validated = True
                        return
                except Exception:
                    # If meta file is corrupted, fall through and revalidate/create collection
                    logger.warning("Collection meta file unreadable, will validate via Qdrant client")

            collections = self.client.get_collections().collections
            collection_exists = any(c.name == self.collection_name for c in collections)

            if not collection_exists:
                logger.info(f"Creating collection: {self.collection_name}")
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(
                        size=self.embedding_dimension,
                        distance=Distance.COSINE
                    )
                )
                logger.info("✅ Collection created successfully")
            else:
                # If a collection exists, ensure its vector size matches the embedding dimension.
                # If it doesn't match, delete and recreate the collection to avoid shape errors.
                try:
                    info = None
                    try:
                        info = self.client.get_collection(self.collection_name)
                    except Exception:
                        # Some client implementations may not expose get_collection; fall back
                        info = None

                    existing_size = None
                    if info is not None:
                        # Try a couple of likely attribute paths to find stored vector size
                        try:
                            existing_size = info.params.vectors.size
                        except Exception:
                            try:
                                # Older/newer versions may expose a nested dict
                                existing_size = info.vectors.get('size') if hasattr(info, 'vectors') else None
                            except Exception:
                                existing_size = None

                    if existing_size is None:
                        logger.warning(
                            f"Could not determine existing collection vector size for '{self.collection_name}'. "
                            f"Recreating to ensure correct size ({self.embedding_dimension})."
                        )
                        self.client.delete_collection(collection_name=self.collection_name)
                        self.client.create_collection(
                            collection_name=self.collection_name,
                            vectors_config=VectorParams(
                                size=self.embedding_dimension,
                                distance=Distance.COSINE
                            )
                        )
                        logger.info(
                            f"✅ Collection '{self.collection_name}' recreated with size {self.embedding_dimension}"
                        )
                        # Persist meta
                        try:
                            Path(path).mkdir(parents=True, exist_ok=True)
                            meta_path.write_text(json.dumps({'collection_name': self.collection_name, 'vector_size': self.embedding_dimension}))
                        except Exception:
                            logger.warning("Failed to write collection meta file")
                    elif int(existing_size) != int(self.embedding_dimension):
                        logger.warning(f"Existing collection vector size ({existing_size}) does not match embedding dimension ({self.embedding_dimension}). Recreating collection.")
                        self.client.delete_collection(collection_name=self.collection_name)
                        self.client.create_collection(
                            collection_name=self.collection_name,
                            vectors_config=VectorParams(
                                size=self.embedding_dimension,
                                distance=Distance.COSINE
                            )
                        )
                        logger.info(
                            f"✅ Collection '{self.collection_name}' recreated with size {self.embedding_dimension}"
                        )
                        # Persist meta
                        try:
                            Path(path).mkdir(parents=True, exist_ok=True)
                            meta_path.write_text(json.dumps({'collection_name': self.collection_name, 'vector_size': self.embedding_dimension}))
                        except Exception:
                            logger.warning("Failed to write collection meta file")
                    else:
                        logger.info(
                            f"✅ Collection '{self.collection_name}' already exists and matches vector size {existing_size}"
                        )
                        # Persist meta for faster checks next time
                        try:
                            Path(path).mkdir(parents=True, exist_ok=True)
                            meta_path.write_text(json.dumps({'collection_name': self.collection_name, 'vector_size': int(existing_size)}))
                        except Exception:
                            logger.debug("Failed to write collection meta file; non-fatal")
                    # Mark validated so we don't repeat heavy checks in this process
                    self._collection_validated = True

                except Exception as e:
                    logger.error(f"Error while validating/recreating collection: {e}")
                    raise
        except Exception as e:
            logger.error(f"Failed to ensure collection: {e}")
            raise

    def set_user_id(self, user_id: str):
        """Set the current user ID for all operations"""
        self.current_user_id = user_id
        logger.info(f"User context set to: {user_id}")

    def _generate_point_id(self, user_id: str, app_name: str, doc_id: str) -> str:
        """Generate a unique point ID based on user, app, and document"""
        unique_string = f"{user_id}:{app_name}:{doc_id}"
        hash_hex = hashlib.sha256(unique_string.encode()).hexdigest()
        # Convert hash to valid UUID format (take first 32 hex chars and format as UUID)
        uuid_str = f"{hash_hex[:8]}-{hash_hex[8:12]}-{hash_hex[12:16]}-{hash_hex[16:20]}-{hash_hex[20:32]}"
        return uuid_str

    def insert_document(self, doc_id: str, app_name: str, content: str,
                       embedding: List[float], metadata: Dict[str, Any]):
        """Insert a single document into Qdrant"""
        if not self.current_user_id:
            raise ValueError("User ID must be set before inserting documents")

        point_id = self._generate_point_id(self.current_user_id, app_name, doc_id)

        payload = {
            'user_id': self.current_user_id,
            'doc_id': doc_id,
            'app_name': app_name,
            'content': content,
            'metadata': metadata
        }

        point = PointStruct(
            id=point_id,
            vector=embedding,
            payload=payload
        )

        self.client.upsert(
            collection_name=self.collection_name,
            points=[point]
        )

    def insert_documents_batch(self, documents: List[Dict[str, Any]]):
        """Insert multiple documents in batch"""
        if not self.current_user_id:
            raise ValueError("User ID must be set before inserting documents")

        points = []
        for doc in documents:
            point_id = self._generate_point_id(
                self.current_user_id,
                doc['app_name'],
                doc['doc_id']
            )

            payload = {
                'user_id': self.current_user_id,
                'doc_id': doc['doc_id'],
                'app_name': doc['app_name'],
                'content': doc['content'],
                'metadata': doc.get('metadata', {})
            }

            point = PointStruct(
                id=point_id,
                vector=doc['embedding'],
                payload=payload
            )
            points.append(point)

        # Batch upsert
        self.client.upsert(
            collection_name=self.collection_name,
            points=points
        )

        logger.info(f"Inserted {len(points)} documents")

    def search(self, query_embedding: List[float], query_text: str = "",
               app_filter: Optional[str] = None, top_k: int = 5) -> List[Dict[str, Any]]:
        """Search for similar documents with hybrid scoring (semantic + text matching)"""
        if not self.current_user_id:
            raise ValueError("User ID must be set before searching")

        # DISTINCTIVE LOG MESSAGE TO VERIFY NEW CODE IS RUNNING
        logger.info("🚀 HYBRID SEARCH METHOD CALLED WITH NEW CODE!")
        logger.info(f"Database search: user_id={self.current_user_id}, "
                   f"query_text='{query_text}', app_filter={app_filter}, "
                   f"top_k={top_k}")
        
        # Log embedding info
        logger.info(f"Query embedding length: {len(query_embedding)}, "
                   f"vs expected dimension: {self.embedding_dimension}")

        # Build filter
        must_conditions = [
            FieldCondition(key="user_id", 
                          match=MatchValue(value=self.current_user_id))
        ]

        if app_filter:
            must_conditions.append(
                FieldCondition(key="app_name", match=MatchValue(value=app_filter))
            )

        search_filter = Filter(must=must_conditions)
        logger.info(f"Search filter: {search_filter.must}")
        
        # Count total matching documents before searching (for debugging)
        try:
            count_result = self.client.count(
                collection_name=self.collection_name,
                count_filter=search_filter
            )
            logger.info(f"Total matching documents with this filter: {count_result.count}")
        except Exception as e:
            logger.warning(f"Failed to count documents: {e}")

        # Perform semantic search
        try:
            # Increase limit to get more candidates for hybrid scoring
            semantic_limit = max(top_k * 3, 50)
            results = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_embedding,
                query_filter=search_filter,
                limit=semantic_limit
            )
            logger.info(f"Semantic search returned {len(results)} results")
        except Exception as e:
            logger.error(f"Search failed: {e}")
            raise

        # Apply hybrid scoring (semantic + text matching)
        formatted_results = []
        query_lower = query_text.lower() if query_text else ""
        
        for result in results:
            content = result.payload['content']
            content_lower = content.lower()
            semantic_score = result.score
            
            # Calculate text matching boost
            text_boost = 0.0
            if query_lower and query_lower in content_lower:
                # Exact substring match gets significant boost
                text_boost = 0.3
                
                # Additional boost for matches in key fields (name, etc.)
                contact_match = f"contact: {query_lower}" in content_lower
                name_match = f"name: {query_lower}" in content_lower
                if contact_match or name_match:
                    text_boost += 0.2
                    
                # Boost for word boundary matches (whole word matches)
                import re
                if re.search(r'\b' + re.escape(query_lower) + r'\b', 
                           content_lower):
                    text_boost += 0.1
                    
                # Boost for matches at the beginning of content
                if content_lower.startswith(f"contact: {query_lower}"):
                    text_boost += 0.2
            
            # Combine semantic score with text boost
            # Semantic score is typically 0.0-1.0, we add text boost on top
            hybrid_score = min(semantic_score + text_boost, 1.0)
            
            formatted_results.append({
                'doc_id': result.payload['doc_id'],
                'app_name': result.payload['app_name'],
                'content': result.payload['content'],
                'metadata': result.payload.get('metadata', {}),
                'similarity': hybrid_score,
                'semantic_score': semantic_score,
                'text_boost': text_boost
            })

        # Sort by hybrid score and take top_k
        formatted_results.sort(key=lambda x: x['similarity'], reverse=True)
        formatted_results = formatted_results[:top_k]
        
        # Log scoring details for debugging
        if formatted_results and query_lower:
            logger.info("Hybrid scoring results:")
            for i, result in enumerate(formatted_results[:3]):
                logger.info(f"  {i+1}. semantic={result['semantic_score']:.3f}, "
                           f"boost={result['text_boost']:.3f}, "
                           f"final={result['similarity']:.3f}, "
                           f"content='{result['content'][:50]}...'")

        logger.info(f"Returning {len(formatted_results)} hybrid-scored results")
        return formatted_results

    def delete_user_documents(self, app_name: Optional[str] = None) -> int:
        """Delete documents for current user, optionally filtered by app"""
        if not self.current_user_id:
            raise ValueError("User ID must be set before deleting documents")

        must_conditions = [
            FieldCondition(key="user_id", match=MatchValue(value=self.current_user_id))
        ]

        if app_name:
            must_conditions.append(
                FieldCondition(key="app_name", match=MatchValue(value=app_name))
            )

        delete_filter = Filter(must=must_conditions)

        # Get count before deletion
        count_result = self.client.count(
            collection_name=self.collection_name,
            count_filter=delete_filter
        )
        count_before = count_result.count

        # Delete
        self.client.delete(
            collection_name=self.collection_name,
            points_selector=delete_filter
        )

        logger.info(f"Deleted {count_before} documents for user {self.current_user_id}")
        return count_before

    def get_stats(self) -> Dict[str, Any]:
        """Get database statistics for current user"""
        if not self.current_user_id:
            return {'total_documents': 0, 'by_app': {}}

        # Get total count for user
        user_filter = Filter(
            must=[FieldCondition(key="user_id", match=MatchValue(value=self.current_user_id))]
        )

        total_count = self.client.count(
            collection_name=self.collection_name,
            count_filter=user_filter
        ).count

        # Get counts by app
        stats_by_app = {}
        for app in ['addressbook', 'calendar', 'infolog']:
            app_filter = Filter(
                must=[
                    FieldCondition(key="user_id", match=MatchValue(value=self.current_user_id)),
                    FieldCondition(key="app_name", match=MatchValue(value=app))
                ]
            )
            count = self.client.count(
                collection_name=self.collection_name,
                count_filter=app_filter
            ).count
            stats_by_app[app] = count

        return {
            'total_documents': total_count,
            'by_app': stats_by_app
        }

    def reset_user_data(self) -> int:
        """Reset all data for current user"""
        return self.delete_user_documents()

    def get_connection(self):
        """Return client for advanced operations (context manager compatibility)"""
        return self

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass
