"""
MariaDB Vector Database Service
Handles all vector storage and retrieval operations using MariaDB
"""

import logging
import hashlib
import json
import pickle
import numpy as np
from typing import List, Dict, Any, Optional
import pymysql
from pymysql.cursors import DictCursor

logger = logging.getLogger(__name__)


class MariaDBDatabase:
    """MariaDB vector database client with user isolation"""

    def __init__(self, config: Dict[str, Any], embedding_dimension: int):
        self.config = config
        self.embedding_dimension = embedding_dimension
        self.current_user_id = None

        # Initialize MariaDB connection
        mariadb_config = config.get('mariadb', {})
        
        self.connection_params = {
            'host': mariadb_config.get('host', 'localhost'),
            'port': mariadb_config.get('port', 3306),
            'user': mariadb_config.get('user', 'rag_user'),
            'password': mariadb_config.get('password', 'rag_password'),
            'database': mariadb_config.get('database', 'rag_vectors'),
            'charset': 'utf8mb4',
            'cursorclass': DictCursor,
            'autocommit': True
        }

        logger.info(f"Initializing MariaDB connection to {self.connection_params['host']}:{self.connection_params['port']}")
        
        # Test connection
        self._test_connection()
        logger.info("✅ MariaDB connection successful")

    def _test_connection(self):
        """Test database connection"""
        try:
            conn = pymysql.connect(**self.connection_params)
            conn.close()
        except Exception as e:
            logger.error(f"Failed to connect to MariaDB: {e}")
            raise

    def _get_connection(self):
        """Get a new database connection"""
        return pymysql.connect(**self.connection_params)

    def set_user_id(self, user_id: str):
        """Set the current user ID for all operations"""
        self.current_user_id = user_id
        logger.info(f"User context set to: {user_id}")

    def _generate_point_id(self, user_id: str, app_name: str, doc_id: str) -> str:
        """Generate a unique point ID based on user, app, and document"""
        unique_string = f"{user_id}:{app_name}:{doc_id}"
        hash_hex = hashlib.sha256(unique_string.encode()).hexdigest()
        return hash_hex

    def _serialize_embedding(self, embedding: List[float]) -> bytes:
        """Serialize embedding vector to binary format"""
        return pickle.dumps(np.array(embedding, dtype=np.float32))

    def _deserialize_embedding(self, data: bytes) -> List[float]:
        """Deserialize embedding vector from binary format"""
        return pickle.loads(data).tolist()

    def insert_document(self, doc_id: str, app_name: str, content: str,
                       embedding: List[float], metadata: Dict[str, Any]):
        """Insert a single document into MariaDB"""
        if not self.current_user_id:
            raise ValueError("User ID must be set before inserting documents")

        point_id = self._generate_point_id(self.current_user_id, app_name, doc_id)
        embedding_blob = self._serialize_embedding(embedding)
        metadata_json = json.dumps(metadata)

        conn = self._get_connection()
        try:
            with conn.cursor() as cursor:
                sql = """
                    INSERT INTO documents (id, user_id, doc_id, app_name, content, embedding, metadata)
                    VALUES (%s, %s, %s, %s, %s, %s, %s)
                    ON DUPLICATE KEY UPDATE
                        content = VALUES(content),
                        embedding = VALUES(embedding),
                        metadata = VALUES(metadata)
                """
                cursor.execute(sql, (point_id, self.current_user_id, doc_id, app_name, 
                                   content, embedding_blob, metadata_json))
            conn.commit()
        finally:
            conn.close()

    def insert_documents_batch(self, documents: List[Dict[str, Any]]):
        """Insert multiple documents in batch"""
        if not self.current_user_id:
            raise ValueError("User ID must be set before inserting documents")

        conn = self._get_connection()
        try:
            with conn.cursor() as cursor:
                sql = """
                    INSERT INTO documents (id, user_id, doc_id, app_name, content, embedding, metadata)
                    VALUES (%s, %s, %s, %s, %s, %s, %s)
                    ON DUPLICATE KEY UPDATE
                        content = VALUES(content),
                        embedding = VALUES(embedding),
                        metadata = VALUES(metadata)
                """
                
                batch_data = []
                for doc in documents:
                    point_id = self._generate_point_id(
                        self.current_user_id,
                        doc['app_name'],
                        doc['doc_id']
                    )
                    embedding_blob = self._serialize_embedding(doc['embedding'])
                    metadata_json = json.dumps(doc.get('metadata', {}))
                    
                    batch_data.append((
                        point_id,
                        self.current_user_id,
                        doc['doc_id'],
                        doc['app_name'],
                        doc['content'],
                        embedding_blob,
                        metadata_json
                    ))
                
                cursor.executemany(sql, batch_data)
            conn.commit()
            logger.info(f"Inserted {len(documents)} documents")
        finally:
            conn.close()

    def _cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Calculate cosine similarity between two vectors"""
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return float(dot_product / (norm1 * norm2))

    def search(self, query_embedding: List[float], query_text: str = "",
               app_filter: Optional[str] = None, top_k: int = 5) -> List[Dict[str, Any]]:
        """Search for similar documents with hybrid scoring (semantic + text matching)"""
        if not self.current_user_id:
            raise ValueError("User ID must be set before searching")

        logger.info("🚀 HYBRID SEARCH METHOD CALLED WITH MARIADB!")
        logger.info(f"Database search: user_id={self.current_user_id}, "
                   f"query_text='{query_text}', app_filter={app_filter}, "
                   f"top_k={top_k}")

        query_vector = np.array(query_embedding, dtype=np.float32)

        conn = self._get_connection()
        try:
            with conn.cursor() as cursor:
                # Build query with filters
                sql = """
                    SELECT id, user_id, doc_id, app_name, content, embedding, metadata
                    FROM documents
                    WHERE user_id = %s
                """
                params = [self.current_user_id]
                
                if app_filter:
                    sql += " AND app_name = %s"
                    params.append(app_filter)
                
                cursor.execute(sql, params)
                rows = cursor.fetchall()
                
                logger.info(f"Retrieved {len(rows)} documents from database")

                # Calculate similarities and apply hybrid scoring
                results = []
                query_lower = query_text.lower() if query_text else ""
                
                for row in rows:
                    # Deserialize embedding
                    doc_embedding = self._deserialize_embedding(row['embedding'])
                    doc_vector = np.array(doc_embedding, dtype=np.float32)
                    
                    # Calculate semantic similarity
                    semantic_score = self._cosine_similarity(query_vector, doc_vector)
                    
                    # Apply text matching boost
                    content = row['content']
                    content_lower = content.lower()
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
                        if re.search(r'\b' + re.escape(query_lower) + r'\b', content_lower):
                            text_boost += 0.1
                            
                        # Boost for matches at the beginning of content
                        if content_lower.startswith(f"contact: {query_lower}"):
                            text_boost += 0.2
                    
                    # Combine semantic score with text boost
                    hybrid_score = min(semantic_score + text_boost, 1.0)
                    
                    metadata = json.loads(row['metadata']) if row['metadata'] else {}
                    
                    results.append({
                        'doc_id': row['doc_id'],
                        'app_name': row['app_name'],
                        'content': row['content'],
                        'metadata': metadata,
                        'similarity': hybrid_score,
                        'semantic_score': semantic_score,
                        'text_boost': text_boost
                    })

                # Sort by hybrid score and take top_k
                results.sort(key=lambda x: x['similarity'], reverse=True)
                results = results[:top_k]
                
                # Log scoring details for debugging
                if results and query_lower:
                    logger.info("Hybrid scoring results:")
                    for i, result in enumerate(results[:3]):
                        logger.info(f"  {i+1}. semantic={result['semantic_score']:.3f}, "
                                   f"boost={result['text_boost']:.3f}, "
                                   f"final={result['similarity']:.3f}, "
                                   f"content='{result['content'][:50]}...'")

                logger.info(f"Returning {len(results)} hybrid-scored results")
                return results
        finally:
            conn.close()

    def delete_user_documents(self, app_name: Optional[str] = None) -> int:
        """Delete documents for current user, optionally filtered by app"""
        if not self.current_user_id:
            raise ValueError("User ID must be set before deleting documents")

        conn = self._get_connection()
        try:
            with conn.cursor() as cursor:
                if app_name:
                    sql = "DELETE FROM documents WHERE user_id = %s AND app_name = %s"
                    cursor.execute(sql, (self.current_user_id, app_name))
                else:
                    sql = "DELETE FROM documents WHERE user_id = %s"
                    cursor.execute(sql, (self.current_user_id,))
                
                deleted_count = cursor.rowcount
            conn.commit()
            logger.info(f"Deleted {deleted_count} documents for user {self.current_user_id}")
            return deleted_count
        finally:
            conn.close()

    def get_stats(self) -> Dict[str, Any]:
        """Get database statistics for current user"""
        if not self.current_user_id:
            return {'total_documents': 0, 'by_app': {}}

        conn = self._get_connection()
        try:
            with conn.cursor() as cursor:
                # Get total count
                sql = "SELECT COUNT(*) as count FROM documents WHERE user_id = %s"
                cursor.execute(sql, (self.current_user_id,))
                total_count = cursor.fetchone()['count']

                # Get counts by app
                sql = """
                    SELECT app_name, COUNT(*) as count 
                    FROM documents 
                    WHERE user_id = %s 
                    GROUP BY app_name
                """
                cursor.execute(sql, (self.current_user_id,))
                rows = cursor.fetchall()
                
                stats_by_app = {row['app_name']: row['count'] for row in rows}
                
                # Ensure all apps are represented
                for app in ['addressbook', 'calendar', 'infolog']:
                    if app not in stats_by_app:
                        stats_by_app[app] = 0

            return {
                'total_documents': total_count,
                'by_app': stats_by_app
            }
        finally:
            conn.close()

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
