"""
Embedding Service for Vector Generation
Supports HuggingFace, OpenAI, and IONOS embedding models
"""

import logging
import numpy as np
from typing import List, Dict, Any

logger = logging.getLogger(__name__)


class EmbeddingService:
    """Service for generating embeddings from text"""

    def __init__(self, config: Dict[str, Any]):
        self.provider = config.get('provider', 'huggingface').lower()
        self.model = None
        self.dimension = 0

        if self.provider == 'huggingface':
            self._init_huggingface(config)
        elif self.provider == 'openai':
            self._init_openai(config)
        elif self.provider == 'ionos':
            self._init_ionos(config)
        else:
            raise ValueError(f"Unsupported embedding provider: {self.provider}")

    def _init_huggingface(self, config: Dict[str, Any]):
        """Initialize HuggingFace sentence transformer"""
        try:
            from sentence_transformers import SentenceTransformer

            model_name = config.get('hf_model', 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
            logger.info(f"Loading HuggingFace model: {model_name}")
            self.model = SentenceTransformer(model_name)
            self.dimension = self.model.get_sentence_embedding_dimension()
            logger.info(f"✅ HuggingFace embeddings initialized (dimension: {self.dimension})")
        except Exception as e:
            logger.error(f"Failed to initialize HuggingFace embeddings: {e}")
            raise

    def _init_openai(self, config: Dict[str, Any]):
        """Initialize OpenAI embeddings"""
        try:
            import openai

            api_key = config.get('openai_key', config.get('api_key', ''))
            if not api_key:
                raise ValueError("OpenAI API key is required")

            self.client = openai.OpenAI(api_key=api_key)
            self.model_name = config.get('openai_model', 'text-embedding-3-small')
            self.dimension = 1536  # Default for text-embedding-3-small

            logger.info(f"✅ OpenAI embeddings initialized: {self.model_name}")
        except Exception as e:
            logger.error(f"Failed to initialize OpenAI embeddings: {e}")
            raise

    def _init_ionos(self, config: Dict[str, Any]):
        """Initialize IONOS embeddings"""
        try:
            import requests

            api_key = config.get('api_key', '')
            api_url = config.get('api_url', '')

            if not api_key or not api_url:
                raise ValueError("IONOS API key and URL are required")

            self.client = requests
            self.api_key = api_key
            self.api_url = api_url.rstrip('/') + '/embeddings'
            self.model_name = config.get('model', 'BAAI/bge-m3')
            self.dimension = 1024  # BGE-M3 dimension

            logger.info(f"✅ IONOS embeddings initialized: {self.model_name}")
        except Exception as e:
            logger.error(f"Failed to initialize IONOS embeddings: {e}")
            raise

    def embed(self, text: str) -> np.ndarray:
        """Generate embedding for a single text"""
        if self.provider == 'huggingface':
            return self._embed_huggingface(text)
        elif self.provider == 'openai':
            return self._embed_openai(text)
        elif self.provider == 'ionos':
            return self._embed_ionos(text)
        else:
            raise ValueError(f"Unsupported provider: {self.provider}")

    def embed_batch(self, texts: List[str]) -> List[np.ndarray]:
        """Generate embeddings for multiple texts"""
        if self.provider == 'huggingface':
            return self._embed_batch_huggingface(texts)
        elif self.provider == 'openai':
            return self._embed_batch_openai(texts)
        elif self.provider == 'ionos':
            return self._embed_batch_ionos(texts)
        else:
            raise ValueError(f"Unsupported provider: {self.provider}")

    def _embed_huggingface(self, text: str) -> np.ndarray:
        """Generate embedding using HuggingFace"""
        embedding = self.model.encode(text, convert_to_numpy=True)
        return embedding.astype(np.float32)

    def _embed_batch_huggingface(self, texts: List[str]) -> List[np.ndarray]:
        """Generate embeddings for multiple texts using HuggingFace"""
        embeddings = self.model.encode(texts, convert_to_numpy=True, show_progress_bar=True)
        return [emb.astype(np.float32) for emb in embeddings]

    def _embed_openai(self, text: str) -> np.ndarray:
        """Generate embedding using OpenAI"""
        response = self.client.embeddings.create(
            model=self.model_name,
            input=text
        )
        embedding = np.array(response.data[0].embedding, dtype=np.float32)
        return embedding

    def _embed_batch_openai(self, texts: List[str]) -> List[np.ndarray]:
        """Generate embeddings for multiple texts using OpenAI"""
        # OpenAI supports batch embedding
        response = self.client.embeddings.create(
            model=self.model_name,
            input=texts
        )
        embeddings = [np.array(item.embedding, dtype=np.float32) for item in response.data]
        return embeddings

    def _embed_ionos(self, text: str) -> np.ndarray:
        """Generate embedding using IONOS"""
        import json

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        # Sanitize input text
        sanitized_text = text.strip()
        sanitized_text = sanitized_text.encode('utf-8', errors='ignore').decode('utf-8')
        sanitized_text = ''.join(char for char in sanitized_text if char.isprintable() or char in '\n\t')
        if len(sanitized_text) > 6000:
            sanitized_text = sanitized_text[:6000]

        payload = {
            "model": self.model_name,
            "input": sanitized_text
        }

        try:
            response = self.client.post(
                self.api_url,
                headers=headers,
                data=json.dumps(payload, ensure_ascii=False),
                timeout=30
            )

            response.raise_for_status()
            result = response.json()

            embedding = np.array(result['data'][0]['embedding'], dtype=np.float32)
            return embedding
        except Exception as e:
            logger.error(f"IONOS embedding error: {e}")
            if hasattr(e, 'response') and e.response is not None:
                logger.error(f"Response status: {e.response.status_code}")
                logger.error(f"Response body: {e.response.text[:500]}")
            raise

    def _embed_batch_ionos(self, texts: List[str]) -> List[np.ndarray]:
        """Generate embeddings for multiple texts using IONOS"""
        import json

        # Filter and sanitize texts
        valid_texts = []
        valid_indices = []
        for i, text in enumerate(texts):
            if text and isinstance(text, str) and text.strip():
                # Sanitize text: remove excessive whitespace and limit length
                sanitized = ' '.join(text.strip().split())
                # Remove non-UTF-8 characters and control characters
                sanitized = sanitized.encode('utf-8', errors='ignore').decode('utf-8')
                # Remove control characters except newlines and tabs
                sanitized = ''.join(char for char in sanitized if char.isprintable() or char in '\n\t')
                # Limit to 6000 characters to avoid API issues
                if len(sanitized) > 6000:
                    sanitized = sanitized[:6000]
                # Skip if too short after sanitization
                if len(sanitized.strip()) < 3:
                    logger.warning(f"Skipping text at index {i}: too short after sanitization")
                    continue
                valid_texts.append(sanitized)
                valid_indices.append(i)
            else:
                logger.warning(f"Skipping empty or invalid text at index {i}")

        if not valid_texts:
            raise ValueError("No valid texts to embed")

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        # Process in smaller batches if the batch is too large
        max_batch_size = 10  # IONOS might have a limit on batch size
        all_embeddings = []

        for batch_start in range(0, len(valid_texts), max_batch_size):
            batch_end = min(batch_start + max_batch_size, len(valid_texts))
            batch_texts = valid_texts[batch_start:batch_end]

            payload = {
                "model": self.model_name,
                "input": batch_texts
            }

            try:
                logger.debug(f"Sending sub-batch {batch_start//max_batch_size + 1}: {len(batch_texts)} texts to IONOS API")
                response = self.client.post(
                    self.api_url,
                    headers=headers,
                    data=json.dumps(payload, ensure_ascii=False),
                    timeout=60
                )

                response.raise_for_status()
                result = response.json()

                batch_embeddings = [np.array(item['embedding'], dtype=np.float32) for item in result['data']]
                all_embeddings.extend(batch_embeddings)

            except Exception as e:
                logger.error(f"IONOS batch embedding error for sub-batch {batch_start//max_batch_size + 1}: {e}")
                if hasattr(e, 'response') and e.response is not None:
                    logger.error(f"Response status: {e.response.status_code}")
                    logger.error(f"Response body: {e.response.text[:500]}")
                    logger.error(f"Payload: {json.dumps(payload, ensure_ascii=False)[:500]}")
                raise

        # Create full results array including placeholders for skipped texts
        full_embeddings = []
        valid_idx = 0
        for i in range(len(texts)):
            if i in valid_indices:
                full_embeddings.append(all_embeddings[valid_idx])
                valid_idx += 1
            else:
                # Create zero vector for invalid texts
                full_embeddings.append(np.zeros(self.dimension, dtype=np.float32))

        return full_embeddings

    def get_dimension(self) -> int:
        """Get embedding dimension"""
        return self.dimension

    def get_stats(self) -> Dict[str, Any]:
        """Get service statistics"""
        return {
            'provider': self.provider,
            'model': getattr(self, 'model_name', str(self.model)),
            'dimension': self.dimension
        }


def create_embedding_service(config: Dict[str, Any]) -> EmbeddingService:
    """Factory function to create embedding service"""
    return EmbeddingService(config)
