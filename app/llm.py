"""
LLM Service for Response Generation
Uses OpenAI API or IONOS API to generate natural language responses
"""

import logging
from typing import List, Dict, Any

logger = logging.getLogger(__name__)


class LLMService:
    """LLM for generating responses with support for OpenAI and IONOS models"""

    def __init__(self, config: Dict[str, Any]):
        self.provider = config.get('provider', 'openai').lower()
        self.temperature = config.get('temperature', 0.7)

        # Initialize as None by default
        self.client = None
        self.model = None

        # OpenAI initialization
        if self.provider == 'openai':
            self._init_openai(config)
        # IONOS initialization
        elif self.provider == 'ionos':
            self._init_ionos(config)
        else:
            logger.warning(f"⚠️  Unsupported LLM provider: {self.provider} - using simple responses")

    def _init_openai(self, config: Dict[str, Any]):
        """Initialize OpenAI client"""
        try:
            import openai
            api_key = config.get('api_key', '')
            if not api_key or len(api_key) < 20:
                logger.warning("⚠️  Invalid or missing OpenAI API key - LLM will use simple responses")
                return

            # Try the new OpenAI client style (v1.0+)
            try:
                from openai import OpenAI as OpenAIClient
                # Initialize with only supported parameters for newer OpenAI SDK
                client_kwargs = {'api_key': api_key}
                
                # Only add supported optional parameters
                if 'base_url' in config:
                    client_kwargs['base_url'] = config['base_url']
                if 'organization' in config:
                    client_kwargs['organization'] = config['organization']
                if 'timeout' in config:
                    client_kwargs['timeout'] = config['timeout']
                
                self.client = OpenAIClient(**client_kwargs)
                self.model = config.get('model', 'gpt-4o-mini')
                logger.info(f"✅ OpenAI LLM initialized (OpenAI client): {self.model}")
                return
            except (ImportError, TypeError) as e:
                logger.debug(f"OpenAI v1+ client init failed: {e}")
                pass

            # Fall back to classic openai module usage (v0.x)
            try:
                openai.api_key = api_key
                # Only set supported classic openai module attributes
                if 'base_url' in config:
                    openai.api_base = config['base_url']
                if 'organization' in config:
                    openai.organization = config['organization']
                    
                self.client = openai
                self.model = config.get('model', 'gpt-4o-mini')
                logger.info(f"✅ OpenAI LLM initialized (module): {self.model}")
                return
            except Exception as e:
                logger.warning(f"⚠️  OpenAI module-style initialization failed: {e}")
                return
        except ImportError:
            logger.warning("⚠️  openai not installed - LLM will use simple responses")
        except Exception as e:
            logger.warning(f"⚠️  OpenAI LLM initialization failed: {e} - using simple responses")

    def _init_ionos(self, config: Dict[str, Any]):
        """Initialize IONOS client"""
        try:
            import requests

            api_key = config.get('api_key', '')
            api_url = config.get('api_url', '')

            if not api_key or not api_url:
                logger.warning("⚠️  Missing IONOS API key or URL - LLM will use simple responses")
                return

            # Test connection by making a small request
            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            }

            # Store configuration
            self.client = requests
            self.headers = headers
            self.api_url = api_url
            self.model = config.get('model', 'mixtral-8x7b')
            logger.info(f"✅ IONOS LLM initialized: {self.model}")
        except ImportError:
            logger.warning("⚠️  requests not installed - LLM will use simple responses")
        except Exception as e:
            logger.warning(f"⚠️  IONOS LLM initialization failed: {e} - using simple responses")

    def generate_response(self, query: str, context_documents: List[Dict[str, Any]]) -> str:
        """Generate a response based on query and retrieved documents"""

        if not context_documents:
            return "I couldn't find any relevant information to answer your question."

        # If no client, use simple response
        if not self.client:
            return self._generate_simple_response(query, context_documents)

        # Build context from top documents
        context_parts = []
        for i, doc in enumerate(context_documents[:5], 1):
            context_parts.append(
                f"Document {i} ({doc['app_name']}):\n{doc['content']}"
            )

        context = "\n\n".join(context_parts)

        # System prompt
        system_prompt = """You are a helpful assistant for EGroupware data. 
Answer questions based only on the provided context from EGroupware applications (contacts, calendar, tasks).
Be concise and friendly. If the context doesn't contain enough information, say so.
Always cite which documents you reference."""

        # User prompt
        user_prompt = f"""Context from EGroupware:

{context}

Question: {query}

Please provide a helpful answer based on the context above."""

        try:
            # Generate response based on provider
            if self.provider == 'openai':
                return self._generate_openai_response(system_prompt, user_prompt)
            elif self.provider == 'ionos':
                return self._generate_ionos_response(system_prompt, user_prompt)
            else:
                return self._generate_simple_response(query, context_documents)
        except Exception as e:
            logger.error(f"LLM generation failed: {e}")
            return self._generate_simple_response(query, context_documents)

    def _generate_openai_response(self, system_prompt: str, user_prompt: str) -> str:
        """Generate response using OpenAI"""
        # Support both the new OpenAI client and classic openai module
        try:
            # New OpenAI client: client.chat.completions.create(...)
            if hasattr(self.client, 'chat') and hasattr(self.client.chat, 'completions'):
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    temperature=self.temperature,
                    max_tokens=500
                )
            # Classic openai module: openai.ChatCompletion.create(...)
            elif hasattr(self.client, 'ChatCompletion'):
                response = self.client.ChatCompletion.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    temperature=self.temperature,
                    max_tokens=500
                )
            # As a last resort, try module-style chat completions if available
            elif hasattr(self.client, 'chat') and hasattr(self.client.chat, 'create'):
                response = self.client.chat.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    temperature=self.temperature,
                    max_tokens=500
                )
            else:
                raise RuntimeError("OpenAI client does not expose a supported chat interface")

            # Try multiple ways to extract the assistant text
            try:
                return response.choices[0].message.content
            except Exception:
                try:
                    return response['choices'][0]['message']['content']
                except Exception:
                    try:
                        return response.choices[0].text
                    except Exception:
                        return str(response)

        except Exception as e:
            logger.error(f"OpenAI response generation failed: {e}")
            raise

    def _generate_ionos_response(self, system_prompt: str, user_prompt: str) -> str:
        """Generate response using IONOS API"""
        import json

        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            "temperature": self.temperature,
            "max_tokens": 500
        }

        try:
            response = self.client.post(
                self.api_url,
                headers=self.headers,
                data=json.dumps(payload),
                timeout=30
            )

            response.raise_for_status()  # Raise exception for HTTP errors
            result = response.json()

            # Extract response based on IONOS API structure
            if "choices" in result and result["choices"] and "message" in result["choices"][0]:
                return result["choices"][0]["message"]["content"]
            else:
                logger.error(f"Unexpected IONOS API response structure: {result}")
                return "Sorry, I received an unexpected response format from the LLM service."

        except Exception as e:
            logger.error(f"IONOS API request failed: {e}")
            raise e

    def _generate_simple_response(self, query: str, context_documents: List[Dict[str, Any]]) -> str:
        """Generate a simple response without LLM APIs"""

        if not context_documents:
            return "No relevant information found."

        # Build simple response
        response_parts = [f"I found {len(context_documents)} relevant result(s):\n"]

        for i, doc in enumerate(context_documents[:3], 1):
            metadata = doc.get('metadata', {})
            app = doc['app_name']

            if app == 'addressbook':
                name = metadata.get('name', 'Contact')
                response_parts.append(f"{i}. Contact: {name}")
            elif app == 'calendar':
                title = metadata.get('title', 'Event')
                response_parts.append(f"{i}. Event: {title}")
            elif app == 'infolog':
                title = metadata.get('title', 'Task')
                response_parts.append(f"{i}. Task: {title}")
            else:
                response_parts.append(f"{i}. {doc['content'][:100]}...")

        if len(context_documents) > 3:
            response_parts.append(f"\n...and {len(context_documents) - 3} more results.")

        return "\n".join(response_parts)
