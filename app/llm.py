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
        # Reduced max_tokens for faster responses (configurable)
        self.max_tokens = config.get('max_tokens', 800)

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
            # Ensure URL ends with /chat/completions for LLM endpoint
            if not api_url.endswith('/chat/completions'):
                api_url = api_url.rstrip('/') + '/chat/completions'
            self.api_url = api_url
            self.model = config.get('model', 'mixtral-8x7b')
            self.timeout = config.get('timeout', 15)  # Configurable timeout
            logger.info(f"✅ IONOS LLM initialized: {self.model}")
        except ImportError:
            logger.warning("⚠️  requests not installed - LLM will use simple responses")
        except Exception as e:
            logger.warning(f"⚠️  IONOS LLM initialization failed: {e} - using simple responses")

    def generate_response(self, query: str, context_documents: List[Dict[str, Any]], use_simple: bool = False) -> str:
        """Generate a response based on query and retrieved documents
        
        Args:
            query: User's search query
            context_documents: Retrieved documents for context
            use_simple: If True, skip LLM and use fast simple response
        """

        if not context_documents:
            return "I couldn't find any relevant information to answer your question."

        # Fast mode: use simple response (no LLM call)
        if use_simple or not self.client:
            return self._generate_simple_response(query, context_documents)

        # Build context from top documents (limit to 3 for speed)
        context_parts = []
        for i, doc in enumerate(context_documents[:3], 1):
            metadata = doc.get('metadata', {})
            app_type = doc['app_name']
            
            # Truncate content if too long (keep first 300 chars for speed)
            content = doc['content']
            if len(content) > 300:
                content = content[:300] + "..."
            
            # Add structured context with metadata
            context_parts.append(
                f"[Source {i} - {app_type.upper()}]\n{content}"
            )

        context = "\n\n".join(context_parts)

        # Get current date for temporal awareness
        from datetime import datetime
        current_date = datetime.now().strftime('%Y-%m-%d')
        current_year = datetime.now().year

        # Concise system prompt for faster processing
        system_prompt = f"""You are an EGroupware assistant. Today's date is {current_date}.

When answering questions:
- Be concise and direct
- Use **bold** for key information
- For date queries (this week, today, upcoming, etc.), filter results by date
- If events are from past years (before {current_year}), clearly state they are historical/old
- List multiple items clearly
- Cite sources"""

        # Shorter user prompt for speed
        user_prompt = f"""Question: {query}

Context:
{context}

Answer briefly and clearly:"""

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
                    max_tokens=self.max_tokens
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
                    max_tokens=self.max_tokens
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
                    max_tokens=self.max_tokens
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
            "max_tokens": self.max_tokens
        }

        try:
            response = self.client.post(
                self.api_url,
                headers=self.headers,
                data=json.dumps(payload),
                timeout=self.timeout
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
        from datetime import datetime

        if not context_documents:
            return "No relevant information found."

        # Check if query is about current/future dates
        query_lower = query.lower()
        temporal_keywords = [
            'this week', 'today', 'tomorrow', 'upcoming',
            'next week', 'this month', 'now', 'current'
        ]
        is_temporal_query = any(kw in query_lower for kw in temporal_keywords)
        current_year = datetime.now().year

        # Build comprehensive simple response
        response_parts = [f"**Found {len(context_documents)} relevant result(s):**\n"]

        for i, doc in enumerate(context_documents[:10], 1):
            metadata = doc.get('metadata', {})
            app = doc['app_name']
            similarity = doc.get('similarity', 0)

            response_parts.append(f"\n**{i}. {app.upper()}** (Match: {similarity*100:.1f}%)")
            
            if app == 'addressbook':
                name = metadata.get('name', 'Contact')
                org = metadata.get('org', '')
                email = metadata.get('email', '')
                title = metadata.get('title', '')
                
                response_parts.append(f"   • **Name:** {name}")
                if org:
                    response_parts.append(f"   • **Organization:** {org}")
                if title:
                    response_parts.append(f"   • **Title:** {title}")
                if email:
                    response_parts.append(f"   • **Email:** {email}")
                    
            elif app == 'calendar':
                event_title = metadata.get('title', 'Event')
                location = metadata.get('location', '')
                start = metadata.get('start', '')
                end = metadata.get('end', '')
                
                response_parts.append(f"   • **Event:** {event_title}")
                if start:
                    # Parse and format the date nicely
                    try:
                        # Handle various date formats
                        if 'T' in str(start):
                            dt = datetime.fromisoformat(
                                str(start).replace('Z', '+00:00')
                            )
                        else:
                            dt = datetime.fromisoformat(str(start))
                        formatted_date = dt.strftime('%Y-%m-%d at %H:%M')
                        
                        # Warn if event is old and query is temporal
                        event_year = dt.year
                        if is_temporal_query and event_year < current_year:
                            formatted_date += f" ⚠️ (OLD - from {event_year})"
                        
                        response_parts.append(f"   • **Start:** {formatted_date}")
                    except Exception:
                        response_parts.append(f"   • **Start:** {start}")
                if end:
                    try:
                        if 'T' in str(end):
                            dt = datetime.fromisoformat(
                                str(end).replace('Z', '+00:00')
                            )
                        else:
                            dt = datetime.fromisoformat(str(end))
                        formatted_date = dt.strftime('%Y-%m-%d at %H:%M')
                        response_parts.append(f"   • **End:** {formatted_date}")
                    except Exception:
                        response_parts.append(f"   • **End:** {end}")
                if location:
                    response_parts.append(f"   • **Location:** {location}")
                    
            elif app == 'infolog':
                task_title = metadata.get('title', 'Task')
                status = metadata.get('status', '')
                priority = metadata.get('priority', '')
                
                response_parts.append(f"   • **Task:** {task_title}")
                if status:
                    response_parts.append(f"   • **Status:** {status}")
                if priority:
                    response_parts.append(f"   • **Priority:** {priority}")
            else:
                # For unknown app types, show content preview
                content_preview = doc['content'][:200]
                if len(doc['content']) > 200:
                    content_preview += "..."
                response_parts.append(f"   {content_preview}")

        if len(context_documents) > 10:
            response_parts.append(f"\n_...and {len(context_documents) - 10} more results not shown._")

        # Add warning if temporal query returned only old calendar events
        if is_temporal_query:
            calendar_docs = [d for d in context_documents[:10] if d['app_name'] == 'calendar']
            if calendar_docs:
                all_old = True
                for doc in calendar_docs:
                    start = doc.get('metadata', {}).get('start', '')
                    if start:
                        try:
                            if 'T' in str(start):
                                dt = datetime.fromisoformat(str(start).replace('Z', '+00:00'))
                                if dt.year >= current_year:
                                    all_old = False
                                    break
                        except Exception:
                            pass
                
                if all_old:
                    response_parts.append(
                        f"\n⚠️ **Note:** All calendar events found are from previous years. "
                        f"No events found for the current timeframe ({current_year})."
                    )

        return "\n".join(response_parts)
