"""
RAG Service - Core RAG functionality
Combines EGroupware data fetching, embedding generation, and vector search
"""

import logging
from typing import List, Dict, Any, Optional
from datetime import datetime

logger = logging.getLogger(__name__)


class RAGService:
    """RAG service for EGroupware data"""

    def __init__(self, egroupware_client, embedding_service, database, llm_service, config: Optional[Dict[str, Any]] = None):
        self.egroupware = egroupware_client
        self.embeddings = embedding_service
        self.database = database
        self.llm = llm_service
        
        # Load chunking configuration
        chunking_config = config.get('chunking', {}) if config else {}
        self.embedding_batch_size = chunking_config.get('embedding_batch_size', 32)

    def _prepare_document(self, item: Dict[str, Any], app_name: str) -> Dict[str, Any]:
        """Convert EGroupware item to searchable document"""

        if app_name == 'addressbook':
            return self._prepare_contact(item)
        elif app_name == 'calendar':
            return self._prepare_event(item)
        elif app_name == 'infolog':
            return self._prepare_task(item)
        else:
            return None

    def _prepare_contact(self, contact: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare contact for indexing"""
        # Debug: Log contact structure
        logger.info(f"Contact keys: {list(contact.keys())}")
        
        # Extract relevant fields - Handle both old and new EGroupware formats
        name = None
        
        # Try new format first (fullName field)
        if 'fullName' in contact and contact['fullName']:
            name = contact['fullName']
        # Try name field
        elif 'name' in contact and contact['name']:
            if isinstance(contact['name'], list):
                # Handle NameComponent array format (old format)
                name_parts = []
                title_part = ''
                for component in contact['name']:
                    if isinstance(component, dict) and 'type' in component and 'value' in component:
                        if component['type'] == 'title':
                            title_part = component['value']
                        elif component['type'] == 'given':
                            name_parts.append(component['value'])
                        elif component['type'] == 'surname':
                            name_parts.append(component['value'])
                
                # Combine title and name parts
                if title_part and name_parts:
                    name = f"{title_part} {' '.join(name_parts)}"
                elif name_parts:
                    name = ' '.join(name_parts)
                elif title_part:
                    name = title_part
            else:
                # Direct string value
                name = str(contact['name'])
        # Try full name fields
        elif 'fn' in contact:
            name = contact['fn']
        # Try structured name fields (old format)
        elif 'n' in contact:
            if isinstance(contact['n'], dict):
                parts = []
                if 'given' in contact['n'] and contact['n']['given']:
                    parts.append(contact['n']['given'])
                if 'family' in contact['n'] and contact['n']['family']:
                    parts.append(contact['n']['family'])
                if parts:
                    name = ' '.join(parts)
            elif isinstance(contact['n'], str):
                name = contact['n']
        
        # Fallback
        if not name:
            name = 'Unknown Contact'
        
        # Handle organization field - new format uses 'organizations' array
        org = ''
        if 'organizations' in contact and contact['organizations']:
            if isinstance(contact['organizations'], list) and len(contact['organizations']) > 0:
                first_org = contact['organizations'][0]
                if isinstance(first_org, dict) and 'name' in first_org:
                    org = first_org['name']
                elif isinstance(first_org, str):
                    org = first_org
            elif isinstance(contact['organizations'], str):
                org = contact['organizations']
        elif 'org' in contact:
            if isinstance(contact['org'], dict) and 'name' in contact['org']:
                org = contact['org']['name']
            elif isinstance(contact['org'], str):
                org = contact['org']
        
        # Handle titles - new format uses 'titles' array
        title = ''
        if 'titles' in contact and contact['titles']:
            if isinstance(contact['titles'], list) and len(contact['titles']) > 0:
                first_title = contact['titles'][0]
                if isinstance(first_title, dict) and 'title' in first_title:
                    title = first_title['title']
                elif isinstance(first_title, str):
                    title = first_title
            elif isinstance(contact['titles'], str):
                title = contact['titles']
        elif 'title' in contact:
            title = contact['title']
        
        # Handle emails - new format uses 'emails' array
        email = ''
        if 'emails' in contact and contact['emails']:
            if isinstance(contact['emails'], list) and len(contact['emails']) > 0:
                first_email = contact['emails'][0]
                if isinstance(first_email, dict) and 'email' in first_email:
                    email = first_email['email']
                elif isinstance(first_email, str):
                    email = first_email
            elif isinstance(contact['emails'], str):
                email = contact['emails']
        elif 'email' in contact:
            if isinstance(contact['email'], dict) and 'value' in contact['email']:
                email = contact['email']['value']
            elif isinstance(contact['email'], str):
                email = contact['email']
            elif isinstance(contact['email'], list) and len(contact['email']) > 0:
                first_email = contact['email'][0]
                if isinstance(first_email, dict) and 'value' in first_email:
                    email = first_email['value']
                elif isinstance(first_email, str):
                    email = first_email
        
        # Handle phones - new format uses 'phones' array
        tel = ''
        if 'phones' in contact and contact['phones']:
            if isinstance(contact['phones'], list) and len(contact['phones']) > 0:
                first_phone = contact['phones'][0]
                if isinstance(first_phone, dict) and 'phone' in first_phone:
                    tel = first_phone['phone']
                elif isinstance(first_phone, str):
                    tel = first_phone
            elif isinstance(contact['phones'], str):
                tel = contact['phones']
        elif 'tel' in contact:
            if isinstance(contact['tel'], dict) and 'value' in contact['tel']:
                tel = contact['tel']['value']
            elif isinstance(contact['tel'], str):
                tel = contact['tel']
            elif isinstance(contact['tel'], list) and len(contact['tel']) > 0:
                first_tel = contact['tel'][0]
                if isinstance(first_tel, dict) and 'value' in first_tel:
                    tel = first_tel['value']
                elif isinstance(first_tel, str):
                    tel = first_tel
        
        # Handle notes
        note = ''
        if 'notes' in contact and contact['notes']:
            if isinstance(contact['notes'], list) and len(contact['notes']) > 0:
                note = contact['notes'][0]
            elif isinstance(contact['notes'], str):
                note = contact['notes']
        elif 'note' in contact:
            note = contact['note']
        
        # Handle addresses - new format uses 'addresses' array
        adr = ''
        if 'addresses' in contact and contact['addresses']:
            if isinstance(contact['addresses'], list) and len(contact['addresses']) > 0:
                first_addr = contact['addresses'][0]
                if isinstance(first_addr, dict):
                    adr_parts = []
                    for field in ['street', 'locality', 'region', 'postal', 'country']:
                        if field in first_addr and first_addr[field]:
                            adr_parts.append(str(first_addr[field]))
                    adr = ', '.join(adr_parts)
                elif isinstance(first_addr, str):
                    adr = first_addr
        elif 'adr' in contact:
            if isinstance(contact['adr'], dict):
                adr_parts = []
                for field in ['street', 'locality', 'region', 'postal', 'country']:
                    if field in contact['adr'] and contact['adr'][field]:
                        adr_parts.append(str(contact['adr'][field]))
                adr = ', '.join(adr_parts)
            elif isinstance(contact['adr'], str):
                adr = contact['adr']

        # Build searchable content
        content_parts = [f"Contact: {name}"]

        if org:
            content_parts.append(f"Organization: {org}")
        if title:
            content_parts.append(f"Title: {title}")
        if email:
            content_parts.append(f"Email: {email}")
        if tel:
            content_parts.append(f"Phone: {tel}")
        if adr:
            content_parts.append(f"Address: {adr}")
        if note:
            content_parts.append(f"Notes: {note}")

        content = "\n".join(content_parts)

        # Extract doc_id
        doc_id = contact.get('_id', contact.get('id', contact.get('uid', '')))

        return {
            'doc_id': str(doc_id),
            'content': content,
            'metadata': {
                'name': name,
                'org': org,
                'email': email,
                'title': title
            }
        }

    def _prepare_event(self, event: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare calendar event for indexing"""
        # Debug: Log calendar event structure
        logger.info(f"Calendar event keys: {list(event.keys())}")
        
        # Extract calendar event information, handling different possible structures
        # Try multiple field names for summary/title
        summary = None
        # EGroupware calendar events might use different field names for the title/summary
        for field in ['summary', 'title', 'subject', 'name']:
            if field in event and event[field]:
                summary = event[field]
                break
                
        # If still no summary, check if there's a structured field
        if not summary and 'vevent' in event:
            # Some EGroupware versions nest data in a vevent structure
            vevent = event['vevent']
            if isinstance(vevent, dict):
                for field in ['summary', 'title', 'subject', 'name']:
                    if field in vevent and vevent[field]:
                        summary = vevent[field]
                        break
        
        # Final fallback
        if not summary:
            summary = 'Untitled Event'
            
        # Get description with proper fallbacks
        description = ''
        for field in ['description', 'desc', 'note', 'notes']:
            if field in event and event[field]:
                description = event[field]
                break
                
        # Get location with proper fallbacks
        location = ''
        for field in ['location', 'venue', 'place']:
            if field in event and event[field]:
                location = event[field]
                break
                
        # Handle different date formats
        start = ''
        if 'dtstart' in event:
            start = event['dtstart']
        elif 'start' in event:
            start = event['start']
        elif 'startdate' in event:
            start = event['startdate']
            
        end = ''
        if 'dtend' in event:
            end = event['dtend']
        elif 'end' in event:
            end = event['end']
        elif 'enddate' in event:
            end = event['enddate']

        content_parts = [f"Event: {summary}"]

        if start:
            content_parts.append(f"Start: {start}")
        if end:
            content_parts.append(f"End: {end}")
        if location:
            content_parts.append(f"Location: {location}")
        if description:
            content_parts.append(f"Description: {description}")

        content = "\n".join(content_parts)

        doc_id = event.get('_id', event.get('id', event.get('uid', '')))

        return {
            'doc_id': str(doc_id),
            'content': content,
            'metadata': {
                'title': summary,
                'location': location,
                'start': start,
                'end': end
            }
        }

    def _prepare_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare InfoLog task for indexing"""
        # Debug: Log task structure
        logger.info(f"Task keys: {list(task.keys())}")
        
        # Extract task information, handling different possible structures
        # Try multiple field names for summary/title
        summary = None
        for field in ['summary', 'title', 'subject', 'name']:
            if field in task and task[field]:
                summary = task[field]
                break
                
        # Final fallback
        if not summary:
            summary = 'Untitled Task'
            
        # Get description with proper fallbacks
        description = ''
        for field in ['description', 'desc', 'note', 'notes', 'content']:
            if field in task and task[field]:
                description = task[field]
                break
                
        # Get status with proper fallbacks
        status = ''
        for field in ['status', 'state', 'completion']:
            if field in task and task[field]:
                status = task[field]
                break
                
        # Get priority with proper fallbacks
        priority = ''
        for field in ['priority', 'importance']:
            if field in task and task[field]:
                priority = task[field]
                break

        content_parts = [f"Task: {summary}"]

        if status:
            content_parts.append(f"Status: {status}")
        if priority:
            content_parts.append(f"Priority: {priority}")
        if description:
            content_parts.append(f"Description: {description}")

        content = "\n".join(content_parts)

        doc_id = task.get('_id', task.get('id', task.get('uid', '')))

        return {
            'doc_id': str(doc_id),
            'content': content,
            'metadata': {
                'title': summary,
                'status': status,
                'priority': priority
            }
        }

    def index_app(self, app_name: str) -> Dict[str, Any]:
        """Index data from a specific EGroupware application"""
        try:
            logger.info(f"Starting indexing for {app_name}")

            # Fetch data from EGroupware using correct method names
            if app_name == 'addressbook':
                raw_items = self.egroupware.get_contacts()
            elif app_name == 'calendar':
                raw_items = self.egroupware.get_calendar_events()
            elif app_name == 'infolog':
                raw_items = self.egroupware.get_infolog_entries()
            else:
                raise ValueError(f"Unsupported app: {app_name}")

            logger.info(f"Fetched {len(raw_items)} items from {app_name}")

            if not raw_items:
                return {
                    'success': True,
                    'app': app_name,
                    'indexed': 0,
                    'message': f"No data found in {app_name}"
                }

            # Prepare documents
            documents = []
            for item in raw_items:
                doc = self._prepare_document(item, app_name)
                if doc and doc['content'].strip():
                    documents.append({
                        'app_name': app_name,
                        **doc
                    })

            logger.info(f"Prepared {len(documents)} documents for indexing")

            if not documents:
                return {
                    'success': True,
                    'app': app_name,
                    'indexed': 0,
                    'message': f"No valid documents to index from {app_name}"
                }

            # Generate embeddings in batches
            batch_size = self.embedding_batch_size
            all_documents_with_embeddings = []

            for i in range(0, len(documents), batch_size):
                batch = documents[i:i + batch_size]
                texts = [doc['content'] for doc in batch]

                logger.info(f"Generating embeddings for batch {i//batch_size + 1}/{(len(documents)-1)//batch_size + 1}")
                embeddings = self.embeddings.embed_batch(texts)

                for doc, embedding in zip(batch, embeddings):
                    doc['embedding'] = embedding.tolist()
                    all_documents_with_embeddings.append(doc)

            # Insert into database
            self.database.insert_documents_batch(all_documents_with_embeddings)

            logger.info(f"✅ Successfully indexed {len(all_documents_with_embeddings)} documents from {app_name}")

            return {
                'success': True,
                'app': app_name,
                'indexed': len(all_documents_with_embeddings),
                'message': f"Successfully indexed {len(all_documents_with_embeddings)} documents from {app_name}"
            }

        except Exception as e:
            logger.error(f"Failed to index {app_name}: {e}", exc_info=True)
            return {
                'success': False,
                'app': app_name,
                'indexed': 0,
                'error': str(e)
            }

    def index_all(self) -> Dict[str, Any]:
        """Index all EGroupware applications"""
        results = []
        total_indexed = 0

        for app in ['addressbook', 'calendar', 'infolog']:
            result = self.index_app(app)
            results.append(result)
            if result['success']:
                total_indexed += result['indexed']

        return {
            'success': True,
            'total_indexed': total_indexed,
            'results': results
        }

    def search(self, query: str, app_filter: Optional[str] = None, top_k: int = 5, fast_mode: bool = False) -> Dict[str, Any]:
        """Search EGroupware data using natural language query
        
        Args:
            query: Search query
            app_filter: Filter by application name
            top_k: Number of results to return
            fast_mode: If True, skip LLM generation for instant results
        """
        try:
            mode_str = "FAST MODE" if fast_mode else "NORMAL"
            logger.info(f"RAG search started ({mode_str}): query='{query}', app_filter={app_filter}, top_k={top_k}")
            
            # Generate query embedding
            logger.info(f"Generating embedding for query: '{query}'")
            query_embedding = self.embeddings.embed(query)
            logger.info(f"Generated embedding with shape: {query_embedding.shape}")
            
            # Search in database
            logger.info(f"Searching database with embedding of dimension {len(query_embedding.tolist())}")
            results = self.database.search(
                query_embedding=query_embedding.tolist(),
                query_text=query,  # Pass original query for hybrid scoring
                app_filter=app_filter,
                top_k=top_k
            )
            
            logger.info(f"Database search returned {len(results)} results")
            
            # Log a few results for debugging
            if results:
                for i, result in enumerate(results[:3]):
                    logger.info(f"Result {i+1}: app={result['app_name']}, "
                               f"similarity={result['similarity']:.4f}, "
                               f"content_length={len(result['content'])}")
            else:
                logger.warning("No results returned from database search")

            # Generate LLM response if available and not in fast mode
            response = None
            if results and self.llm and not fast_mode:
                try:
                    logger.info("Generating LLM response")
                    response = self.llm.generate_response(query, results, use_simple=False)
                    logger.info(f"LLM response generated: {len(response)} chars")
                except Exception as e:
                    logger.warning(f"LLM generation failed: {e}")
            elif results and fast_mode:
                # In fast mode, always use simple response
                logger.info("Fast mode: using simple response (no LLM)")
                response = self.llm.generate_response(query, results, use_simple=True) if self.llm else None

            return {
                'success': True,
                'query': query,
                'results': results,
                'response': response,
                'count': len(results),
                'fast_mode': fast_mode
            }

        except Exception as e:
            logger.error(f"Search failed: {e}", exc_info=True)
            return {
                'success': False,
                'error': str(e)
            }

    def get_stats(self) -> Dict[str, Any]:
        """Get RAG system statistics"""
        try:
            db_stats = self.database.get_stats()
            embedding_stats = self.embeddings.get_stats()

            return {
                'success': True,
                'database': db_stats,
                'embeddings': embedding_stats
            }
        except Exception as e:
            logger.error(f"Failed to get stats: {e}")
            return {
                'success': False,
                'error': str(e)
            }

    def reset_data(self, app_name: Optional[str] = None) -> Dict[str, Any]:
        """Reset indexed data"""
        try:
            count = self.database.delete_user_documents(app_name)

            return {
                'success': True,
                'deleted': count,
                'message': f"Deleted {count} documents"
            }
        except Exception as e:
            logger.error(f"Failed to reset data: {e}")
            return {
                'success': False,
                'error': str(e)
            }
