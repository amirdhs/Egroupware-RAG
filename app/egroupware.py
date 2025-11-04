"""
EGroupware REST API Client
Fetches data from EGroupware applications using proper pagination
"""

import requests
import logging
import time
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class EGroupwareClient:
    """EGroupware REST API client with proper pagination support"""

    def __init__(self, config: Dict[str, Any]):
        self.base_url = config['base_url'].rstrip('/')
        self.username = config['username']
        self.password = config['password']
        self.timeout = config.get('timeout', 30)
        self.api_fetch_size = config.get('api_fetch_size', 500)  # Configurable chunk size

        self.session = requests.Session()
        self.session.auth = (self.username, self.password)
        self.session.headers.update({
            'Accept': 'application/json',
            'Content-Type': 'application/json'
        })

    def _make_paginated_request(
            self, 
            endpoint: str, 
            max_results_per_chunk: Optional[int] = None,
            since_date: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Make paginated API requests to fetch ALL data using sync-tokens
        Based on EGroupware API documentation for proper data fetching

        Args:
            endpoint: API endpoint to fetch from
            max_results_per_chunk: Maximum number of results per page (uses config default if None)
            since_date: ISO formatted date to fetch only items modified since this date
        """
        if max_results_per_chunk is None:
            max_results_per_chunk = self.api_fetch_size
            
        url = f"{self.base_url}/{endpoint.lstrip('/')}"
        all_items = []
        sync_token = ""  # Start with empty sync-token to get all data

        try:
            while True:
                # Build request parameters for pagination
                params = {
                    'sync-token': sync_token,
                    'nresults': max_results_per_chunk
                }

                # Add since parameter for incremental fetching if provided
                if since_date:
                    # EGroupware uses standard HTTP date format for filtering
                    params['since'] = since_date
                    logger.info(f"Fetching only data modified since: {since_date}")

                logger.info(f"Fetching data from {url} (chunk size: {max_results_per_chunk})")
                response = self.session.get(url, params=params, timeout=self.timeout)
                response.raise_for_status()

                if not response.content:
                    break

                data = response.json()

                # Process responses
                if "responses" in data:
                    for url_path, item_data in data["responses"].items():
                        if item_data and isinstance(item_data, dict):
                            # Add the URL path for reference
                            item_data["_url"] = url_path
                            # Extract ID from URL path if possible
                            if "/" in url_path:
                                item_data["_id"] = url_path.split("/")[-1]
                            all_items.append(item_data)

                # Check if there are more results
                if not data.get("more-results", False):
                    break

                # Update sync-token for next chunk
                sync_token = data.get("sync-token", "")
                if not sync_token:
                    break

                logger.info(f"Fetched {len(data.get('responses', {}))} items, continuing with next chunk...")

        except Exception as e:
            logger.error(f"API request failed for {endpoint}: {e}")
            raise

        return all_items

    def _request_single(self, endpoint: str) -> Dict[str, Any]:
        """Make API request for single resource"""
        url = f"{self.base_url}/{endpoint.lstrip('/')}"

        try:
            response = self.session.get(url, timeout=self.timeout)
            response.raise_for_status()
            return response.json() if response.content else {}
        except Exception as e:
            logger.error(f"Single API request failed: {e}")
            raise

    def _discover_addressbook_collections(self) -> List[str]:
        """
        Discover all available addressbook collections including group-based ones
        """
        collections = []

        # Add the default addressbook collection
        collections.append("/addressbook/")

        # Try to discover additional addressbook collections
        # EGroupware may have group-specific addressbook collections
        try:
            # First, try to get all available collections via PROPFIND
            url = f"{self.base_url}/"
            response = self.session.request(
                'PROPFIND',
                url,
                headers={
                    'Accept': 'application/xml',
                    'Depth': '1',
                    'Content-Type': 'application/xml'
                },
                timeout=self.timeout
            )

            if response.status_code == 200:
                import re
                # Extract all addressbook hrefs from the XML response
                addressbook_hrefs = re.findall(r'<D:href>([^<]*addressbook[^<]*)</D:href>', response.text)
                for href in addressbook_hrefs:
                    # Extract the collection path relative to base_url
                    if href.startswith('/'):
                        # Convert absolute path to relative path
                        path_parts = href.split('/')
                        if 'addressbook' in path_parts:
                            idx = path_parts.index('addressbook')
                            collection_path = '/' + '/'.join(path_parts[idx:])
                            if collection_path not in collections:
                                collections.append(collection_path)

            # Also try some common group-based addressbook paths
            common_paths = [
                "/addressbook-shared/",
                "/addressbook-global/",
                "/addressbook-public/",
                "/shared/addressbook/",
                "/public/addressbook/"
            ]

            for path in common_paths:
                if path not in collections:
                    # Test if this collection exists by making a small request
                    try:
                        test_url = f"{self.base_url}{path}"
                        test_response = self.session.get(
                            test_url,
                            params={'nresults': 1},
                            timeout=10
                        )
                        if test_response.status_code == 200:
                            collections.append(path)
                            logger.info(f"Found additional addressbook collection: {path}")
                    except:
                        pass

        except Exception as e:
            logger.warning(f"Collection discovery failed, using default only: {e}")

        logger.info(f"Discovered {len(collections)} addressbook collections: {collections}")
        return collections

    def _discover_group_addressbooks(self) -> List[str]:
        """
        Discover group-specific addressbook collections using EGroupware-specific patterns
        EGroupware often organizes contacts by group ownership/ACL
        """
        group_collections = []

        # EGroupware group-based addressbook patterns
        # Groups in EGroupware are often accessible via different URL patterns
        try:
            logger.info("Starting comprehensive group addressbook discovery...")

            # Pattern 1: /addressbook-<group_id>/ (Test more group IDs)
            logger.info("Testing pattern: /addressbook-<group_id>/")
            for group_id in range(1, 50):  # Test more group IDs (1-50)
                path = f"/addressbook-{group_id}/"
                try:
                    test_url = f"{self.base_url}{path}"
                    test_response = self.session.get(
                        test_url,
                        params={'nresults': 1},
                        timeout=5
                    )
                    if test_response.status_code == 200:
                        try:
                            data = test_response.json()
                            if data.get('responses'):
                                group_collections.append(path)
                                logger.info(f"✅ Found group addressbook collection: {path} (Status: {test_response.status_code})")
                        except:
                            # Even if JSON parsing fails, if we get 200 it might be valid
                            if test_response.content:
                                group_collections.append(path)
                                logger.info(f"✅ Found group addressbook collection: {path} (Non-JSON response)")
                except Exception as e:
                    logger.debug(f"Group {group_id} failed: {e}")
                    continue

            # Pattern 2: Try accessing shared/group addressbooks with more variants
            logger.info("Testing shared/group patterns...")
            shared_patterns = [
                "/shared/",
                "/groups/",
                "/addressbook/shared/",
                "/addressbook/groups/",
                "/addressbook/public/",
                "/public/",
                "/global/",
                "/addressbook/global/",
                # EGroupware specific patterns
                "/addressbook/Business/",
                "/addressbook/business/",
                "/Business/",
                "/business/",
                "/Gruppe/",
                "/gruppe/",
                # Try with group names that might exist
                "/addressbook/Gruppe%20Business%20Contacts/",
                "/Gruppe%20Business%20Contacts/",
            ]

            for pattern in shared_patterns:
                try:
                    test_url = f"{self.base_url}{pattern}"
                    test_response = self.session.get(test_url, timeout=5)
                    logger.debug(f"Testing {pattern}: Status {test_response.status_code}")

                    if test_response.status_code == 200:
                        # Check if it contains addressbook data
                        try:
                            data = test_response.json()
                            if data.get('responses'):
                                group_collections.append(pattern)
                                logger.info(f"✅ Found shared addressbook collection: {pattern}")
                        except:
                            # Check if it's a valid response even without JSON
                            if test_response.content and len(test_response.content) > 100:
                                group_collections.append(pattern)
                                logger.info(f"✅ Found shared addressbook collection: {pattern} (Non-JSON)")
                except Exception as e:
                    logger.debug(f"Shared pattern {pattern} failed: {e}")
                    continue

            # Pattern 3: Try to discover via PROPFIND on the base URL to see all available collections
            logger.info("Attempting PROPFIND discovery for all collections...")
            try:
                url = f"{self.base_url}/"
                response = self.session.request(
                    'PROPFIND',
                    url,
                    headers={
                        'Accept': 'application/xml',
                        'Depth': '1',
                        'Content-Type': 'application/xml'
                    },
                    timeout=self.timeout
                )

                if response.status_code == 207:  # Multi-Status response
                    logger.info("PROPFIND successful, parsing collections...")
                    import re

                    # Extract all href entries
                    all_hrefs = re.findall(r'<D:href>([^<]+)</D:href>', response.text)
                    logger.info(f"Found {len(all_hrefs)} href entries in PROPFIND response")

                    for href in all_hrefs:
                        logger.debug(f"Found href: {href}")
                        # Look for addressbook-related paths
                        if 'addressbook' in href.lower() or 'contacts' in href.lower():
                            # Extract the collection path
                            if href.startswith('/'):
                                # Convert to relative path
                                path_parts = href.split('/')
                                if len(path_parts) > 3:  # Has enough parts
                                    # Find the collection part after username
                                    try:
                                        username_idx = -1
                                        for i, part in enumerate(path_parts):
                                            if part == self.username:
                                                username_idx = i
                                                break

                                        if username_idx > 0 and username_idx < len(path_parts) - 1:
                                            collection_parts = path_parts[username_idx + 1:]
                                            collection_path = '/' + '/'.join(collection_parts)
                                            if collection_path not in group_collections and collection_path != '/addressbook/':
                                                group_collections.append(collection_path)
                                                logger.info(f"✅ Discovered collection via PROPFIND: {collection_path}")
                                    except Exception as parse_error:
                                        logger.debug(f"Failed to parse href {href}: {parse_error}")
                                        pass
                else:
                    logger.warning(f"PROPFIND failed with status: {response.status_code}")

            except Exception as propfind_error:
                logger.warning(f"PROPFIND discovery failed: {propfind_error}")

        except Exception as e:
            logger.warning(f"Group addressbook discovery failed: {e}")

        logger.info(f"Group discovery complete. Found {len(group_collections)} additional collections: {group_collections}")
        return group_collections

    def _get_all_contacts_comprehensive(self, since_date: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Comprehensive method to fetch ALL contacts from all possible sources
        Including default, discovered, and group-based collections

        Args:
            since_date: ISO formatted date to fetch only items modified since this date
        """
        all_contacts = []
        all_collections = []

        # Get standard discovered collections
        standard_collections = self._discover_addressbook_collections()
        all_collections.extend(standard_collections)

        # Get group-specific collections
        group_collections = self._discover_group_addressbooks()
        all_collections.extend(group_collections)

        # Remove duplicates
        unique_collections = list(set(all_collections))

        logger.info(f"Found {len(unique_collections)} total addressbook collections to check")

        if since_date:
            logger.info(f"Fetching only contacts modified since: {since_date}")

        for collection in unique_collections:
            try:
                logger.info(f"Fetching contacts from collection: {collection}")
                contacts = self._make_paginated_request(collection, since_date=since_date)

                # Filter and process contacts
                individual_contacts = []
                for contact in contacts:
                    if contact.get("@type") == "Card":  # Individual contact
                        # Debug: Print contact keys and example values
                        if len(individual_contacts) == 0:
                            logger.info(f"Contact keys: {list(contact.keys())}")
                            logger.info(f"Contact FN: {contact.get('fn', 'N/A')}, N: {contact.get('n', 'N/A')}")
                            
                        # Add collection info to the contact
                        contact["_collection"] = collection
                        individual_contacts.append(contact)
                    elif contact.get("@type") == "CardGroup":  # Distribution list
                        logger.debug(f"Skipping CardGroup: {contact.get('name', 'Unknown')}")

                if individual_contacts:
                    logger.info(f"✅ Fetched {len(individual_contacts)} contacts from {collection}")
                    all_contacts.extend(individual_contacts)
                else:
                    logger.debug(f"No contacts found in collection: {collection}")

            except Exception as e:
                logger.warning(f"Failed to fetch contacts from collection {collection}: {e}")
                continue

        # Remove duplicates based on UID
        unique_contacts = {}
        for contact in all_contacts:
            uid = contact.get('uid', contact.get('_id', ''))
            if uid and uid not in unique_contacts:
                unique_contacts[uid] = contact
            elif uid in unique_contacts:
                # Keep the contact from the most specific collection
                # Group collections usually have more specific access rights
                if 'group' in contact.get('_collection', '').lower() or 'shared' in contact.get('_collection', '').lower():
                    unique_contacts[uid] = contact

        final_contacts = list(unique_contacts.values())

        if since_date:
            logger.info(f"✅ INCREMENTAL FETCH: {len(final_contacts)} unique contacts modified since {since_date}")
        else:
            logger.info(f"✅ COMPREHENSIVE FETCH: {len(final_contacts)} unique contacts from {len(unique_collections)} collections")

        # Log collection summary
        collection_summary = {}
        for contact in final_contacts:
            collection = contact.get('_collection', 'unknown')
            collection_summary[collection] = collection_summary.get(collection, 0) + 1

        for collection, count in collection_summary.items():
            logger.info(f"  - {collection}: {count} contacts")

        return final_contacts

    def get_contacts(self, since_date: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Fetch addressbook contacts from all available collections using proper pagination
        This includes contacts from group-based addressbook collections

        Args:
            since_date: Optional ISO formatted date to fetch only items modified since this date
        """
        if since_date:
            logger.info(f"Fetching contacts modified since {since_date} from all EGroupware addressbook collections...")
        else:
            logger.info("Fetching ALL contacts from all EGroupware addressbook collections (including groups)...")

        return self._get_all_contacts_comprehensive(since_date)

    def _get_all_calendar_events_comprehensive(self, since_date: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Comprehensive method to fetch ALL calendar events from all possible sources
        Including default, discovered, and group-based collections

        Args:
            since_date: ISO formatted date to fetch only items modified since this date
        """
        all_events = []
        all_collections = []

        # Get standard discovered collections
        standard_collections = self._discover_calendar_collections()
        all_collections.extend(standard_collections)

        # Get group-specific collections
        group_collections = self._discover_group_calendars()
        all_collections.extend(group_collections)

        # Remove duplicates
        unique_collections = list(set(all_collections))

        logger.info(f"Found {len(unique_collections)} total calendar collections to check")

        for collection in unique_collections:
            try:
                logger.info(f"Fetching calendar events from collection: {collection}")
                events = self._make_paginated_request(collection, since_date=since_date)

                # Filter and process events
                valid_events = []
                for event in events:
                    if event.get("@type") == "Event":  # Valid calendar event
                        # Add collection info to the event
                        event["_collection"] = collection
                        valid_events.append(event)
                    else:
                        logger.debug(f"Skipping non-Event item: {event.get('@type', 'Unknown')}")

                if valid_events:
                    logger.info(f"✅ Fetched {len(valid_events)} events from {collection}")
                    all_events.extend(valid_events)
                else:
                    logger.debug(f"No events found in collection: {collection}")

            except Exception as e:
                logger.warning(f"Failed to fetch events from collection {collection}: {e}")
                continue

        # Remove duplicates based on UID
        unique_events = {}
        for event in all_events:
            uid = event.get('uid', event.get('_id', ''))
            if uid and uid not in unique_events:
                unique_events[uid] = event
            elif uid in unique_events:
                # Keep the event from the most specific collection
                if 'group' in event.get('_collection', '').lower() or 'shared' in event.get('_collection', '').lower():
                    unique_events[uid] = event

        final_events = list(unique_events.values())
        logger.info(f"✅ COMPREHENSIVE CALENDAR FETCH: {len(final_events)} unique events from {len(unique_collections)} collections")

        # Log collection summary
        collection_summary = {}
        for event in final_events:
            collection = event.get('_collection', 'unknown')
            collection_summary[collection] = collection_summary.get(collection, 0) + 1

        for collection, count in collection_summary.items():
            logger.info(f"  - {collection}: {count} events")

        return final_events

    def get_calendar_events(self, since_date: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Fetch calendar events using simple GET request approach
        This fetches ALL events from the API and is more reliable than CalDAV REPORT

        Args:
            since_date: Optional ISO formatted date to fetch only items modified since this date
        """
        logger.info(f"Fetching {'modified' if since_date else 'ALL'} calendar events...")
        
        all_events = []
        
        # Get all calendar collections
        standard_collections = self._discover_calendar_collections()
        group_collections = self._discover_group_calendars()
        all_collections = list(set(standard_collections + group_collections))
        
        logger.info(f"Found {len(all_collections)} total calendar collections to check")
        
        for collection in all_collections:
            try:
                logger.info(f"Fetching calendar events from collection: {collection}")
                
                # Use simple GET request without date filters
                # The API will return all available events
                events = self._make_paginated_request(collection, since_date=since_date)
                
                # Filter and process events
                valid_events = []
                for event in events:
                    if event.get("@type") == "Event":  # Valid calendar event
                        # Add collection info to the event
                        event["_collection"] = collection
                        valid_events.append(event)
                    else:
                        logger.debug(f"Skipping non-Event item: {event.get('@type', 'Unknown')}")
                
                if valid_events:
                    logger.info(f"✅ Fetched {len(valid_events)} events from {collection}")
                    all_events.extend(valid_events)
                else:
                    logger.debug(f"No events found in collection: {collection}")
                    
            except Exception as e:
                logger.warning(f"Failed to fetch events from collection {collection}: {e}")
                continue
        
        # Remove duplicates based on UID
        unique_events = {}
        for event in all_events:
            uid = event.get('uid', event.get('_id', ''))
            if uid and uid not in unique_events:
                unique_events[uid] = event
            elif uid in unique_events:
                # Keep the event from the most specific collection
                if 'group' in event.get('_collection', '').lower() or 'shared' in event.get('_collection', '').lower():
                    unique_events[uid] = event
        
        final_events = list(unique_events.values())
        logger.info(f"✅ COMPREHENSIVE CALENDAR FETCH: {len(final_events)} unique events from {len(all_collections)} collections")
        
        # Log collection summary
        collection_summary = {}
        for event in final_events:
            collection = event.get('_collection', 'unknown')
            collection_summary[collection] = collection_summary.get(collection, 0) + 1
        
        for collection, count in collection_summary.items():
            logger.info(f"  - {collection}: {count} events")
        
        return final_events

    def _get_all_infolog_entries_comprehensive(self, since_date: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Comprehensive method to fetch ALL infolog entries from all possible sources
        Including default, discovered, and group-based collections

        Args:
            since_date: ISO formatted date to fetch only items modified since this date
        """
        all_entries = []
        all_collections = []

        # Get standard discovered collections
        standard_collections = self._discover_infolog_collections()
        all_collections.extend(standard_collections)

        # Get group-specific collections
        group_collections = self._discover_group_infologs()
        all_collections.extend(group_collections)

        # Remove duplicates
        unique_collections = list(set(all_collections))

        logger.info(f"Found {len(unique_collections)} total infolog collections to check")

        for collection in unique_collections:
            try:
                logger.info(f"Fetching infolog entries from collection: {collection}")
                entries = self._make_paginated_request(collection, since_date=since_date)

                # Filter and process entries
                valid_entries = []
                for entry in entries:
                    if entry.get("@type") == "Task":  # Valid infolog task
                        # Add collection info to the entry
                        entry["_collection"] = collection
                        valid_entries.append(entry)
                    else:
                        logger.debug(f"Skipping non-Task item: {entry.get('@type', 'Unknown')}")

                if valid_entries:
                    logger.info(f"✅ Fetched {len(valid_entries)} entries from {collection}")
                    all_entries.extend(valid_entries)
                else:
                    logger.debug(f"No entries found in collection: {collection}")

            except Exception as e:
                logger.warning(f"Failed to fetch entries from collection {collection}: {e}")
                continue

        # Remove duplicates based on UID
        unique_entries = {}
        for entry in all_entries:
            uid = entry.get('uid', entry.get('_id', ''))
            if uid and uid not in unique_entries:
                unique_entries[uid] = entry
            elif uid in unique_entries:
                # Keep the entry from the most specific collection
                if 'group' in entry.get('_collection', '').lower() or 'shared' in entry.get('_collection', '').lower():
                    unique_entries[uid] = entry

        final_entries = list(unique_entries.values())
        logger.info(f"✅ COMPREHENSIVE INFOLOG FETCH: {len(final_entries)} unique entries from {len(unique_collections)} collections")

        # Log collection summary
        collection_summary = {}
        for entry in final_entries:
            collection = entry.get('_collection', 'unknown')
            collection_summary[collection] = collection_summary.get(collection, 0) + 1

        for collection, count in collection_summary.items():
            logger.info(f"  - {collection}: {count} entries")

        return final_entries

    def get_infolog_entries(self, since_date: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Fetch InfoLog entries with support for incremental updates

        Args:
            since_date: Optional ISO formatted date to fetch only items modified since this date
        """
        logger.info(f"Fetching {'modified' if since_date else 'ALL'} InfoLog entries...")
        return self._get_all_infolog_entries_comprehensive(since_date)

    def get_contact_by_id(self, contact_id: str) -> Optional[Dict[str, Any]]:
        """Get a single contact by ID"""
        try:
            return self._request_single(f"/addressbook/{contact_id}")
        except Exception as e:
            logger.error(f"Failed to fetch contact {contact_id}: {e}")
            return None

    def get_event_by_id(self, event_id: str) -> Optional[Dict[str, Any]]:
        """Get a single calendar event by ID"""
        try:
            return self._request_single(f"/calendar/{event_id}")
        except Exception as e:
            logger.error(f"Failed to fetch event {event_id}: {e}")
            return None

    def get_infolog_by_id(self, entry_id: str) -> Optional[Dict[str, Any]]:
        """Get a single InfoLog entry by ID"""
        try:
            return self._request_single(f"/infolog/{entry_id}")
        except Exception as e:
            logger.error(f"Failed to fetch InfoLog entry {entry_id}: {e}")
            return None

    def search_contacts(self, search_pattern: str) -> List[Dict[str, Any]]:
        """
        Search contacts using EGroupware's built-in search functionality
        """
        logger.info(f"Searching contacts for pattern: {search_pattern}")
        url = f"{self.base_url}/addressbook/"

        try:
            params = {
                'filters[search]': search_pattern
            }

            response = self.session.get(url, params=params, timeout=self.timeout)
            response.raise_for_status()

            if not response.content:
                return []

            data = response.json()

            contacts = []
            if "responses" in data:
                for url_path, contact_data in data["responses"].items():
                    if contact_data and isinstance(contact_data, dict) and contact_data.get("@type") == "Card":
                        contact_data["_url"] = url_path
                        if "/" in url_path:
                            contact_data["_id"] = url_path.split("/")[-1]
                        contacts.append(contact_data)

            logger.info(f"✅ Found {len(contacts)} contacts matching '{search_pattern}'")
            return contacts

        except Exception as e:
            logger.error(f"Contact search failed: {e}")
            return []

    def _discover_calendar_collections(self) -> List[str]:
        """
        Discover all available calendar collections including group-based ones
        """
        collections = []

        # Add the default calendar collection
        collections.append("/calendar/")

        # Try to discover additional calendar collections
        try:
            # First, try to get all available collections via PROPFIND
            url = f"{self.base_url}/"
            response = self.session.request(
                'PROPFIND',
                url,
                headers={
                    'Accept': 'application/xml',
                    'Depth': '1',
                    'Content-Type': 'application/xml'
                },
                timeout=self.timeout
            )

            if response.status_code == 200:
                import re
                # Extract all calendar hrefs from the XML response
                calendar_hrefs = re.findall(r'<D:href>([^<]*calendar[^<]*)</D:href>', response.text)
                for href in calendar_hrefs:
                    # Extract the collection path relative to base_url
                    if href.startswith('/'):
                        # Convert absolute path to relative path
                        path_parts = href.split('/')
                        if 'calendar' in path_parts:
                            idx = path_parts.index('calendar')
                            collection_path = '/' + '/'.join(path_parts[idx:])
                            if collection_path not in collections:
                                collections.append(collection_path)

            # Also try some common group-based calendar paths
            common_paths = [
                "/calendar-shared/",
                "/calendar-global/",
                "/calendar-public/",
                "/shared/calendar/",
                "/public/calendar/"
            ]

            for path in common_paths:
                if path not in collections:
                    # Test if this collection exists by making a small request
                    try:
                        test_url = f"{self.base_url}{path}"
                        test_response = self.session.get(
                            test_url,
                            params={'nresults': 1},
                            timeout=10
                        )
                        if test_response.status_code == 200:
                            collections.append(path)
                            logger.info(f"Found additional calendar collection: {path}")
                    except:
                        pass

        except Exception as e:
            logger.warning(f"Calendar collection discovery failed, using default only: {e}")

        logger.info(f"Discovered {len(collections)} calendar collections: {collections}")
        return collections

    def _discover_group_calendars(self) -> List[str]:
        """
        Discover group-specific calendar collections using EGroupware-specific patterns
        """
        group_collections = []

        try:
            logger.info("Starting comprehensive group calendar discovery...")

            # Pattern 1: /calendar-<group_id>/
            logger.info("Testing pattern: /calendar-<group_id>/")
            for group_id in range(1, 50):  # Test group IDs (1-50)
                path = f"/calendar-{group_id}/"
                try:
                    test_url = f"{self.base_url}{path}"
                    test_response = self.session.get(
                        test_url,
                        params={'nresults': 1},
                        timeout=5
                    )
                    if test_response.status_code == 200:
                        try:
                            data = test_response.json()
                            if data.get('responses'):
                                group_collections.append(path)
                                logger.info(f"✅ Found group calendar collection: {path}")
                        except:
                            if test_response.content:
                                group_collections.append(path)
                                logger.info(f"✅ Found group calendar collection: {path} (Non-JSON)")
                except Exception as e:
                    logger.debug(f"Calendar group {group_id} failed: {e}")
                    continue

            # Pattern 2: Try accessing shared/group calendars
            shared_patterns = [
                "/shared/",
                "/groups/",
                "/calendar/shared/",
                "/calendar/groups/",
                "/calendar/public/",
                "/public/",
                "/global/",
                "/calendar/global/",
                # EGroupware specific patterns
                "/calendar/Business/",
                "/calendar/business/",
                "/Business/calendar/",
                "/business/calendar/",
                "/Gruppe/calendar/",
                "/gruppe/calendar/",
                "/calendar/Gruppe/",
                "/calendar/gruppe/",
            ]

            for pattern in shared_patterns:
                try:
                    test_url = f"{self.base_url}{pattern}"
                    test_response = self.session.get(test_url, timeout=5)
                    logger.debug(f"Testing calendar {pattern}: Status {test_response.status_code}")

                    if test_response.status_code == 200:
                        try:
                            data = test_response.json()
                            if data.get('responses'):
                                group_collections.append(pattern)
                                logger.info(f"✅ Found shared calendar collection: {pattern}")
                        except:
                            if test_response.content and len(test_response.content) > 100:
                                group_collections.append(pattern)
                                logger.info(f"✅ Found shared calendar collection: {pattern} (Non-JSON)")
                except Exception as e:
                    logger.debug(f"Shared calendar pattern {pattern} failed: {e}")
                    continue

        except Exception as e:
            logger.warning(f"Group calendar discovery failed: {e}")

        logger.info(f"Calendar group discovery complete. Found {len(group_collections)} additional collections: {group_collections}")
        return group_collections

    def _discover_infolog_collections(self) -> List[str]:
        """
        Discover all available infolog collections including group-based ones
        """
        collections = []

        # Add the default infolog collection
        collections.append("/infolog/")

        # Try to discover additional infolog collections
        try:
            # First, try to get all available collections via PROPFIND
            url = f"{self.base_url}/"
            response = self.session.request(
                'PROPFIND',
                url,
                headers={
                    'Accept': 'application/xml',
                    'Depth': '1',
                    'Content-Type': 'application/xml'
                },
                timeout=self.timeout
            )

            if response.status_code == 200:
                import re
                # Extract all infolog hrefs from the XML response
                infolog_hrefs = re.findall(r'<D:href>([^<]*infolog[^<]*)</D:href>', response.text)
                for href in infolog_hrefs:
                    # Extract the collection path relative to base_url
                    if href.startswith('/'):
                        # Convert absolute path to relative path
                        path_parts = href.split('/')
                        if 'infolog' in path_parts:
                            idx = path_parts.index('infolog')
                            collection_path = '/' + '/'.join(path_parts[idx:])
                            if collection_path not in collections:
                                collections.append(collection_path)

            # Also try some common group-based infolog paths
            common_paths = [
                "/infolog-shared/",
                "/infolog-global/",
                "/infolog-public/",
                "/shared/infolog/",
                "/public/infolog/"
            ]

            for path in common_paths:
                if path not in collections:
                    # Test if this collection exists by making a small request
                    try:
                        test_url = f"{self.base_url}{path}"
                        test_response = self.session.get(
                            test_url,
                            params={'nresults': 1},
                            timeout=10
                        )
                        if test_response.status_code == 200:
                            collections.append(path)
                            logger.info(f"Found additional infolog collection: {path}")
                    except:
                        pass

        except Exception as e:
            logger.warning(f"InfoLog collection discovery failed, using default only: {e}")

        logger.info(f"Discovered {len(collections)} infolog collections: {collections}")
        return collections

    def _discover_group_infologs(self) -> List[str]:
        """
        Discover group-specific infolog collections using EGroupware-specific patterns
        """
        group_collections = []

        try:
            logger.info("Starting comprehensive group infolog discovery...")

            # Pattern 1: /infolog-<group_id>/
            logger.info("Testing pattern: /infolog-<group_id>/")
            for group_id in range(1, 50):  # Test group IDs (1-50)
                path = f"/infolog-{group_id}/"
                try:
                    test_url = f"{self.base_url}{path}"
                    test_response = self.session.get(
                        test_url,
                        params={'nresults': 1},
                        timeout=5
                    )
                    if test_response.status_code == 200:
                        try:
                            data = test_response.json()
                            if data.get('responses'):
                                group_collections.append(path)
                                logger.info(f"✅ Found group infolog collection: {path}")
                        except:
                            if test_response.content:
                                group_collections.append(path)
                                logger.info(f"✅ Found group infolog collection: {path} (Non-JSON)")
                except Exception as e:
                    logger.debug(f"InfoLog group {group_id} failed: {e}")
                    continue

            # Pattern 2: Try accessing shared/group infologs
            shared_patterns = [
                "/shared/",
                "/groups/",
                "/infolog/shared/",
                "/infolog/groups/",
                "/infolog/public/",
                "/public/",
                "/global/",
                "/infolog/global/",
                # EGroupware specific patterns
                "/infolog/Business/",
                "/infolog/business/",
                "/Business/infolog/",
                "/business/infolog/",
                "/Gruppe/infolog/",
                "/gruppe/infolog/",
                "/infolog/Gruppe/",
                "/infolog/gruppe/",
            ]

            for pattern in shared_patterns:
                try:
                    test_url = f"{self.base_url}{pattern}"
                    test_response = self.session.get(test_url, timeout=5)
                    logger.debug(f"Testing infolog {pattern}: Status {test_response.status_code}")

                    if test_response.status_code == 200:
                        try:
                            data = test_response.json()
                            if data.get('responses'):
                                group_collections.append(pattern)
                                logger.info(f"✅ Found shared infolog collection: {pattern}")
                        except:
                            if test_response.content and len(test_response.content) > 100:
                                group_collections.append(pattern)
                                logger.info(f"✅ Found shared infolog collection: {pattern} (Non-JSON)")
                except Exception as e:
                    logger.debug(f"Shared infolog pattern {pattern} failed: {e}")
                    continue

        except Exception as e:
            logger.warning(f"Group infolog discovery failed: {e}")

        logger.info(f"InfoLog group discovery complete. Found {len(group_collections)} additional collections: {group_collections}")
        return group_collections
