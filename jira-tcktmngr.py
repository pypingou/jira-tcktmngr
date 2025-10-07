#!/usr/bin/env python3
"""
jira-tcktmngr - Jira Ticket Manager

A comprehensive tool for managing Jira tickets and their hierarchies.
Features include finding descendants, managing labels, closing/reopening tickets,
and working with subtasks and issue links.

Copyright (c) 2025
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its
   contributors may be used to endorse or promote products derived from
   this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

Enhanced with assistance from Claude (Anthropic AI Assistant)
"""

import requests
import json
import sys
import os
import time
import logging
from pathlib import Path
from typing import Set, List, Dict, Optional, Union
from dataclasses import dataclass
import argparse
import configparser
from functools import wraps
from enum import Enum
import random


# Constants
class Constants:
    """Application constants."""

    # Rate limiting
    MAX_RETRIES = 3
    BASE_RETRY_DELAY = 1.0
    MAX_RETRY_DELAY = 30.0
    BACKOFF_FACTOR = 2.0

    # API
    API_TIMEOUT = 30
    API_BATCH_SIZE = 100
    API_RATE_LIMIT_DELAY = 0.1

    # Status categories
    CLOSED_STATUSES = {
        "closed",
        "done",
        "resolved",
        "cancelled",
        "rejected",
        "completed",
        "finished",
        "released",
        "verified",
        "fixed",
    }

    # Link types that indicate child relationships
    CHILD_LINK_KEYWORDS = {"child", "subtask", "contains"}
    PARENT_LINK_KEYWORDS = {"parent", "epic"}

    # Issue types that can have Epic Link children
    EPIC_TYPES = {"Epic", "Initiative", "Feature"}


class LogLevel(Enum):
    """Logging levels."""

    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"


class JiraError(Exception):
    """Base exception for Jira-related errors."""

    pass


class AuthenticationError(JiraError):
    """Authentication failed."""

    pass


class RateLimitError(JiraError):
    """Rate limit exceeded."""

    pass


class IssueNotFoundError(JiraError):
    """Issue not found."""

    pass


class APIError(JiraError):
    """General API error."""

    def __init__(
        self,
        message: str,
        status_code: Optional[int] = None,
        response_text: Optional[str] = None,
    ):
        super().__init__(message)
        self.status_code = status_code
        self.response_text = response_text


def setup_logging(
    level: LogLevel = LogLevel.INFO, include_timestamp: bool = True
) -> logging.Logger:
    """Set up logging configuration."""
    logger = logging.getLogger("jira-tcktmngr")
    logger.setLevel(level.value)

    # Remove existing handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

    # Create handler
    handler = logging.StreamHandler(sys.stdout)

    # Create formatter
    if include_timestamp:
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
    else:
        formatter = logging.Formatter("%(levelname)s - %(message)s")

    handler.setFormatter(formatter)
    logger.addHandler(handler)

    return logger


def retry_with_backoff(
    max_retries: int = Constants.MAX_RETRIES,
    base_delay: float = Constants.BASE_RETRY_DELAY,
    max_delay: float = Constants.MAX_RETRY_DELAY,
    backoff_factor: float = Constants.BACKOFF_FACTOR,
):
    """Decorator for exponential backoff retry logic."""

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            logger = logging.getLogger("jira-tcktmngr")

            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except RateLimitError as e:
                    if attempt == max_retries:
                        logger.error(
                            f"Max retries ({max_retries}) exceeded for {func.__name__}"
                        )
                        raise

                    # Calculate delay with jitter
                    delay = min(base_delay * (backoff_factor**attempt), max_delay)
                    jitter = random.uniform(0.1, 0.3) * delay
                    total_delay = delay + jitter

                    logger.debug(
                        f"Rate limited, retrying {func.__name__} in {total_delay:.2f}s (attempt {attempt + 1}/{max_retries + 1})"
                    )
                    time.sleep(total_delay)
                except requests.RequestException as e:
                    if attempt == max_retries:
                        logger.error(
                            f"Max retries ({max_retries}) exceeded for {func.__name__}: {e}"
                        )
                        raise APIError(
                            f"Request failed after {max_retries} retries: {e}"
                        )

                    delay = min(base_delay * (backoff_factor**attempt), max_delay)
                    logger.debug(
                        f"Request failed, retrying {func.__name__} in {delay:.2f}s: {e}"
                    )
                    time.sleep(delay)

            return None

        return wrapper

    return decorator


class Colors:
    """ANSI color codes for terminal output."""

    # Status colors
    GREEN = "\033[92m"  # Open issues
    RED = "\033[91m"  # Closed issues

    # Text formatting
    BOLD = "\033[1m"  # Issue keys
    DIM = "\033[2m"  # Details/metadata

    # Hierarchy colors
    BLUE = "\033[94m"  # Tree structure
    CYAN = "\033[96m"  # Issue summaries
    YELLOW = "\033[93m"  # Labels

    # Reset
    RESET = "\033[0m"  # Reset to default

    @staticmethod
    def disable_colors():
        """Disable colors for non-terminal output."""
        Colors.GREEN = Colors.RED = Colors.BOLD = Colors.DIM = ""
        Colors.BLUE = Colors.CYAN = Colors.YELLOW = Colors.RESET = ""


@dataclass
class JiraIssue:
    key: str
    summary: str
    issue_type: str
    status: str
    labels: List[str]
    level: int = 0
    sub_system_group: Optional[str] = None


class JiraConfig:
    def __init__(self, config_path: Optional[str] = None) -> None:
        if config_path is None:
            config_path = os.path.expanduser("~/.config/jira")

        self.config_path = config_path
        self.config = configparser.ConfigParser()

        if os.path.exists(config_path):
            self.config.read(config_path)
        else:
            raise FileNotFoundError(f"Config file not found: {config_path}")

    def get(self, section: str, key: str, fallback: str = None) -> str:
        return self.config.get(section, key, fallback=fallback)

    @property
    def base_url(self) -> str:
        return self.get("jira", "base_url") or self.get("DEFAULT", "base_url")

    @property
    def username(self) -> str:
        return self.get("jira", "username") or self.get("DEFAULT", "username")

    @property
    def api_token(self) -> str:
        return self.get("jira", "api_token") or self.get("DEFAULT", "api_token")


class JiraDescendantFinder:
    def __init__(self, base_url: str, username: str, api_token: str) -> None:
        self.base_url = base_url.rstrip("/")
        self.session = requests.Session()
        self.session.headers.update({"Authorization": f"Bearer {api_token}"})
        self.session.timeout = Constants.API_TIMEOUT
        self.visited: Set[str] = set()
        self.logger = logging.getLogger("jira-tcktmngr")

    @retry_with_backoff()
    def test_auth(self) -> bool:
        """Test authentication by calling the /myself endpoint."""
        url = f"{self.base_url}/rest/api/2/myself"

        try:
            response = self.session.get(url)
            self._handle_response_errors(response)
            user_info = response.json()

            self.logger.info("Authentication successful")
            print(f"✓ Authentication successful!")
            print(f"  User: {user_info.get('displayName', 'Unknown')}")
            print(f"  Email: {user_info.get('emailAddress', 'Unknown')}")
            print(f"  Account ID: {user_info.get('accountId', 'Unknown')}")
            return True
        except AuthenticationError as e:
            self.logger.error(f"Authentication failed: {e}")
            print(f"✗ Authentication failed: {e}")
            return False
        except APIError as e:
            self.logger.error(f"API error during authentication: {e}")
            print(f"✗ Authentication failed: {e}")
            return False

    def _handle_response_errors(self, response: requests.Response) -> None:
        """Handle common HTTP response errors."""
        if response.status_code == 401:
            raise AuthenticationError("Invalid credentials (username/API token)")
        elif response.status_code == 403:
            raise AuthenticationError("Access forbidden (check permissions)")
        elif response.status_code == 404:
            raise IssueNotFoundError("Resource not found")
        elif response.status_code == 429:
            raise RateLimitError("Rate limit exceeded")
        elif not response.ok:
            raise APIError(
                f"HTTP {response.status_code}: {response.reason}",
                status_code=response.status_code,
                response_text=response.text[:500] if response.text else None,
            )

    def check_claude_available(self) -> bool:
        """Check if claude CLI is available on the system."""
        import subprocess

        try:
            result = subprocess.run(
                ["claude", "--version"], capture_output=True, text=True, timeout=5
            )
            return result.returncode == 0
        except (
            subprocess.TimeoutExpired,
            FileNotFoundError,
            subprocess.SubprocessError,
        ):
            return False

    @retry_with_backoff()
    def get_issue_detailed(self, issue_key: str) -> Optional[Dict]:
        """Fetch detailed issue information including description and comments."""
        url = f"{self.base_url}/rest/api/2/issue/{issue_key}"
        params = {
            "expand": "subtask,issuelinks,comments",
            "fields": "summary,description,issuetype,status,subtasks,issuelinks,epic,parent,labels,customfield_12313140,customfield_12315542,customfield_12316342,customfield_12321140,customfield_12326540,customfield_12320851,customfield_12312940,customfield_12313940,comment",
        }

        try:
            response = self.session.get(url, params=params)
            if response.status_code == 429:
                raise RateLimitError(f"Rate limited while fetching issue {issue_key}")
            self._handle_response_errors(response)
            return response.json()
        except (RateLimitError, AuthenticationError, APIError):
            raise
        except IssueNotFoundError:
            self.logger.warning(f"Issue {issue_key} not found")
            return None
        except Exception as e:
            self.logger.error(f"Unexpected error fetching issue {issue_key}: {e}")
            raise APIError(f"Failed to fetch issue {issue_key}: {e}")

    def summarize_issue(self, issue_key: str) -> bool:
        """Summarize an issue using claude CLI if available."""
        import subprocess

        if not self.check_claude_available():
            print("Error: claude CLI is not available on this system.")
            print("Please install claude CLI to use the summarize feature.")
            return False

        issue_data = self.get_issue_detailed(issue_key)
        if not issue_data:
            print(f"Could not fetch issue {issue_key}")
            return False

        fields = issue_data.get("fields", {})

        # Build content to summarize
        content_parts = []
        content_parts.append(f"Issue: {issue_key}")
        content_parts.append(f"Summary: {fields.get('summary', 'No summary')}")
        content_parts.append(
            f"Type: {fields.get('issuetype', {}).get('name', 'Unknown')}"
        )
        content_parts.append(
            f"Status: {fields.get('status', {}).get('name', 'Unknown')}"
        )

        if fields.get("labels"):
            content_parts.append(f"Labels: {', '.join(fields.get('labels'))}")

        description = fields.get("description")
        if description:
            content_parts.append(f"\nDescription:\n{description}")

        # Add comments if any
        comments = fields.get("comment", {}).get("comments", [])
        if comments:
            content_parts.append("\nComments:")
            for i, comment in enumerate(comments[-5:], 1):  # Last 5 comments
                author = comment.get("author", {}).get("displayName", "Unknown")
                body = comment.get("body", "")
                content_parts.append(f"\nComment {i} by {author}:\n{body}")

        issue_content = "\n".join(content_parts)

        try:
            # Use claude CLI to summarize
            full_prompt = f"""Please provide a concise summary of this Jira ticket, highlighting the key points, current status, and any important details or recent developments:

{issue_content}"""

            cmd = ["claude", "--print", full_prompt]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)

            if result.returncode == 0:
                # Disable colors if not outputting to a terminal
                if not sys.stdout.isatty():
                    Colors.disable_colors()

                print(
                    f"\n{Colors.BOLD}{Colors.CYAN}Summary for {issue_key}:{Colors.RESET}"
                )
                print(f"{Colors.BLUE}{'=' * 50}{Colors.RESET}")

                # Color-code the summary output
                summary_lines = result.stdout.strip().split("\n")
                for line in summary_lines:
                    if line.startswith("**") and line.endswith("**"):
                        # Headers/titles in bold cyan
                        print(f"{Colors.BOLD}{Colors.CYAN}{line}{Colors.RESET}")
                    elif line.startswith("- ") or line.startswith("• "):
                        # Bullet points in green
                        print(f"{Colors.GREEN}{line}{Colors.RESET}")
                    elif ":" in line and not line.strip().startswith("-"):
                        # Lines with colons (key-value pairs) - make the key bold
                        parts = line.split(":", 1)
                        if len(parts) == 2:
                            print(f"{Colors.BOLD}{parts[0]}:{Colors.RESET}{parts[1]}")
                        else:
                            print(line)
                    else:
                        # Regular text
                        print(line)

                return True
            else:
                print(f"Error running claude CLI: {result.stderr}")
                return False

        except subprocess.TimeoutExpired:
            print("Error: claude CLI timed out")
            return False
        except Exception as e:
            print(f"Error running claude CLI: {e}")
            return False

    @retry_with_backoff()
    def get_issue(self, issue_key: str) -> Optional[Dict]:
        """Fetch issue details from Jira API."""
        url = f"{self.base_url}/rest/api/2/issue/{issue_key}"
        params = {
            "expand": "subtask,issuelinks",
            "fields": "summary,issuetype,status,subtasks,issuelinks,epic,parent,labels,customfield_12313140,customfield_12315542,customfield_12316342,customfield_12321140,customfield_12326540,customfield_12320851,customfield_12312940,customfield_12313940",
        }

        try:
            response = self.session.get(url, params=params)
            if response.status_code == 429:
                raise RateLimitError(f"Rate limited while fetching issue {issue_key}")
            self._handle_response_errors(response)
            return response.json()
        except (RateLimitError, AuthenticationError, APIError):
            raise
        except IssueNotFoundError:
            self.logger.warning(f"Issue {issue_key} not found")
            return None
        except Exception as e:
            self.logger.error(f"Unexpected error fetching issue {issue_key}: {e}")
            raise APIError(f"Failed to fetch issue {issue_key}: {e}")

    def debug_issue_links(self, issue_key: str) -> None:
        """Debug what links and subtasks exist for an issue."""
        # For debug, fetch ALL fields
        url = f"{self.base_url}/rest/api/2/issue/{issue_key}"
        params = {
            "expand": "subtask,issuelinks",
            "fields": "*all",
        }

        try:
            response = self.session.get(url, params=params)
            self._handle_response_errors(response)
            issue_data = response.json()
        except Exception as e:
            print(f"Error fetching issue data: {e}")
            return

        fields = issue_data.get("fields", {})
        print(f"\nDEBUG - Issue {issue_key}:")
        print(f"  Summary: {fields.get('summary', '')}")

        # Debug subtasks
        subtasks = fields.get("subtasks", [])
        print(f"  Subtasks ({len(subtasks)}):")
        for subtask in subtasks:
            print(
                f"    - {subtask.get('key')}: {subtask.get('fields', {}).get('summary', '')}"
            )

        # Debug issue links
        issue_links = fields.get("issuelinks", [])
        print(f"  Issue Links ({len(issue_links)}):")
        for link in issue_links:
            link_type = link.get("type", {})
            if "outwardIssue" in link:
                direction = "outward"
                linked_issue = link["outwardIssue"]
                link_name = link_type.get("outward", "")
            elif "inwardIssue" in link:
                direction = "inward"
                linked_issue = link["inwardIssue"]
                link_name = link_type.get("inward", "")
            else:
                continue

            print(f"    - {direction}: {linked_issue.get('key')} ({link_name})")
            print(f"      Summary: {linked_issue.get('fields', {}).get('summary', '')}")

        # Debug other fields
        print(f"  Epic: {fields.get('epic')}")
        print(f"  Parent: {fields.get('parent')}")
        print(f"  Issue Type: {fields.get('issuetype', {}).get('name')}")

        # Show all available fields
        print(f"  All Fields: {list(fields.keys())}")

        # Look for custom fields that might contain the parent reference
        print(f"\n  Custom fields with values:")
        for field_name, field_value in fields.items():
            if field_name.startswith("customfield_") and field_value:
                # Check if the value contains any ticket reference
                field_str = str(field_value)
                if (
                    any(proj in field_str for proj in ["AUTOBU", "VROOM"])
                    or "key" in field_str.lower()
                ):
                    print(f"    {field_name}: {field_value}")

        # Look for fields that might be Sub-System Group related
        print(f"\n  Fields potentially related to Sub-System Group:")
        for field_name, field_value in fields.items():
            if field_value:
                field_str = str(field_value).lower()
                if any(keyword in field_str for keyword in ["rhivos", "sub", "group", "system"]) or any(keyword in field_name.lower() for keyword in ["sub", "group", "system"]):
                    print(f"    {field_name}: {field_value}")

        # Check for sub-system group field specifically (customfield_12320851 is the actual Sub-System Group field)
        sub_system_group_candidates = ["customfield_12320851", "customfield_12312940", "customfield_12313940", "Sub-System Group"]
        print(f"\n  Sub-System Group field candidates:")
        for field_candidate in sub_system_group_candidates:
            value = fields.get(field_candidate)
            if value:
                print(f"    {field_candidate}: {value}")
            else:
                print(f"    {field_candidate}: (not found)")

        # Search for issues that have this as Epic
        if fields.get("issuetype", {}).get("name") in ["Epic", "Feature", "Initiative"]:
            self.search_epic_children(issue_key)

        # Search for any issues that reference this one
        self.search_referencing_issues(issue_key)

    def search_epic_children(self, epic_key: str) -> None:
        """Search for issues that belong to this epic."""
        print(f"\n  Searching for Epic children...")
        url = f"{self.base_url}/rest/api/2/search"
        params = {
            "jql": f'"Epic Link" = {epic_key}',
            "fields": "key,summary,issuetype,status,labels",
            "maxResults": 100,
        }

        try:
            response = self.session.get(url, params=params)
            response.raise_for_status()
            data = response.json()
            issues = data.get("issues", [])
            print(f"  Epic children found: {len(issues)}")
            for issue in issues:
                fields = issue.get("fields", {})
                print(f"    - {issue.get('key')}: {fields.get('summary', '')}")
        except requests.RequestException as e:
            print(f"    Error searching epic children: {e}")

    def search_referencing_issues(self, issue_key: str) -> None:
        """Search for issues that reference this issue."""
        print(f"\n  Searching for issues that reference {issue_key}...")

        # Try various JQL patterns
        queries = [
            f'text ~ "{issue_key}"',
            f'summary ~ "{issue_key}"',
            f'description ~ "{issue_key}"',
            f'parent = "{issue_key}"',
            f'project in (AUTOBU, VROOM) AND text ~ "{issue_key}"',
            f'project in (AUTOBU, VROOM) AND summary ~ "{issue_key}"',
            f'"Epic Link" = "{issue_key}"',
            f'"Feature Link" = "{issue_key}"',
            f'"Outcome Link" = "{issue_key}"',
            f'"Parent Feature" = "{issue_key}"',
            f'"Initiative Link" = "{issue_key}"',
            f'"Parent Initiative" = "{issue_key}"',
        ]

        for query in queries:
            url = f"{self.base_url}/rest/api/2/search"
            params = {
                "jql": query,
                "fields": "key,summary,issuetype,status,labels",
                "maxResults": 10,
            }

            try:
                response = self.session.get(url, params=params)
                if response.status_code == 200:
                    data = response.json()
                    issues = data.get("issues", [])
                    if issues:
                        print(f"    Query '{query}' found {len(issues)} issues:")
                        for issue in issues:
                            fields = issue.get("fields", {})
                            print(
                                f"      - {issue.get('key')}: {fields.get('summary', '')}"
                            )
                        break
            except requests.RequestException:
                continue
        else:
            print("    No referencing issues found")

    def find_custom_field_children(self, issue_key: str) -> List[JiraIssue]:
        """Find children via custom fields like Feature Link and Initiative Link."""
        children = []

        # Try multiple custom field searches
        custom_field_queries = [
            f'"Feature Link" = "{issue_key}"',
            f'cf[12313140] = "{issue_key}"',  # Initiative/Parent link field
            f'"Initiative Link" = "{issue_key}"',
            f'"Parent Initiative" = "{issue_key}"',
        ]

        for jql_query in custom_field_queries:
            url = f"{self.base_url}/rest/api/2/search"
            params = {
                "jql": jql_query,
                "fields": "key,summary,issuetype,status,labels",
                "maxResults": 100,
            }

            try:
                response = self.session.get(url, params=params)
                if response.status_code == 200:
                    data = response.json()
                    issues = data.get("issues", [])

                    for issue_data in issues:
                        issue_key_found = issue_data.get("key", "")
                        # Avoid duplicates
                        if not any(child.key == issue_key_found for child in children):
                            fields = issue_data.get("fields", {})
                            child = JiraIssue(
                                key=issue_key_found,
                                summary=fields.get("summary", ""),
                                issue_type=fields.get("issuetype", {}).get("name", ""),
                                status=fields.get("status", {}).get("name", ""),
                                labels=fields.get("labels", []),
                                level=1,  # Will be adjusted by caller
                            )
                            children.append(child)

            except requests.RequestException:
                continue

        return children

    def search_epic_link_children(self, epic_key: str, level: int) -> List[JiraIssue]:
        """Search for issues that have this epic as their Epic Link and recursively find their children."""
        descendants = []

        url = f"{self.base_url}/rest/api/2/search"
        params = {
            "jql": f'"Epic Link" = {epic_key}',
            "fields": "key,summary,issuetype,status,labels",
            "maxResults": 100,
        }

        try:
            response = self.session.get(url, params=params)
            if response.status_code == 429:  # Rate limited
                time.sleep(2)
                response = self.session.get(url, params=params)
            if response.status_code == 200:
                data = response.json()
                issues = data.get("issues", [])

                for issue_data in issues:
                    child_key = issue_data.get("key", "")
                    if child_key and child_key not in self.visited:
                        # Recursively find all descendants of this child
                        descendants.extend(self.find_descendants(child_key, level))

        except requests.RequestException:
            pass

        return descendants

    def find_epic_children(self, epic_key: str) -> List[JiraIssue]:
        """Find issues that belong to this epic via Epic Link."""
        children = []

        url = f"{self.base_url}/rest/api/2/search"
        params = {
            "jql": f'"Epic Link" = {epic_key}',
            "fields": "key,summary,issuetype,status,labels",
            "maxResults": 100,
        }

        try:
            response = self.session.get(url, params=params)
            if response.status_code == 200:
                data = response.json()
                issues = data.get("issues", [])

                for issue_data in issues:
                    fields = issue_data.get("fields", {})
                    child = JiraIssue(
                        key=issue_data.get("key", ""),
                        summary=fields.get("summary", ""),
                        issue_type=fields.get("issuetype", {}).get("name", ""),
                        status=fields.get("status", {}).get("name", ""),
                        labels=fields.get("labels", []),
                        level=1,  # Will be adjusted by caller
                    )
                    children.append(child)

        except requests.RequestException:
            pass

        return children

    def _create_jira_issue_from_data(
        self, issue_key: str, issue_data: Dict, level: int = 0
    ) -> JiraIssue:
        """Create a JiraIssue object from API response data."""
        fields = issue_data.get("fields", {})

        # Extract sub-system group field - customfield_12320851 is the actual Sub-System Group field
        sub_system_group = None
        for field_candidate in ["customfield_12320851", "customfield_12312940", "customfield_12313940", "Sub-System Group"]:
            value = fields.get(field_candidate)
            if value:
                # Handle different possible formats (string, dict with value/name, list of dicts, etc.)
                if isinstance(value, list) and value:
                    # Handle list of dictionaries (common for multi-select fields)
                    if isinstance(value[0], dict):
                        sub_system_group = value[0].get("value") or value[0].get("name") or str(value[0])
                    else:
                        sub_system_group = str(value[0])
                elif isinstance(value, dict):
                    sub_system_group = value.get("value") or value.get("name") or str(value)
                else:
                    sub_system_group = str(value)
                break

        return JiraIssue(
            key=issue_key,
            summary=fields.get("summary", ""),
            issue_type=fields.get("issuetype", {}).get("name", ""),
            status=fields.get("status", {}).get("name", ""),
            labels=fields.get("labels", []),
            level=level,
            sub_system_group=sub_system_group,
        )

    def _find_subtask_children(self, fields: Dict, level: int) -> List[JiraIssue]:
        """Find direct subtasks of an issue."""
        children = []
        subtasks = fields.get("subtasks", [])
        for subtask in subtasks:
            child_key = subtask.get("key")
            if child_key and child_key not in self.visited:
                children.extend(self.find_descendants(child_key, level + 1))
        return children

    def _find_link_children(self, fields: Dict, level: int) -> List[JiraIssue]:
        """Find children through issue links."""
        children = []
        issue_links = fields.get("issuelinks", [])

        for link in issue_links:
            # Check for outward links (this issue links to another)
            if "outwardIssue" in link:
                link_type = link.get("type", {}).get("outward", "").lower()
                if any(
                    keyword in link_type for keyword in Constants.CHILD_LINK_KEYWORDS
                ):
                    child_key = link["outwardIssue"].get("key")
                    if child_key and child_key not in self.visited:
                        children.extend(self.find_descendants(child_key, level + 1))

        return children

    def _find_epic_children(
        self, issue_key: str, issue_type: str, level: int
    ) -> List[JiraIssue]:
        """Find children through Epic Link if this is an epic-type issue."""
        if issue_type not in Constants.EPIC_TYPES:
            return []

        return self.search_epic_link_children(issue_key, level + 1)

    def _find_custom_field_children(
        self, issue_key: str, level: int
    ) -> List[JiraIssue]:
        """Find children through custom fields."""
        children = []
        custom_children = self.find_custom_field_children(issue_key)

        for child in custom_children:
            if child.key not in self.visited:
                child.level = level + 1
                child_descendants = self.find_descendants(child.key, level + 1)
                children.extend(child_descendants)

        return children

    def find_descendants(self, issue_key: str, level: int = 0) -> List[JiraIssue]:
        """Recursively find all descendants of an issue."""
        if issue_key in self.visited:
            return []

        self.visited.add(issue_key)

        # Get issue data
        issue_data = self.get_issue(issue_key)
        if not issue_data:
            self.logger.warning(f"Could not fetch issue data for {issue_key}")
            return []

        # Create main issue
        descendants = [self._create_jira_issue_from_data(issue_key, issue_data, level)]

        fields = issue_data.get("fields", {})
        issue_type = fields.get("issuetype", {}).get("name", "")

        # Find all types of children
        try:
            # Direct subtasks
            descendants.extend(self._find_subtask_children(fields, level))

            # Epic Link children (if this is an epic-type issue)
            descendants.extend(self._find_epic_children(issue_key, issue_type, level))

            # Linked issues
            descendants.extend(self._find_link_children(fields, level))

            # Custom field children
            descendants.extend(self._find_custom_field_children(issue_key, level))

        except Exception as e:
            self.logger.error(f"Error finding descendants for {issue_key}: {e}")

        return descendants

    def is_issue_closed(self, status: str) -> bool:
        """Check if an issue status indicates it's closed."""
        return status.lower() in Constants.CLOSED_STATUSES

    def _perform_bulk_operation(
        self,
        issues: List[JiraIssue],
        operation_name: str,
        operation_func,
        *args,
        confirm: bool = True,
        show_sub_system_group: bool = False,
    ) -> int:
        """Perform a bulk operation on multiple issues with confirmation."""
        if not issues:
            self.logger.warning(f"No issues provided for {operation_name}")
            return 0

        # Show affected tickets
        print(f"\nAFFECTED TICKETS ({len(issues)}):")
        print("=" * 50)
        self.print_hierarchy(issues, show_labels=True, show_sub_system_group=show_sub_system_group)

        if confirm:
            response = input(
                f"\nProceed with {operation_name} on {len(issues)} issue(s) above? (y/N): "
            )
            if response.lower() not in ["y", "yes"]:
                print("Cancelled.")
                return 0

        # Perform operation
        print(f"\n{operation_name.capitalize()}...")
        success_count = 0
        for issue in issues:
            try:
                if operation_func(issue.key, *args):
                    success_count += 1
                time.sleep(Constants.API_RATE_LIMIT_DELAY)  # Be nice to the API
            except Exception as e:
                self.logger.error(f"Failed to {operation_name} for {issue.key}: {e}")
                print(f"  ✗ {issue.key}: Error - {e}")

        print(
            f"\nCompleted: {success_count}/{len(issues)} issues updated successfully."
        )
        return success_count

    @retry_with_backoff()
    def modify_label_on_issue(
        self, issue_key: str, label: str, add: bool = True
    ) -> bool:
        """Add or remove a label from a single issue."""
        url = f"{self.base_url}/rest/api/2/issue/{issue_key}"

        # First get current labels
        issue_data = self.get_issue(issue_key)
        if not issue_data:
            return False

        current_labels = issue_data.get("fields", {}).get("labels", [])

        if add:
            # Adding label
            if label in current_labels:
                print(f"  {issue_key}: Label '{label}' already exists")
                return True
            new_labels = current_labels + [label]
            action_verb = "Added"
        else:
            # Removing label
            if label not in current_labels:
                print(f"  {issue_key}: Label '{label}' does not exist")
                return True
            new_labels = [l for l in current_labels if l != label]
            action_verb = "Removed"

        payload = {"fields": {"labels": new_labels}}

        try:
            response = self.session.put(url, json=payload)
            if response.status_code == 429:
                raise RateLimitError(
                    f"Rate limited while modifying label on {issue_key}"
                )

            if response.status_code == 204:  # Success
                print(f"  ✓ {issue_key}: {action_verb} label '{label}'")
                return True
            else:
                self.logger.error(
                    f"Failed to modify label on {issue_key}: HTTP {response.status_code}"
                )
                print(
                    f"  ✗ {issue_key}: Failed to modify label (HTTP {response.status_code})"
                )
                return False

        except (RateLimitError, AuthenticationError, APIError):
            raise
        except Exception as e:
            self.logger.error(f"Unexpected error modifying label on {issue_key}: {e}")
            print(f"  ✗ {issue_key}: Error modifying label - {e}")
            return False

    def add_label_to_issue(self, issue_key: str, label: str) -> bool:
        """Add a label to a single issue."""
        return self.modify_label_on_issue(issue_key, label, add=True)

    def remove_label_from_issue(self, issue_key: str, label: str) -> bool:
        """Remove a label from a single issue."""
        return self.modify_label_on_issue(issue_key, label, add=False)

    @retry_with_backoff()
    def get_sub_system_group_options(self) -> List[Dict]:
        """Get available Sub-System group options from Jira."""
        # First, get field metadata to find the custom field options
        url = f"{self.base_url}/rest/api/2/field"

        try:
            response = self.session.get(url)
            if response.status_code == 429:
                raise RateLimitError("Rate limited while fetching field metadata")
            self._handle_response_errors(response)

            fields = response.json()
            sub_system_field = None

            for field in fields:
                if field.get("id") == "customfield_12320851":
                    sub_system_field = field
                    self.logger.debug(f"Found Sub-System group field: {field}")
                    # Found the Sub-System group field, get its options
                    if "allowedValues" in field:
                        self.logger.debug(f"Found allowedValues: {field['allowedValues']}")
                        return field["allowedValues"]
                    break

            if not sub_system_field:
                self.logger.debug("Sub-System group field customfield_12320851 not found in field list")
                # Log all custom fields for debugging
                custom_fields = [f for f in fields if f.get("id", "").startswith("customfield_")]
                self.logger.debug(f"Found {len(custom_fields)} custom fields")
                for cf in custom_fields[:10]:  # Log first 10 for debugging
                    self.logger.debug(f"Custom field: {cf.get('id')} - {cf.get('name')}")

            # If no allowedValues in field metadata, try alternative approaches

            # Try getting field configuration directly
            field_url = f"{self.base_url}/rest/api/2/customField/customfield_12320851/option"
            self.logger.debug(f"Trying field options URL: {field_url}")
            response = self.session.get(field_url)
            self.logger.debug(f"Field options response: {response.status_code}")
            if response.status_code == 200:
                data = response.json()
                self.logger.debug(f"Field options data: {data}")
                return data.get("values", [])

            # Try searching for existing tickets to extract possible values
            self.logger.debug("Trying to extract values from existing tickets")
            search_url = f"{self.base_url}/rest/api/2/search"
            params = {
                "jql": "project in (AUTOBU, VROOM) AND \"Sub-System Group\" is not EMPTY",
                "fields": "customfield_12320851",
                "maxResults": 50,
            }

            response = self.session.get(search_url, params=params)
            if response.status_code == 200:
                data = response.json()
                issues = data.get("issues", [])
                self.logger.debug(f"Found {len(issues)} issues with Sub-System group values")

                # Extract unique values with their IDs if available
                unique_options = {}
                for issue in issues:
                    field_value = issue.get("fields", {}).get("customfield_12320851")
                    if field_value:
                        if isinstance(field_value, dict):
                            value = field_value.get("value")
                            option_id = field_value.get("id")
                            if value:
                                unique_options[value] = option_id
                        elif isinstance(field_value, list):
                            for item in field_value:
                                if isinstance(item, dict):
                                    value = item.get("value")
                                    option_id = item.get("id")
                                    if value:
                                        unique_options[value] = option_id

                self.logger.debug(f"Found unique options: {unique_options}")
                # Convert to format similar to field options
                result = []
                for value in sorted(unique_options.keys()):
                    option_dict = {"value": value}
                    if unique_options[value]:
                        option_dict["id"] = unique_options[value]
                    result.append(option_dict)
                return result

            return []

        except (RateLimitError, AuthenticationError, APIError):
            raise
        except Exception as e:
            self.logger.error(f"Unexpected error fetching sub-system group options: {e}")
            return []

    @retry_with_backoff()
    def modify_sub_system_group_on_issue(
        self, issue_key: str, value: str, operation: str = "append"
    ) -> bool:
        """Modify the Sub-System group field on a single issue.

        Args:
            issue_key: The issue key to modify
            value: The sub-system group value to add/remove/set
            operation: 'append', 'remove', or 'replace'
        """
        url = f"{self.base_url}/rest/api/2/issue/{issue_key}"

        # First get current sub-system group values
        issue_data = self.get_issue(issue_key)
        if not issue_data:
            return False

        fields = issue_data.get("fields", {})
        current_field_value = fields.get("customfield_12320851")

        # Parse current values
        current_values = []
        if current_field_value:
            if isinstance(current_field_value, list):
                current_values = current_field_value
            else:
                current_values = [current_field_value]

        # Get current value strings for comparison
        current_value_strings = []
        for val in current_values:
            if isinstance(val, dict):
                current_value_strings.append(val.get("value", str(val)))
            else:
                current_value_strings.append(str(val))

        # Determine new values based on operation
        if operation == "replace":
            # Replace entire list with single new value
            new_values = [{"value": value}]
            action_verb = "Set"
        elif operation == "append":
            # Add new value if not already present
            if value in current_value_strings:
                print(f"  {issue_key}: Sub-System group '{value}' already exists")
                return True
            new_values = current_values + [{"value": value}]
            action_verb = "Added"
        elif operation == "remove":
            # Remove value if present
            if value not in current_value_strings:
                print(f"  {issue_key}: Sub-System group '{value}' does not exist")
                return True
            new_values = [val for val in current_values
                         if (isinstance(val, dict) and val.get("value") != value) or
                            (not isinstance(val, dict) and str(val) != value)]
            action_verb = "Removed"
        else:
            raise ValueError(f"Invalid operation: {operation}")

        # If field is single-select, only take first value
        if len(new_values) > 1:
            # Check if field allows multiple values by trying to set multiple
            # For now, assume single-select and use first value
            new_values = new_values[:1] if new_values else []

        payload = {"fields": {"customfield_12320851": new_values[0] if new_values else None}}

        try:
            response = self.session.put(url, json=payload)
            if response.status_code == 429:
                raise RateLimitError(
                    f"Rate limited while modifying sub-system group on {issue_key}"
                )

            if response.status_code == 204:  # Success
                print(f"  ✓ {issue_key}: {action_verb} sub-system group '{value}'")
                return True
            else:
                self.logger.error(
                    f"Failed to modify sub-system group on {issue_key}: HTTP {response.status_code}"
                )
                print(
                    f"  ✗ {issue_key}: Failed to modify sub-system group (HTTP {response.status_code})"
                )
                if response.text:
                    print(f"      Error details: {response.text}")
                return False

        except (RateLimitError, AuthenticationError, APIError):
            raise
        except Exception as e:
            self.logger.error(f"Unexpected error modifying sub-system group on {issue_key}: {e}")
            print(f"  ✗ {issue_key}: Error modifying sub-system group - {e}")
            return False

    def add_sub_system_group_to_issue(self, issue_key: str, value: str) -> bool:
        """Add a sub-system group to a single issue."""
        return self.modify_sub_system_group_on_issue(issue_key, value, "append")

    def remove_sub_system_group_from_issue(self, issue_key: str, value: str) -> bool:
        """Remove a sub-system group from a single issue."""
        return self.modify_sub_system_group_on_issue(issue_key, value, "remove")

    def replace_sub_system_group_on_issue(self, issue_key: str, value: str) -> bool:
        """Replace the sub-system group on a single issue."""
        return self.modify_sub_system_group_on_issue(issue_key, value, "replace")

    @retry_with_backoff()
    def get_available_transitions(self, issue_key: str) -> List[Dict]:
        """Get available transitions for an issue."""
        url = f"{self.base_url}/rest/api/2/issue/{issue_key}/transitions"

        try:
            response = self.session.get(url)
            if response.status_code == 429:
                raise RateLimitError(
                    f"Rate limited while fetching transitions for {issue_key}"
                )
            self._handle_response_errors(response)
            data = response.json()
            return data.get("transitions", [])
        except (RateLimitError, AuthenticationError, APIError):
            raise
        except Exception as e:
            self.logger.error(
                f"Unexpected error fetching transitions for {issue_key}: {e}"
            )
            return []

    def close_ticket(self, issue_key: str, resolution: str = "Done") -> bool:
        """Close a single ticket with the specified resolution."""
        # First, get available transitions
        transitions = self.get_available_transitions(issue_key)

        if not transitions:
            print(f"  ✗ {issue_key}: No transitions available")
            return False

        # Look for a transition that closes the ticket
        close_transition = None
        for transition in transitions:
            transition_name = transition.get("name", "").lower()
            if any(
                keyword in transition_name
                for keyword in ["close", "done", "resolve", "complete", "finish"]
            ):
                close_transition = transition
                break

        if not close_transition:
            print(f"  ✗ {issue_key}: No closing transition found")
            print(f"    Available transitions: {[t.get('name') for t in transitions]}")
            return False

        # Perform the transition
        url = f"{self.base_url}/rest/api/2/issue/{issue_key}/transitions"
        payload = {
            "transition": {"id": close_transition.get("id")},
            "fields": {"resolution": {"name": resolution}},
        }

        try:
            response = self.session.post(url, json=payload)
            if response.status_code == 429:  # Rate limited
                time.sleep(2)
                response = self.session.post(url, json=payload)

            if response.status_code == 204:  # Success
                print(
                    f"  ✓ {issue_key}: Closed with resolution '{resolution}' via '{close_transition.get('name')}'"
                )
                return True
            else:
                print(f"  ✗ {issue_key}: Failed to close (HTTP {response.status_code})")
                if response.text:
                    print(f"    Error: {response.text}")
                return False

        except requests.RequestException as e:
            print(f"  ✗ {issue_key}: Error closing ticket - {e}")
            return False

    def reopen_ticket(self, issue_key: str) -> bool:
        """Reopen a closed ticket."""
        # First, get available transitions
        transitions = self.get_available_transitions(issue_key)

        if not transitions:
            print(f"  ✗ {issue_key}: No transitions available")
            return False

        # Look for a transition that reopens the ticket
        reopen_transition = None
        for transition in transitions:
            transition_name = transition.get("name", "").lower()
            if any(
                keyword in transition_name
                for keyword in [
                    "reopen",
                    "open",
                    "start",
                    "begin",
                    "activate",
                    "resume",
                    "in progress",
                    "to do",
                ]
            ):
                reopen_transition = transition
                break

        if not reopen_transition:
            print(f"  ✗ {issue_key}: No reopening transition found")
            print(f"    Available transitions: {[t.get('name') for t in transitions]}")
            return False

        # Perform the transition
        url = f"{self.base_url}/rest/api/2/issue/{issue_key}/transitions"
        payload = {"transition": {"id": reopen_transition.get("id")}}

        try:
            response = self.session.post(url, json=payload)
            if response.status_code == 429:  # Rate limited
                time.sleep(2)
                response = self.session.post(url, json=payload)

            if response.status_code == 204:  # Success
                print(
                    f"  ✓ {issue_key}: Reopened via '{reopen_transition.get('name')}'"
                )
                return True
            else:
                print(
                    f"  ✗ {issue_key}: Failed to reopen (HTTP {response.status_code})"
                )
                if response.text:
                    print(f"    Error: {response.text}")
                return False

        except requests.RequestException as e:
            print(f"  ✗ {issue_key}: Error reopening ticket - {e}")
            return False

    def print_hierarchy(
        self,
        issues: List[JiraIssue],
        show_type: bool = False,
        show_status: bool = False,
        show_labels: bool = False,
        show_sub_system_group: bool = False,
    ) -> None:
        """Print the issue hierarchy in a tree format with color coding."""
        if not issues:
            print("No issues found.")
            return

        # Disable colors if not outputting to a terminal
        if not sys.stdout.isatty():
            Colors.disable_colors()

        for issue in issues:
            indent = "    " * issue.level
            if issue.level == 0:
                prefix = ""
            elif issue.level == 1:
                prefix = f"{Colors.BLUE}├─ {Colors.RESET}"
            else:
                prefix = f"{Colors.BLUE}└─ {Colors.RESET}"

            # Add colored status symbol
            is_closed = self.is_issue_closed(issue.status)
            if is_closed:
                status_symbol = f"{Colors.RED}✗{Colors.RESET}"
            else:
                status_symbol = f"{Colors.GREEN}✓{Colors.RESET}"

            # Color the issue key and summary
            colored_key = f"{Colors.BOLD}{issue.key}{Colors.RESET}"
            colored_summary = f"{Colors.CYAN}{issue.summary}{Colors.RESET}"

            # Build the details line
            details = []
            if show_type:
                details.append(f"Type: {issue.issue_type}")
            if show_status:
                status_color = Colors.RED if is_closed else Colors.GREEN
                details.append(f"Status: {status_color}{issue.status}{Colors.RESET}")
            if show_labels and issue.labels:
                colored_labels = (
                    f"{Colors.YELLOW}{', '.join(issue.labels)}{Colors.RESET}"
                )
                details.append(f"Labels: {colored_labels}")
            if show_sub_system_group and issue.sub_system_group:
                details.append(f"Sub-System Group: {Colors.CYAN}{issue.sub_system_group}{Colors.RESET}")

            details_str = (
                f"    {Colors.DIM}[{', '.join(details)}]{Colors.RESET}"
                if details
                else ""
            )

            print(
                f"{indent}{prefix}{status_symbol} {colored_key}: {colored_summary}{details_str}"
            )

    def export_to_json(self, issues: List[JiraIssue], filename: str) -> None:
        """Export the hierarchy to a JSON file."""
        data = [
            {
                "key": issue.key,
                "summary": issue.summary,
                "issue_type": issue.issue_type,
                "status": issue.status,
                "labels": issue.labels,
                "level": issue.level,
                "sub_system_group": issue.sub_system_group,
            }
            for issue in issues
        ]

        with open(filename, "w") as f:
            json.dump(data, f, indent=2)
        print(f"Exported {len(issues)} issues to {filename}")


def create_sample_config(config_path: Optional[str] = None) -> None:
    """Create a sample config file."""
    if config_path is None:
        config_dir = os.path.expanduser("~/.config")
        config_path = os.path.join(config_dir, "jira")
    else:
        config_dir = os.path.dirname(config_path)

    os.makedirs(config_dir, exist_ok=True)

    sample_config = """[jira]
base_url = https://your-company.atlassian.net
username = your-email@company.com
api_token = your-api-token-here

# Alternative format using DEFAULT section:
# [DEFAULT]
# base_url = https://your-company.atlassian.net
# username = your-email@company.com
# api_token = your-api-token-here
"""

    with open(config_path, "w") as f:
        f.write(sample_config)

    print(f"Sample config created at {config_path}")
    print("Please edit the file with your actual Jira credentials.")


def action_create_config(args) -> None:
    """Create a sample config file."""
    create_sample_config(args.config)


def action_test_auth(args) -> None:
    """Test Jira authentication."""
    try:
        config = JiraConfig(args.config)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Use 'create-config' action to create a sample config file.")
        sys.exit(1)

    if not all([config.base_url, config.username, config.api_token]):
        print(
            "Error: Missing required configuration values (base_url, username, api_token)"
        )
        print("Please check your config file")
        sys.exit(1)

    print(f"Testing authentication with {config.base_url}...")
    finder = JiraDescendantFinder(config.base_url, config.username, config.api_token)

    if finder.test_auth():
        print("\n✓ Ready to use Jira API!")
    else:
        print("\n✗ Please check your credentials and try again.")
        sys.exit(1)


def action_add_label(args) -> None:
    """Add a label to an issue and optionally its descendants."""
    logger = logging.getLogger("jira-tcktmngr")

    try:
        config = JiraConfig(args.config)
    except FileNotFoundError as e:
        logger.error(f"Config file not found: {e}")
        print(f"Error: {e}")
        print("Use 'create-config' action to create a sample config file.")
        sys.exit(1)

    if not all([config.base_url, config.username, config.api_token]):
        logger.error("Missing required configuration values")
        print(
            "Error: Missing required configuration values (base_url, username, api_token)"
        )
        print("Please check your config file")
        sys.exit(1)

    finder = JiraDescendantFinder(config.base_url, config.username, config.api_token)

    try:
        if args.include_children:
            logger.info(f"Finding all descendants of {args.issue_key}")
            print(f"Finding all descendants of {args.issue_key}...")
            all_issues = finder.find_descendants(args.issue_key)
            if not all_issues:
                print(f"No issues found for {args.issue_key}")
                return

            # Filter out closed issues
            open_issues = [
                issue
                for issue in all_issues
                if not finder.is_issue_closed(issue.status)
            ]
            closed_issues = [
                issue for issue in all_issues if finder.is_issue_closed(issue.status)
            ]

            if not open_issues:
                print(
                    f"No open issues found for {args.issue_key} (all {len(closed_issues)} issues are closed)"
                )
                return

            logger.info(
                f"Found {len(all_issues)} total issues ({len(open_issues)} open, {len(closed_issues)} closed)"
            )
            print(
                f"\nFound {len(all_issues)} total issues ({len(open_issues)} open, {len(closed_issues)} closed):"
            )
            print("Open issues that will be affected:")
            print("=" * 50)
            finder.print_hierarchy(open_issues, show_labels=True)

            if closed_issues:
                print(f"\nSkipping {len(closed_issues)} closed issues:")
                for issue in closed_issues:
                    print(f"  {issue.key}: {issue.summary} (Status: {issue.status})")

            print(
                f"\nThis will add label '{args.label}' to {len(open_issues)} open issues."
            )
            target_issues = open_issues

        else:
            # Just the single issue
            issue_data = finder.get_issue(args.issue_key)
            if not issue_data:
                print(f"Issue {args.issue_key} not found")
                return

            fields = issue_data.get("fields", {})
            current_labels = fields.get("labels", [])
            issue_status = fields.get("status", {}).get("name", "")

            # Check if single issue is closed
            if finder.is_issue_closed(issue_status):
                print(
                    f"Issue {args.issue_key} is closed (Status: {issue_status}). Skipping label addition."
                )
                return

            print(f"Issue: {args.issue_key}: {fields.get('summary', '')}")
            print(f"Current labels: {current_labels if current_labels else 'None'}")
            print(f"Status: {issue_status}")
            print(f"\nThis will add label '{args.label}' to this issue only.")
            target_issues = [
                finder._create_jira_issue_from_data(args.issue_key, issue_data, 0)
            ]

        # Use bulk operation
        finder._perform_bulk_operation(
            target_issues,
            f"adding label '{args.label}'",
            finder.add_label_to_issue,
            args.label,
        )

    except (AuthenticationError, APIError, RateLimitError) as e:
        logger.error(f"API error in add_label: {e}")
        print(f"Error: {e}")
        sys.exit(1)


def action_remove_label(args):
    """Remove a label from an issue and optionally its descendants."""
    try:
        config = JiraConfig(args.config)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Use 'create-config' action to create a sample config file.")
        sys.exit(1)

    if not all([config.base_url, config.username, config.api_token]):
        print(
            "Error: Missing required configuration values (base_url, username, api_token)"
        )
        print("Please check your config file")
        sys.exit(1)

    finder = JiraDescendantFinder(config.base_url, config.username, config.api_token)

    if args.include_children:
        print(f"Finding all descendants of {args.issue_key}...")
        all_issues = finder.find_descendants(args.issue_key)
        if not all_issues:
            print(f"No issues found for {args.issue_key}")
            return

        # Filter to only issues that actually have the label
        issues_with_label = [
            issue for issue in all_issues if args.label in issue.labels
        ]

        if not issues_with_label:
            print(f"No issues found with label '{args.label}'")
            return

        # Filter out closed issues from those that have the label
        open_issues_with_label = [
            issue
            for issue in issues_with_label
            if not finder.is_issue_closed(issue.status)
        ]
        closed_issues_with_label = [
            issue for issue in issues_with_label if finder.is_issue_closed(issue.status)
        ]

        if not open_issues_with_label:
            print(
                f"No open issues found with label '{args.label}' (all {len(closed_issues_with_label)} issues with the label are closed)"
            )
            return

        print(
            f"\nFound {len(issues_with_label)} total issues with label '{args.label}' ({len(open_issues_with_label)} open, {len(closed_issues_with_label)} closed):"
        )
        print("Open issues that will be affected:")
        print("=" * 50)
        finder.print_hierarchy(open_issues_with_label, show_labels=True)

        if closed_issues_with_label:
            print(
                f"\nSkipping {len(closed_issues_with_label)} closed issues with the label:"
            )
            for issue in closed_issues_with_label:
                print(f"  {issue.key}: {issue.summary} (Status: {issue.status})")

        print(
            f"\nThis will remove label '{args.label}' from {len(open_issues_with_label)} open issues."
        )
        all_issues = open_issues_with_label

    else:
        # Just the single issue
        issue_data = finder.get_issue(args.issue_key)
        if not issue_data:
            print(f"Issue {args.issue_key} not found")
            return

        fields = issue_data.get("fields", {})
        current_labels = fields.get("labels", [])
        issue_status = fields.get("status", {}).get("name", "")

        if args.label not in current_labels:
            print(f"Issue {args.issue_key} does not have label '{args.label}'")
            print(f"Current labels: {current_labels if current_labels else 'None'}")
            return

        # Check if single issue is closed
        if finder.is_issue_closed(issue_status):
            print(
                f"Issue {args.issue_key} is closed (Status: {issue_status}). Skipping label removal."
            )
            return

        print(f"Issue: {args.issue_key}: {fields.get('summary', '')}")
        print(f"Current labels: {current_labels}")
        print(f"Status: {issue_status}")
        print(f"\nThis will remove label '{args.label}' from this issue only.")
        all_issues = [
            JiraIssue(
                key=args.issue_key,
                summary=fields.get("summary", ""),
                issue_type=fields.get("issuetype", {}).get("name", ""),
                status=issue_status,
                labels=current_labels,
                level=0,
            )
        ]

    # Confirmation with affected tickets hierarchy
    print(f"\nAFFECTED TICKETS ({len(all_issues)}):")
    print("=" * 50)
    finder.print_hierarchy(all_issues, show_labels=True)

    response = input(
        f"\nProceed with removing label '{args.label}' from {len(all_issues)} issue(s) above? (y/N): "
    )
    if response.lower() not in ["y", "yes"]:
        print("Cancelled.")
        return

    # Remove the label
    print(f"\nRemoving label '{args.label}'...")
    success_count = 0
    for issue in all_issues:
        if finder.remove_label_from_issue(issue.key, args.label):
            success_count += 1
        time.sleep(0.1)  # Be nice to the API

    print(
        f"\nCompleted: {success_count}/{len(all_issues)} issues updated successfully."
    )


def action_close_ticket(args):
    """Close a ticket and optionally its descendants."""
    try:
        config = JiraConfig(args.config)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Use 'create-config' action to create a sample config file.")
        sys.exit(1)

    if not all([config.base_url, config.username, config.api_token]):
        print(
            "Error: Missing required configuration values (base_url, username, api_token)"
        )
        print("Please check your config file")
        sys.exit(1)

    finder = JiraDescendantFinder(config.base_url, config.username, config.api_token)

    if args.include_children:
        print(f"Finding all descendants of {args.issue_key}...")
        all_issues = finder.find_descendants(args.issue_key)
        if not all_issues:
            print(f"No issues found for {args.issue_key}")
            return

        # Filter to only open issues (skip already closed ones)
        open_issues = [
            issue for issue in all_issues if not finder.is_issue_closed(issue.status)
        ]
        closed_issues = [
            issue for issue in all_issues if finder.is_issue_closed(issue.status)
        ]

        if not open_issues:
            print(
                f"No open issues found for {args.issue_key} (all {len(closed_issues)} issues are already closed)"
            )
            return

        print(
            f"\nFound {len(all_issues)} total issues ({len(open_issues)} open, {len(closed_issues)} closed):"
        )
        print("Open issues that will be closed:")
        print("=" * 50)
        finder.print_hierarchy(open_issues, show_labels=True)

        if closed_issues:
            print(f"\nSkipping {len(closed_issues)} already closed issues:")
            for issue in closed_issues:
                print(f"  {issue.key}: {issue.summary} (Status: {issue.status})")

        print(
            f"\nThis will close {len(open_issues)} open issues with resolution '{args.resolution}'."
        )
        all_issues = open_issues

    else:
        # Just the single issue
        issue_data = finder.get_issue(args.issue_key)
        if not issue_data:
            print(f"Issue {args.issue_key} not found")
            return

        fields = issue_data.get("fields", {})
        issue_status = fields.get("status", {}).get("name", "")

        # Check if single issue is already closed
        if finder.is_issue_closed(issue_status):
            print(f"Issue {args.issue_key} is already closed (Status: {issue_status}).")
            return

        print(f"Issue: {args.issue_key}: {fields.get('summary', '')}")
        print(f"Status: {issue_status}")
        print(f"\nThis will close this issue with resolution '{args.resolution}'.")
        all_issues = [
            JiraIssue(
                key=args.issue_key,
                summary=fields.get("summary", ""),
                issue_type=fields.get("issuetype", {}).get("name", ""),
                status=issue_status,
                labels=fields.get("labels", []),
                level=0,
            )
        ]

    # Confirmation with affected tickets hierarchy
    print(f"\nAFFECTED TICKETS ({len(all_issues)}):")
    print("=" * 50)
    finder.print_hierarchy(all_issues, show_labels=True)

    response = input(
        f"\nProceed with closing {len(all_issues)} issue(s) above with resolution '{args.resolution}'? (y/N): "
    )
    if response.lower() not in ["y", "yes"]:
        print("Cancelled.")
        return

    # Close the tickets
    print(f"\nClosing tickets with resolution '{args.resolution}'...")
    success_count = 0
    for issue in all_issues:
        if finder.close_ticket(issue.key, args.resolution):
            success_count += 1
        time.sleep(0.1)  # Be nice to the API

    print(f"\nCompleted: {success_count}/{len(all_issues)} issues closed successfully.")


def action_reopen_ticket(args):
    """Reopen a single ticket."""
    try:
        config = JiraConfig(args.config)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Use 'create-config' action to create a sample config file.")
        sys.exit(1)

    if not all([config.base_url, config.username, config.api_token]):
        print(
            "Error: Missing required configuration values (base_url, username, api_token)"
        )
        print("Please check your config file")
        sys.exit(1)

    finder = JiraDescendantFinder(config.base_url, config.username, config.api_token)

    # Get the issue details
    issue_data = finder.get_issue(args.issue_key)
    if not issue_data:
        print(f"Issue {args.issue_key} not found")
        return

    fields = issue_data.get("fields", {})
    issue_status = fields.get("status", {}).get("name", "")

    # Check if issue is already open
    if not finder.is_issue_closed(issue_status):
        print(f"Issue {args.issue_key} is already open (Status: {issue_status}).")
        return

    print(f"Issue: {args.issue_key}: {fields.get('summary', '')}")
    print(f"Current Status: {issue_status}")
    print(f"\nThis will reopen this ticket.")

    # Confirmation
    response = input(f"\nProceed with reopening {args.issue_key}? (y/N): ")
    if response.lower() not in ["y", "yes"]:
        print("Cancelled.")
        return

    # Reopen the ticket
    print(f"\nReopening ticket...")
    if finder.reopen_ticket(args.issue_key):
        print(f"\nCompleted: Issue {args.issue_key} reopened successfully.")
    else:
        print(f"\nFailed to reopen issue {args.issue_key}.")


def action_summarize(args):
    """Summarize a Jira issue using Claude AI."""
    try:
        config = JiraConfig(args.config)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Use 'create-config' action to create a sample config file.")
        sys.exit(1)

    if not all([config.base_url, config.username, config.api_token]):
        print(
            "Error: Missing required configuration values (base_url, username, api_token)"
        )
        print("Please check your config file")
        sys.exit(1)

    finder = JiraDescendantFinder(config.base_url, config.username, config.api_token)

    print(f"Summarizing issue {args.issue_key}...")
    success = finder.summarize_issue(args.issue_key)

    if not success:
        sys.exit(1)


def action_find_descendants(args):
    """Find all descendants of a Jira issue."""
    try:
        config = JiraConfig(args.config)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Use 'create-config' action to create a sample config file.")
        sys.exit(1)

    if not all([config.base_url, config.username, config.api_token]):
        print(
            "Error: Missing required configuration values (base_url, username, api_token)"
        )
        print("Please check your config file")
        sys.exit(1)

    finder = JiraDescendantFinder(config.base_url, config.username, config.api_token)

    # Debug what links exist
    if hasattr(args, "debug") and args.debug:
        finder.debug_issue_links(args.issue_key)
        return

    print(f"Finding all descendants of {args.issue_key}...")
    descendants = finder.find_descendants(args.issue_key)

    if not descendants:
        print(f"No descendants found for {args.issue_key}")
        print("Run with --debug to see what links exist:")
        print(f"python jira_descendants.py find {args.issue_key} --debug")
        return

    print(f"\nFound {len(descendants)} issues:")
    print("=" * 50)

    finder.print_hierarchy(
        descendants,
        show_type=args.type,
        show_status=args.status,
        show_labels=args.labels,
        show_sub_system_group=args.sub_system_group,
    )

    if args.export:
        finder.export_to_json(descendants, args.export)

    # Summary
    print(f"\nSummary:")
    open_count = len(
        [issue for issue in descendants if not finder.is_issue_closed(issue.status)]
    )
    closed_count = len(
        [issue for issue in descendants if finder.is_issue_closed(issue.status)]
    )
    print(
        f"Total issues: {len(descendants)} ({open_count} open, {closed_count} closed)"
    )
    max_level = max(issue.level for issue in descendants) if descendants else 0
    print(f"Maximum depth: {max_level}")


def action_edit_sub_system_group(args):
    """Edit the Sub-System group field on an issue and optionally its descendants."""
    try:
        config = JiraConfig(args.config)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Use 'create-config' action to create a sample config file.")
        sys.exit(1)

    if not all([config.base_url, config.username, config.api_token]):
        print(
            "Error: Missing required configuration values (base_url, username, api_token)"
        )
        print("Please check your config file")
        sys.exit(1)

    finder = JiraDescendantFinder(config.base_url, config.username, config.api_token)

    # Determine operation and method
    if args.operation == "append":
        operation_func = finder.add_sub_system_group_to_issue
        operation_name = f"adding sub-system group '{args.value}'"
    elif args.operation == "remove":
        operation_func = finder.remove_sub_system_group_from_issue
        operation_name = f"removing sub-system group '{args.value}'"
    elif args.operation == "replace":
        operation_func = finder.replace_sub_system_group_on_issue
        operation_name = f"setting sub-system group to '{args.value}'"
    else:
        print(f"Error: Invalid operation '{args.operation}'")
        sys.exit(1)

    try:
        if args.include_children:
            print(f"Finding all descendants of {args.issue_key}...")
            all_issues = finder.find_descendants(args.issue_key)
            if not all_issues:
                print(f"No issues found for {args.issue_key}")
                return

            # Filter out closed issues
            open_issues = [
                issue
                for issue in all_issues
                if not finder.is_issue_closed(issue.status)
            ]
            closed_issues = [
                issue for issue in all_issues if finder.is_issue_closed(issue.status)
            ]

            if not open_issues:
                print(
                    f"No open issues found for {args.issue_key} (all {len(closed_issues)} issues are closed)"
                )
                return

            print(
                f"\nFound {len(all_issues)} total issues ({len(open_issues)} open, {len(closed_issues)} closed):"
            )
            print("Open issues that will be affected:")
            print("=" * 50)
            finder.print_hierarchy(open_issues, show_labels=True, show_sub_system_group=True)

            if closed_issues:
                print(f"\nSkipping {len(closed_issues)} closed issues:")
                for issue in closed_issues:
                    print(f"  {issue.key}: {issue.summary} (Status: {issue.status})")

            print(f"\nThis will perform '{operation_name}' on {len(open_issues)} open issues.")
            target_issues = open_issues

        else:
            # Just the single issue
            issue_data = finder.get_issue(args.issue_key)
            if not issue_data:
                print(f"Issue {args.issue_key} not found")
                return

            fields = issue_data.get("fields", {})
            current_sub_system_group = fields.get("customfield_12320851")
            issue_status = fields.get("status", {}).get("name", "")

            # Check if single issue is closed
            if finder.is_issue_closed(issue_status):
                print(
                    f"Issue {args.issue_key} is closed (Status: {issue_status}). Skipping sub-system group modification."
                )
                return

            # Display current sub-system group
            if current_sub_system_group:
                if isinstance(current_sub_system_group, list) and current_sub_system_group:
                    # Handle list of dictionaries (common for multi-select fields)
                    if isinstance(current_sub_system_group[0], dict):
                        current_value = current_sub_system_group[0].get("value", str(current_sub_system_group[0]))
                    else:
                        current_value = str(current_sub_system_group[0])
                elif isinstance(current_sub_system_group, dict):
                    current_value = current_sub_system_group.get("value", str(current_sub_system_group))
                else:
                    current_value = str(current_sub_system_group)
                print(f"Current Sub-System Group: {current_value}")
            else:
                print("Current Sub-System Group: None")

            print(f"Issue: {args.issue_key}: {fields.get('summary', '')}")
            print(f"Status: {issue_status}")
            print(f"\nThis will perform '{operation_name}' on this issue only.")
            target_issues = [
                finder._create_jira_issue_from_data(args.issue_key, issue_data, 0)
            ]

        # Use bulk operation
        finder._perform_bulk_operation(
            target_issues,
            operation_name,
            operation_func,
            args.value,
            show_sub_system_group=True,
        )

    except (AuthenticationError, APIError, RateLimitError) as e:
        print(f"Error: {e}")
        sys.exit(1)


def action_list_sub_system_groups(args):
    """List available Sub-System group options."""
    try:
        config = JiraConfig(args.config)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Use 'create-config' action to create a sample config file.")
        sys.exit(1)

    if not all([config.base_url, config.username, config.api_token]):
        print(
            "Error: Missing required configuration values (base_url, username, api_token)"
        )
        print("Please check your config file")
        sys.exit(1)

    finder = JiraDescendantFinder(config.base_url, config.username, config.api_token)

    try:
        print("Fetching available Sub-System group options...")
        options = finder.get_sub_system_group_options()

        if not options:
            print("No Sub-System group options found or unable to fetch options.")
            print("This might be due to permissions or field configuration.")
            return

        print(f"\nAvailable Sub-System group options ({len(options)}):")
        print("=" * 50)
        for option in options:
            if isinstance(option, dict):
                value = option.get("value", "")
                option_id = option.get("id")
                disabled = option.get("disabled", False)
                status = " (disabled)" if disabled else ""

                # Only show ID if it exists and is not empty
                if option_id:
                    print(f"  {value}{status} (ID: {option_id})")
                else:
                    print(f"  {value}{status}")
            else:
                print(f"  {option}")

    except (AuthenticationError, APIError, RateLimitError) as e:
        print(f"Error: {e}")
        sys.exit(1)


def main() -> None:
    parser = argparse.ArgumentParser(description="Jira descendants finder")
    parser.add_argument(
        "--config", help="Path to config file (default: ~/.config/jira)"
    )
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Set logging level",
    )
    parser.add_argument(
        "--no-timestamp", action="store_true", help="Disable timestamps in log output"
    )

    subparsers = parser.add_subparsers(
        dest="action", required=True, help="Available actions"
    )

    # create-config action
    create_config_parser = subparsers.add_parser(
        "create-config", help="Create a sample config file"
    )
    create_config_parser.set_defaults(func=action_create_config)

    # test-auth action
    test_auth_parser = subparsers.add_parser(
        "test-auth", help="Test Jira authentication"
    )
    test_auth_parser.set_defaults(func=action_test_auth)

    # add-label action
    add_label_parser = subparsers.add_parser(
        "add-label", help="Add a label to an issue and optionally its descendants"
    )
    add_label_parser.add_argument(
        "issue_key", help="The issue key to add label to (e.g., PROJ-123)"
    )
    add_label_parser.add_argument("label", help="The label to add")
    add_label_parser.add_argument(
        "--include-children",
        action="store_true",
        help="Also add label to all descendant issues",
    )
    add_label_parser.set_defaults(func=action_add_label)

    # remove-label action
    remove_label_parser = subparsers.add_parser(
        "remove-label",
        help="Remove a label from an issue and optionally its descendants",
    )
    remove_label_parser.add_argument(
        "issue_key", help="The issue key to remove label from (e.g., PROJ-123)"
    )
    remove_label_parser.add_argument("label", help="The label to remove")
    remove_label_parser.add_argument(
        "--include-children",
        action="store_true",
        help="Also remove label from all descendant issues",
    )
    remove_label_parser.set_defaults(func=action_remove_label)

    # close-ticket action
    close_ticket_parser = subparsers.add_parser(
        "close-ticket", help="Close a ticket and optionally its descendants"
    )
    close_ticket_parser.add_argument(
        "issue_key", help="The issue key to close (e.g., PROJ-123)"
    )
    close_ticket_parser.add_argument(
        "--resolution", default="Done", help="The resolution to set (default: Done)"
    )
    close_ticket_parser.add_argument(
        "--include-children",
        action="store_true",
        help="Also close all descendant issues",
    )
    close_ticket_parser.set_defaults(func=action_close_ticket)

    # reopen-ticket action
    reopen_ticket_parser = subparsers.add_parser(
        "reopen-ticket", help="Reopen a closed ticket"
    )
    reopen_ticket_parser.add_argument(
        "issue_key", help="The issue key to reopen (e.g., PROJ-123)"
    )
    reopen_ticket_parser.set_defaults(func=action_reopen_ticket)

    # find action
    find_parser = subparsers.add_parser("find", help="Find descendants of a Jira issue")
    find_parser.add_argument("issue_key", help="The parent issue key (e.g., PROJ-123)")
    find_parser.add_argument("--export", help="Export results to JSON file")
    find_parser.add_argument(
        "--debug", action="store_true", help="Show debug info about links and subtasks"
    )
    find_parser.add_argument(
        "--type", action="store_true", help="Show issue type in output"
    )
    find_parser.add_argument(
        "--status", action="store_true", help="Show issue status in output"
    )
    find_parser.add_argument(
        "--labels", action="store_true", help="Show issue labels in output"
    )
    find_parser.add_argument(
        "--sub-system-group", action="store_true", help="Show sub-system group in output"
    )
    find_parser.set_defaults(func=action_find_descendants)

    # summarize action
    summarize_parser = subparsers.add_parser(
        "summarize", help="Summarize a Jira issue using Claude AI (requires claude CLI)"
    )
    summarize_parser.add_argument(
        "issue_key", help="The issue key to summarize (e.g., PROJ-123)"
    )
    summarize_parser.set_defaults(func=action_summarize)

    # edit-sub-system-group action
    edit_sub_system_group_parser = subparsers.add_parser(
        "edit-sub-system-group",
        help="Edit the Sub-System group field on an issue and optionally its descendants"
    )
    edit_sub_system_group_parser.add_argument(
        "issue_key", help="The issue key to modify (e.g., PROJ-123)"
    )
    edit_sub_system_group_parser.add_argument(
        "operation",
        choices=["append", "remove", "replace"],
        help="Operation to perform: append (add to existing), remove (remove value), or replace (set to value)"
    )
    edit_sub_system_group_parser.add_argument(
        "value", help="The sub-system group value to add/remove/set"
    )
    edit_sub_system_group_parser.add_argument(
        "--include-children",
        action="store_true",
        help="Also modify sub-system group on all descendant issues",
    )
    edit_sub_system_group_parser.set_defaults(func=action_edit_sub_system_group)

    # list-sub-system-groups action
    list_sub_system_groups_parser = subparsers.add_parser(
        "list-sub-system-groups",
        help="List available Sub-System group options"
    )
    list_sub_system_groups_parser.set_defaults(func=action_list_sub_system_groups)

    args = parser.parse_args()

    # Set up logging
    log_level = LogLevel(args.log_level)
    setup_logging(log_level, not args.no_timestamp)

    logger = logging.getLogger("jira-tcktmngr")
    logger.info(f"Starting jira-tcktmngr with action: {args.action}")

    try:
        args.func(args)
    except KeyboardInterrupt:
        logger.info("Operation cancelled by user")
        print("\nOperation cancelled.")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        print(f"\nError: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
