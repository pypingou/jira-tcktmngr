#!/usr/bin/env python3
"""
Script to create Jira ticket hierarchies from the input file.
Uses jira-tcktmngr.py configuration for authentication.
"""

import sys
import re
import json
import os
import configparser
import itertools
from typing import Dict, List, Optional, Tuple

# Import requests for API calls
import requests
from requests.auth import HTTPBasicAuth


class JiraConfig:
    """Configuration reader for Jira credentials (copied from jira-tcktmngr.py)."""

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


class TicketCreator:
    """Creates Jira tickets using the REST API."""

    def __init__(self, base_url: str, username: str, api_token: str):
        self.base_url = base_url.rstrip('/')
        self.session = requests.Session()
        self.session.auth = HTTPBasicAuth(username, api_token)
        self.session.headers.update({
            'Content-Type': 'application/json',
            'Accept': 'application/json'
        })

    def create_issue(
        self,
        project: str,
        issue_type: str,
        summary: str,
        description: str,
        parent_key: Optional[str] = None
    ) -> str:
        """
        Create a Jira issue.

        Args:
            project: Project key (e.g., 'AUTOBU', 'VROOM')
            issue_type: Issue type name (e.g., 'Initiative', 'Epic')
            summary: Issue summary/title
            description: Issue description
            parent_key: Optional parent issue key

        Returns:
            Created issue key (e.g., 'AUTOBU-1234')
        """
        url = f"{self.base_url}/rest/api/2/issue"

        fields = {
            "project": {"key": project},
            "summary": summary,
            "description": description,
            "issuetype": {"name": issue_type}
        }

        # Add parent field if provided
        if parent_key:
            fields["parent"] = {"key": parent_key}

        payload = {"fields": fields}

        print(f"Creating {issue_type} in {project}: {summary[:60]}...")

        response = self.session.post(url, data=json.dumps(payload))

        if response.status_code == 201:
            issue_key = response.json()['key']
            print(f"  ✓ Created: {issue_key}")
            return issue_key
        else:
            print(f"  ✗ Failed: {response.status_code}")
            print(f"  Response: {response.text}")
            raise Exception(f"Failed to create issue: {response.text}")


class ParseError:
    """Represents a parsing error with line number context."""
    def __init__(self, line_num: int, message: str, line_content: str = ""):
        self.line_num = line_num
        self.message = message
        self.line_content = line_content

    def __str__(self):
        if self.line_content:
            return f"Line {self.line_num}: {self.message}\n  > {self.line_content.strip()}"
        return f"Line {self.line_num}: {self.message}"


class TicketSection:
    """Represents a ticket section in the input file."""
    def __init__(self, section_type: str, line_num: int):
        self.section_type = section_type  # 'AUTOBU Initiative' or 'VROOM Epic N'
        self.start_line = line_num
        self.id = None  # Existing ticket ID (e.g., 'AUTOBU-1234')
        self.id_line = None
        self.parent = None
        self.parent_line = None
        self.title = None
        self.title_line = None
        self.description = None
        self.description_line = None
        self.has_goal = False
        self.has_acceptance_criteria = False


def parse_sections(file_path: str) -> tuple[List[TicketSection], List[ParseError]]:
    """
    Parse the input file into ticket sections.

    Args:
        file_path: Path to input file

    Returns:
        Tuple of (sections, errors)
    """
    try:
        with open(file_path, 'r') as f:
            lines = f.readlines()
    except FileNotFoundError:
        return [], [ParseError(0, f"Input file not found: {file_path}")]
    except Exception as e:
        return [], [ParseError(0, f"Error reading input file: {e}")]

    sections = []
    errors = []
    current_section = None
    in_description = False
    description_lines = []

    for line_num, line in enumerate(lines, start=1):
        stripped = line.strip()

        # Check for section headers
        if stripped.startswith('# AUTOBU Initiative'):
            # Save previous section if exists
            if current_section and in_description:
                current_section.description = '\n'.join(description_lines).strip()
                in_description = False
                description_lines = []

            current_section = TicketSection('AUTOBU Initiative', line_num)
            sections.append(current_section)
            continue

        if stripped.startswith('# VROOM Epic '):
            # Save previous section if exists
            if current_section and in_description:
                current_section.description = '\n'.join(description_lines).strip()
                in_description = False
                description_lines = []

            # Extract epic number
            match = re.match(r'# VROOM Epic (\d+)', stripped)
            if match:
                epic_num = match.group(1)
                current_section = TicketSection(f'VROOM Epic {epic_num}', line_num)
                sections.append(current_section)
            else:
                errors.append(ParseError(line_num, "Invalid VROOM Epic header format", stripped))
            continue

        # Parse fields within a section
        if current_section:
            if stripped.startswith('ID:'):
                current_section.id = stripped[3:].strip()
                current_section.id_line = line_num
                in_description = False
            elif stripped.startswith('Parent:'):
                current_section.parent = stripped[7:].strip()
                current_section.parent_line = line_num
                in_description = False
            elif stripped.startswith('Title:'):
                current_section.title = stripped[6:].strip()
                current_section.title_line = line_num
                in_description = False
            elif stripped.startswith('Description:'):
                current_section.description_line = line_num
                in_description = True
                description_lines = []
            elif in_description:
                description_lines.append(line.rstrip())
                # Check for Goal and Acceptance criteria (Jira markup)
                if 'h1. Goal' in line or 'h2. Goal' in line:
                    current_section.has_goal = True
                if 'h1. Acceptance criteria' in line or 'h2. Acceptance criteria' in line:
                    current_section.has_acceptance_criteria = True

    # Save last section's description
    if current_section and in_description:
        current_section.description = '\n'.join(description_lines).strip()

    return sections, errors


def validate_sections(sections: List[TicketSection]) -> List[ParseError]:
    """
    Validate parsed sections.

    Args:
        sections: List of parsed ticket sections

    Returns:
        List of validation errors
    """
    errors = []

    # Check for at least one AUTOBU Initiative
    autobu_sections = [s for s in sections if s.section_type == 'AUTOBU Initiative']
    if not autobu_sections:
        errors.append(ParseError(1, "Missing '# AUTOBU Initiative' section header"))
        return errors  # Can't continue validation without this

    # Validate AUTOBU Initiative
    for autobu in autobu_sections:
        # Validate ID format if present (optional)
        if autobu.id:
            if not re.match(r'^[A-Z]+-\d+$', autobu.id):
                errors.append(ParseError(
                    autobu.id_line,
                    f"Invalid ID format: '{autobu.id}' (expected format: PROJECT-123)",
                    f"ID: {autobu.id}"
                ))

        # Validate parent format if present (optional)
        if autobu.parent:
            if not re.match(r'^[A-Z]+-\d+$', autobu.parent):
                errors.append(ParseError(
                    autobu.parent_line,
                    f"Invalid parent format: '{autobu.parent}' (expected format: PROJECT-123)",
                    f"Parent: {autobu.parent}"
                ))

        # Check required fields (only if ID is not set)
        if not autobu.id:
            if not autobu.title:
                errors.append(ParseError(
                    autobu.start_line,
                    "AUTOBU Initiative is missing 'Title:' field (required unless ID: is specified)"
                ))

            if not autobu.description:
                errors.append(ParseError(
                    autobu.start_line,
                    "AUTOBU Initiative is missing 'Description:' field (required unless ID: is specified)"
                ))

    # Validate VROOM Epics
    vroom_sections = [s for s in sections if s.section_type.startswith('VROOM Epic')]
    for vroom in vroom_sections:
        if not vroom.title:
            errors.append(ParseError(
                vroom.start_line,
                f"{vroom.section_type} is missing 'Title:' field"
            ))

        if not vroom.description:
            errors.append(ParseError(
                vroom.start_line,
                f"{vroom.section_type} is missing 'Description:' field"
            ))

    # Check for Goal and Acceptance criteria across sections that will be created
    # (skip AUTOBU Initiative if it has an ID, since we won't create it)
    sections_to_create = [
        s for s in sections
        if not (s.section_type == 'AUTOBU Initiative' and s.id)
    ]

    has_any_goal = any(s.has_goal for s in sections_to_create)
    has_any_criteria = any(s.has_acceptance_criteria for s in sections_to_create)

    if sections_to_create and not has_any_goal:
        errors.append(ParseError(
            1,
            "Missing 'h2. Goal' sections in descriptions - all tickets should have goals defined (use Jira markup: h2. Goal)"
        ))

    if sections_to_create and not has_any_criteria:
        errors.append(ParseError(
            1,
            "Missing 'h2. Acceptance criteria' sections in descriptions - all tickets should have acceptance criteria (use Jira markup: h2. Acceptance criteria)"
        ))

    return errors


def validate_input_file(file_path: str) -> List[str]:
    """
    Validate the input file structure and return any errors found.

    Args:
        file_path: Path to input file

    Returns:
        List of error messages (empty if valid)
    """
    sections, parse_errors = parse_sections(file_path)

    if parse_errors:
        return [str(e) for e in parse_errors]

    validation_errors = validate_sections(sections)

    return [str(e) for e in validation_errors]


def apply_variables(text: str, variables: Dict[str, str]) -> str:
    """
    Replace variable placeholders in text.

    Args:
        text: Text containing placeholders like {AREA}
        variables: Dictionary of variables to replace

    Returns:
        Text with placeholders replaced
    """
    if not text:
        return text

    for key, value in variables.items():
        placeholder = f"{{{key}}}"
        text = text.replace(placeholder, value)

    return text


def parse_input_file(file_path: str, variables: Dict[str, str]) -> List[Dict]:
    """
    Parse the input file and extract ticket information.

    Args:
        file_path: Path to input file
        variables: Dictionary of variables to replace (e.g., {'AREA': 'Graphics'})

    Returns:
        List of ticket dictionaries with keys: type, title, description, parent
    """
    # Parse sections using the structured parser
    sections, errors = parse_sections(file_path)

    if errors:
        # This shouldn't happen if validation passed, but handle it gracefully
        raise Exception(f"Parse errors: {[str(e) for e in errors]}")

    tickets = []

    # Convert sections to ticket dictionaries
    existing_autobu_id = None
    for section in sections:
        if section.section_type == 'AUTOBU Initiative':
            # If ID is specified, don't create this ticket - just save the ID for linking
            if section.id:
                existing_autobu_id = apply_variables(section.id, variables)
                continue

            tickets.append({
                'project': 'AUTOBU',
                'type': 'Initiative',
                'title': apply_variables(section.title or "", variables),
                'description': apply_variables(section.description or "", variables),
                'parent': apply_variables(section.parent, variables) if section.parent else None,
                'header': 'Initiative',
                'existing_id': None
            })
        elif section.section_type.startswith('VROOM Epic'):
            epic_num = section.section_type.split()[-1]
            tickets.append({
                'project': 'VROOM',
                'type': 'Epic',
                'title': apply_variables(section.title or "", variables),
                'description': apply_variables(section.description or "", variables),
                'parent': None,  # Will be set to AUTOBU Initiative after creation
                'header': f'Epic {epic_num}',
                'existing_id': None
            })

    # If we have an existing AUTOBU ID, add it to the result so caller knows
    if existing_autobu_id:
        # Return existing ID in a way the caller can detect
        return tickets, existing_autobu_id

    return tickets, None


def generate_template(output_path: str) -> None:
    """
    Generate a sample template file.

    Args:
        output_path: Path where the template should be created
    """
    template_content = """# AUTOBU Initiative

Title: {FEATURE_NAME} for {AREA}

Description:
h2. Feature Overview

Describe the high-level feature here. You can use variables like {AREA}, {VERSION}, etc.
that can be replaced using --vars when creating tickets.

h2. Goals

The goal of this feature is to...

At minimum we must:
* First requirement
* Second requirement
* Third requirement

As stretch goals we should:
* Optional enhancement 1
* Optional enhancement 2


# VROOM Epic 1

Title: First Epic for {AREA}

Description:
h2. Goal
* The goal of this epic is to...

h2. Acceptance criteria
* We have completed...
* We can demonstrate...
* All tests pass


# VROOM Epic 2

Title: Second Epic for {AREA}

Description:
h2. Goal
* The goal of this epic is to...

h2. Acceptance criteria
* Acceptance criterion 1
* Acceptance criterion 2
* Acceptance criterion 3


# VROOM Epic 3

Title: Third Epic for {AREA}

Description:
h2. Goal
* The goal of this epic is to...

h2. Acceptance criteria
* Acceptance criterion 1
* Acceptance criterion 2


# VROOM Epic 4

Title: Fourth Epic for {AREA}

Description:
h2. Goal
* The goal of this epic is to...

h2. Acceptance criteria
* Acceptance criterion 1
* Acceptance criterion 2


# VROOM Epic 5

Title: Fifth Epic for {AREA}

Description:
h2. Goal
* The goal of this epic is to...

h2. Acceptance criteria
* Acceptance criterion 1
* Acceptance criterion 2
"""

    # Check if file already exists
    if os.path.exists(output_path):
        response = input(f"File '{output_path}' already exists. Overwrite? [y/N]: ")
        if response.lower() not in ['y', 'yes']:
            print("Template generation cancelled.")
            return

    # Write template
    try:
        with open(output_path, 'w') as f:
            f.write(template_content)
        print(f"✓ Template created: {output_path}")
        print(f"\nNext steps:")
        print(f"1. Edit the template:")
        print(f"   - Customize the content for your use case")
        print(f"   - Add/modify variable placeholders (e.g., {{AREA}}, {{VERSION}})")
        print(f"   - Optionally add 'Parent: AUTOBU-123' line after '# AUTOBU Initiative' if needed")
        print(f"   - Add or remove VROOM Epic sections as needed")
        print(f"2. Create tickets: python create_tickets.py --vars AREA=YourValue --input {output_path}")
        print(f"3. For batch processing: python create_tickets.py --vars AREA=Value1,Value2,Value3 --input {output_path}")
        print(f"\nJira markup reference:")
        print(f"  - Headers: h1. h2. h3. etc.")
        print(f"  - Bullets: * item")
        print(f"  - Bold: *bold text*")
        print(f"  - See JIRA_MARKUP.md for more details")
    except Exception as e:
        print(f"Error creating template: {e}")
        sys.exit(1)


def main():
    """Main execution function."""
    import argparse

    parser = argparse.ArgumentParser(
        description='Create Jira ticket hierarchies from input file',
        epilog='Example: python create_tickets.py --vars AREA=Graphics --vars VERSION=4.5'
    )
    parser.add_argument(
        '--vars',
        action='append',
        metavar='KEY=VALUE',
        help='Variable to replace in template. Use comma-separated values for batch processing (e.g., --vars AREA=Graphics,Audio,Camera creates 3 ticket sets)'
    )
    parser.add_argument(
        '--input',
        default='input',
        help='Path to input file (default: input)'
    )
    parser.add_argument(
        '--config',
        help='Path to Jira config file (default: ~/.config/jira)'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Validate and display tickets that would be created without actually creating them'
    )
    parser.add_argument(
        '--generate-template',
        metavar='FILENAME',
        help='Generate a sample template file and exit'
    )

    args = parser.parse_args()

    # Handle template generation
    if args.generate_template:
        generate_template(args.generate_template)
        return

    # Parse variables from --vars arguments and handle comma-separated values
    variable_options = {}  # Dict[str, List[str]] - each key maps to list of possible values
    if args.vars:
        for var in args.vars:
            if '=' not in var:
                print(f"Error: Invalid variable format '{var}'. Expected KEY=VALUE")
                sys.exit(1)
            key, value = var.split('=', 1)
            key = key.strip()
            # Split by comma to support multiple values
            values = [v.strip() for v in value.split(',')]
            variable_options[key] = values

    # Generate all combinations of variables (cartesian product)
    if variable_options:
        # Get keys and their corresponding value lists
        keys = list(variable_options.keys())
        value_lists = [variable_options[k] for k in keys]

        # Generate all combinations
        variable_sets = [
            dict(zip(keys, combo))
            for combo in itertools.product(*value_lists)
        ]
    else:
        # No variables specified
        variable_sets = [{}]

    # Always validate input file structure first
    print("="*70)
    print("Validating input file structure...")
    print("="*70 + "\n")

    errors = validate_input_file(args.input)

    if errors:
        print("❌ Validation failed with the following errors:\n")
        for i, error in enumerate(errors, 1):
            print(f"{i}. {error}")
        print("\nPlease fix the input file and try again.")
        sys.exit(1)
    else:
        print("✓ Input file structure is valid\n")

    # Display batch information
    if len(variable_sets) > 1:
        print("="*70)
        print(f"Batch Processing: {len(variable_sets)} ticket set(s) will be created")
        print("="*70 + "\n")
        for i, var_set in enumerate(variable_sets, 1):
            print(f"Set {i}:")
            for key, value in var_set.items():
                print(f"  {key} = {value}")
        print()

    # Parse and process each variable set
    all_created_keys = []
    creator = None  # Will be initialized on first non-dry-run

    for batch_num, variables in enumerate(variable_sets, 1):
        if len(variable_sets) > 1:
            print("\n" + "="*70)
            print(f"Processing Set {batch_num}/{len(variable_sets)}")
            print("="*70 + "\n")

        print("="*70)
        print(f"Parsing input file: {args.input}")
        if variables:
            print("Variables:")
            for key, value in variables.items():
                print(f"  {key} = {value}")
        else:
            print("Variables: None (placeholders will remain unchanged)")
        print("="*70 + "\n")

        tickets, existing_autobu_id = parse_input_file(args.input, variables)

        if not tickets and not existing_autobu_id:
            print("No tickets found in input file!")
            sys.exit(1)

        if existing_autobu_id:
            print(f"Using existing AUTOBU Initiative: {existing_autobu_id}\n")

        print(f"Found {len(tickets)} tickets to create:\n")
        for i, ticket in enumerate(tickets, 1):
            print(f"{i}. [{ticket['project']}] {ticket['type']}: {ticket['title']}")
            if ticket['parent']:
                print(f"   Parent: {ticket['parent']}")

        if args.dry_run:
            print("\n--dry-run mode: Not creating tickets for this set.")
            continue

        print("\n" + "="*70)
        print("Creating tickets in Jira...")
        print("="*70 + "\n")

        # Load Jira configuration (only once on first actual creation)
        if creator is None:
            config = JiraConfig(args.config)
            creator = TicketCreator(
                base_url=config.base_url,
                username=config.username,
                api_token=config.api_token
            )

        # Create tickets for this set
        created_keys = []
        autobu_initiative_key = existing_autobu_id  # Use existing ID if provided

        for ticket in tickets:
            # Determine parent
            parent_key = ticket['parent']

            # If this is a VROOM epic and we have an AUTOBU initiative (created or existing),
            # link it to that initiative
            if ticket['project'] == 'VROOM' and autobu_initiative_key:
                parent_key = autobu_initiative_key

            # Create the issue
            issue_key = creator.create_issue(
                project=ticket['project'],
                issue_type=ticket['type'],
                summary=ticket['title'],
                description=ticket['description'],
                parent_key=parent_key
            )

            created_keys.append(issue_key)

            # Save the AUTOBU initiative key for linking VROOM epics (only if we created it)
            if ticket['project'] == 'AUTOBU':
                autobu_initiative_key = issue_key

        print("\n" + "="*70)
        print(f"Summary for Set {batch_num}")
        print("="*70)
        print(f"\n✓ Successfully created {len(created_keys)} tickets:\n")
        for key in created_keys:
            print(f"  - {key}")

        if autobu_initiative_key:
            if existing_autobu_id:
                print(f"\nLinked to existing Initiative: {autobu_initiative_key}")
            else:
                print(f"\nParent Initiative: {autobu_initiative_key}")
            print(f"View hierarchy: python jira-tcktmngr.py find {autobu_initiative_key}")

        all_created_keys.extend(created_keys)

    # Final summary if batch processing
    if len(variable_sets) > 1 and not args.dry_run:
        print("\n" + "="*70)
        print("FINAL SUMMARY")
        print("="*70)
        print(f"\n✓ Successfully created {len(all_created_keys)} tickets across {len(variable_sets)} set(s):\n")
        for key in all_created_keys:
            print(f"  - {key}")


if __name__ == '__main__':
    main()
