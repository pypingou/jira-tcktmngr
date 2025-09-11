# Jira Ticket Manager (jira-tcktmngr)

A comprehensive command-line tool for managing Jira tickets and their hierarchies. This tool provides functionality for finding descendants, managing labels, closing/reopening tickets, and working with subtasks and issue links.

## Features

- **Find Descendants**: Recursively discover all child issues, subtasks, and linked tickets
- **Label Management**: Add or remove labels from issues and their descendants
- **Ticket State Management**: Close or reopen tickets individually or in bulk
- **Hierarchy Visualization**: Display ticket hierarchies in a colored tree format
- **Authentication Testing**: Verify Jira API credentials
- **Export Capabilities**: Export hierarchies to JSON format
- **Debug Mode**: Analyze issue links and relationships
- **AI Summarization**: Generate concise summaries of tickets using Claude AI (requires claude CLI)

## Requirements

- Python 3.6+
- `requests` library
- Valid Jira API credentials (username and API token)
- Optional: `claude` CLI for AI summarization features

## Configuration

The tool requires a configuration file with your Jira credentials. Create one using:

```bash
python jira-tcktmngr.py create-config
```

This creates a config file at `~/.config/rh_jira` with the following format:

```ini
[jira]
base_url = https://your-company.atlassian.net
username = your-email@company.com
api_token = your-api-token-here
```

## Usage

### Test Authentication
```bash
python jira-tcktmngr.py test-auth
```

### Find Descendants
```bash
# Basic hierarchy display
python jira-tcktmngr.py find PROJ-123

# With additional details
python jira-tcktmngr.py find PROJ-123 --type --status --labels

# Debug mode to see all links and relationships
python jira-tcktmngr.py find PROJ-123 --debug

# Export to JSON
python jira-tcktmngr.py find PROJ-123 --export hierarchy.json
```

### Label Management
```bash
# Add label to a single issue
python jira-tcktmngr.py add-label PROJ-123 "needs-review"

# Add label to issue and all descendants
python jira-tcktmngr.py add-label PROJ-123 "needs-review" --include-children

# Remove label from a single issue
python jira-tcktmngr.py remove-label PROJ-123 "needs-review"

# Remove label from issue and all descendants
python jira-tcktmngr.py remove-label PROJ-123 "needs-review" --include-children
```

### Ticket State Management
```bash
# Close a single ticket
python jira-tcktmngr.py close-ticket PROJ-123

# Close ticket with custom resolution
python jira-tcktmngr.py close-ticket PROJ-123 --resolution "Won't Fix"

# Close ticket and all descendants
python jira-tcktmngr.py close-ticket PROJ-123 --include-children

# Reopen a closed ticket
python jira-tcktmngr.py reopen-ticket PROJ-123
```

### AI Summarization
```bash
# Generate a concise summary of a ticket using Claude AI
python jira-tcktmngr.py summarize PROJ-123
```

**Note**: The summarize feature requires the `claude` CLI tool to be installed and available in your PATH. The tool will automatically check for availability and provide instructions if not found.

## How It Works

The tool discovers issue relationships through multiple mechanisms:

1. **Subtasks**: Direct parent-child relationships
2. **Issue Links**: "contains", "child", "subtask" relationships
3. **Epic Links**: Issues belonging to an Epic
4. **Custom Fields**: Feature Links, Initiative Links, and other custom field relationships

The hierarchy is displayed with color-coded output:
- ✓ Green checkmark for open issues
- ✗ Red X for closed issues
- Hierarchical tree structure with indentation
- Color-coded summaries and labels

## Safety Features

- **Confirmation prompts** before bulk operations
- **Closed issue filtering** (operations skip closed tickets by default)
- **Preview mode** showing exactly which tickets will be affected
- **Rate limiting** to respect Jira API limits
- **Error handling** with descriptive messages

## Configuration File Location

By default, the tool looks for configuration at `~/.config/rh_jira`. You can specify an alternative location using:

```bash
python jira-tcktmngr.py --config /path/to/config action
```

## License

This software is provided under the BSD 3-Clause License. See the license header in the source code for full details.