# create_tickets.py - Jira Ticket Creator from Templates

Create Jira ticket hierarchies from markdown-like template files with variable substitution and batch processing.

## Quick Start

```bash
# Generate a template to get started
python create_tickets.py --generate-template my_template.txt

# Preview what would be created
python create_tickets.py --vars AREA=Graphics --dry-run

# Actually create the tickets
python create_tickets.py --vars AREA=Graphics

# Batch process multiple areas
python create_tickets.py --vars AREA=Graphics,Audio,Camera
```

## Features

- ✅ **Template-based ticket creation** from markdown-like files
- ✅ **Variable substitution** with `{PLACEHOLDER}` syntax
- ✅ **Batch processing** with comma-separated values
- ✅ **Cartesian product** for multiple variables
- ✅ **Validation with line numbers** for easy error correction
- ✅ **Dry-run mode** to preview before creating
- ✅ **Hierarchical relationships** (PARENT_PROJECT Initiative → EPIC_PROJECT Epics)

## Creating Your Own Template

### Generate a Template

The easiest way to start is to generate a sample template:

```bash
python create_tickets.py --generate-template my_template.txt
```

This creates a template with:
- Proper structure (1 PARENT_PROJECT Initiative + 5 EPIC_PROJECT Epics)
- Example variable placeholders (`{AREA}`, `{FEATURE_NAME}`)
- Correct Jira markup (h2. headers, * bullets)
- Required sections (Goal, Acceptance criteria)

After generation:
1. Edit the template to match your needs
2. Customize variable placeholders
3. Add or remove epic sections
4. Optionally add a `Parent: PROJECT-123` line

### Template Format

Templates use a simple markdown-like format:

```markdown
# PARENT_PROJECT Initiative
Parent: PARENT_PROJECT-932

Title: Enable {PRODUCT} for {AREA} software

Description:
h2. Feature Overview
Vendor provides user-space software...

# Goals
The goal of this feature is to track...

# EPIC_PROJECT Epic 1

Title: Onboarding into {AREA} software

Description:
h2. Goal
* The goal of this epic is to...

h2. Acceptance criteria
* We have access to vendor documentation
```

### Required Fields

**PARENT_PROJECT Initiative:**
- `Title:` (required)
- `Description:` (required)
- `Parent:` (optional, format: `PROJECT-123`)

**EPIC_PROJECT Epics:**
- `Title:` (required)
- `Description:` (required)

**All Descriptions:**
- Must contain `# Goal` section
- Must contain `# Acceptance criteria` section

## Usage

### Basic Usage

```bash
# Single set of tickets
python create_tickets.py --vars AREA=Graphics

# Custom input file
python create_tickets.py --vars AREA=Audio --input my_template.txt

# Different Jira config
python create_tickets.py --vars AREA=Camera --config ~/.jira-prod
```

### Variable Substitution

Variables are specified with `--vars KEY=VALUE`:

```bash
# Single variable
python create_tickets.py --vars AREA=Graphics

# Multiple variables
python create_tickets.py \
  --vars AREA=Graphics \
  --vars VERSION=4.5 \
  --vars PLATFORM="Linux"
```

In your template, use `{AREA}`, `{VERSION}`, `{PLATFORM}`, etc.

### Batch Processing

Use comma-separated values to create multiple ticket sets:

```bash
# Create tickets for 3 areas
python create_tickets.py --vars AREA=Graphics,Audio,Camera

# Creates:
# - Set 1: AREA=Graphics (6 tickets)
# - Set 2: AREA=Audio (6 tickets)
# - Set 3: AREA=Camera (6 tickets)
# Total: 18 tickets
```

### Cartesian Product (Advanced)

Multiple variables with multiple values create all combinations:

```bash
python create_tickets.py \
  --vars AREA=Graphics,Audio \
  --vars VERSION=4.5,4.6

# Creates 4 sets (2 × 2):
# - Set 1: AREA=Graphics, VERSION=4.5
# - Set 2: AREA=Graphics, VERSION=4.6
# - Set 3: AREA=Audio, VERSION=4.5
# - Set 4: AREA=Audio, VERSION=4.6
```

See [BATCH_PROCESSING.md](BATCH_PROCESSING.md) for more details.

### Dry-Run Mode

Preview without creating tickets:

```bash
python create_tickets.py --vars AREA=Graphics,Audio,Camera --dry-run
```

Output shows:
- Validation results
- Batch processing plan
- All tickets that would be created
- No actual API calls made

## Validation

The script validates your template and provides helpful error messages with line numbers:

```
❌ Validation failed with the following errors:

1. Line 2: Invalid parent format: 'BAD-FORMAT' (expected format: PROJECT-123)
  > Parent: BAD-FORMAT
2. Line 15: EPIC_PROJECT Epic 1 is missing 'Title:' field
3. Line 1: Missing '# Goal' sections in descriptions
```

### Validation Rules

1. **Structure:**
   - Must have `# PARENT_PROJECT Initiative` section
   - Can have any number of `# EPIC_PROJECT Epic N` sections (flexible numbering)

2. **Required Fields:**
   - All sections must have `Title:` and `Description:`
   - Parent field (if present) must match `PROJECT-123` format

3. **Content Quality:**
   - All descriptions must contain `# Goal` sections
   - All descriptions must contain `# Acceptance criteria` sections

## Parser Features

The script uses a robust line-by-line parser (not regex):

- ✅ **Clear error messages** with line numbers
- ✅ **Shows problematic content** for context
- ✅ **State machine parsing** (easy to maintain)
- ✅ **Handles edge cases** (trailing whitespace, empty lines, etc.)

See [PARSER_IMPROVEMENTS.md](PARSER_IMPROVEMENTS.md) for technical details.

## Examples

### Example 0: Creating a New Template from Scratch

```bash
# Generate a template
python create_tickets.py --generate-template camera_initiative.txt

# Edit the template (use your favorite editor)
vim camera_initiative.txt

# Test it with dry-run
python create_tickets.py --vars AREA=Camera --vars FEATURE_NAME="Camera Support" --dry-run --input camera_initiative.txt

# Create the tickets
python create_tickets.py --vars AREA=Camera --vars FEATURE_NAME="Camera Support" --input camera_initiative.txt
```

### Example 1: Single Area

```bash
python create_tickets.py --vars AREA=Graphics --dry-run
```

Creates:
- 1 PARENT_PROJECT Initiative: "Enable Graphics support"
- 5 EPIC_PROJECT Epics under that Initiative

### Example 2: Multiple Areas

```bash
python create_tickets.py --vars AREA=Graphics,Audio,Camera
```

Creates:
- 3 PARENT_PROJECT Initiatives (one per area)
- 15 EPIC_PROJECT Epics (5 per area)
- Total: 18 tickets

### Example 3: Complex Template

```bash
python create_tickets.py \
  --vars AREA=Graphics \
  --vars PRODUCT=MyProduct \
  --vars PLATFORM="Linux" \
  --vars VERSION=4.5 \
  --input test/input_multi_vars.txt \
  --dry-run
```

Uses a template with multiple variables for more customization.

## Test Files

Located in `test/` directory:

**Valid Templates:**
- `input_no_parent.txt` - Template without parent field
- `input_single_epic.txt` - Template with just 1 epic
- `input_multi_vars.txt` - Template using multiple variables

**Invalid Templates (for testing validation):**
- `input_broken.txt` - Missing Goal/Acceptance criteria
- `input_bad_parent.txt` - Invalid parent format
- `input_missing_fields.txt` - Missing required fields
- `input_wrong_structure.txt` - No proper structure

See [test/README.md](test/README.md) for details.

## Output

### Success Output

```
======================================================================
Summary for Set 1
======================================================================

✓ Successfully created 6 tickets:

  - PARENT_PROJECT-1234
  - EPIC_PROJECT-5678
  - EPIC_PROJECT-5679
  - EPIC_PROJECT-5680
  - EPIC_PROJECT-5681
  - EPIC_PROJECT-5682

Parent Initiative: PARENT_PROJECT-1234
View hierarchy: python jira-tcktmngr.py find PARENT_PROJECT-1234
```

### Batch Output

When processing multiple sets, you get:
1. Batch summary showing all sets
2. Progress for each set
3. Final summary with all created tickets

## Configuration

Uses the same Jira configuration as `jira-tcktmngr.py`:

**Default:** `~/.config/jira`

```ini
[jira]
base_url = https://your-jira.atlassian.net
username = your-email@example.com
api_token = your-api-token
```

Create config:
```bash
python jira-tcktmngr.py create-config
```

## Integration with jira-tcktmngr.py

After creating tickets, use the main tool to manage them:

```bash
# View the hierarchy
python jira-tcktmngr.py find PARENT_PROJECT-1234

# Add labels to all tickets
python jira-tcktmngr.py add-label PARENT_PROJECT-1234 "gen4.5" --include-children

# Set fix version
python jira-tcktmngr.py add-fix-version PARENT_PROJECT-1234 "4.5.0" --include-children
```

## Troubleshooting

### "Invalid parent format"
Parent must be in format `PROJECT-123` (uppercase letters, hyphen, numbers).

### "Missing '# Goal' sections"
All ticket descriptions must include a `# Goal` section.

### "Missing 'Title:' field"
Every section (PARENT_PROJECT Initiative, EPIC_PROJECT Epics) must have a `Title:` field.

### Template has wrong structure
Make sure your file has:
```markdown
# PARENT_PROJECT Initiative
Parent: ...
Title: ...
Description:
...

# EPIC_PROJECT Epic 1
Title: ...
Description:
...
```

## See Also

- [BATCH_PROCESSING.md](BATCH_PROCESSING.md) - Detailed batch processing guide
- [PARSER_IMPROVEMENTS.md](PARSER_IMPROVEMENTS.md) - Parser technical details
- [test/README.md](test/README.md) - Test file documentation
