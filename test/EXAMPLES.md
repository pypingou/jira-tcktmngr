# create_tickets.py - Usage Examples

Real-world examples showing different use cases.

## Example 1: First Time User

Generate a template and create tickets:

```bash
# 1. Generate your first template
python create_tickets.py --generate-template my_first_template.txt

# 2. (Edit the template in your editor)

# 3. Test it with dry-run
python create_tickets.py \
  --vars AREA=Graphics \
  --vars FEATURE_NAME="Graphics Pipeline" \
  --dry-run \
  --input my_first_template.txt

# 4. Create the tickets
python create_tickets.py \
  --vars AREA=Graphics \
  --vars FEATURE_NAME="Graphics Pipeline" \
  --input my_first_template.txt
```

## Example 2: Batch Create for Multiple Areas

Create initiatives for Graphics, Audio, and Camera in one run:

```bash
python create_tickets.py \
  --vars AREA=Graphics,Audio,Camera \
  --vars FEATURE_NAME="Vendor Integration" \
  --input my_template.txt
```

This creates:
- 3 PARENT_PROJECT Initiatives (one per area)
- 15 EPIC_PROJECT Epics (5 per area)
- **Total: 18 tickets**

## Example 3: Multi-Dimensional Matrix

Create tickets for all combinations of areas and versions:

```bash
python create_tickets.py \
  --vars AREA=Graphics,Audio \
  --vars VERSION=4.5,4.6,5.0 \
  --input version_template.txt
```

This creates:
- 6 combinations (2 areas × 3 versions)
- 36 tickets total (6 initiatives + 30 epics)

## Example 4: Using Existing Template

Use the included production template:

```bash
# Single area
python create_tickets.py --vars AREA=Graphics

# Multiple areas
python create_tickets.py --vars AREA=Graphics,Audio,Camera

# With dry-run first
python create_tickets.py --vars AREA=Graphics --dry-run
```

## Example 5: Custom Parent Relationship

Create a sub-initiative under an existing initiative:

1. Edit your template:
```
# PARENT_PROJECT Initiative
Parent: PARENT_PROJECT-932

Title: {AREA} Support for Gen 4.5
...
```

2. Create tickets:
```bash
python create_tickets.py --vars AREA=Graphics --input custom_template.txt
```

All EPIC_PROJECT epics will be linked to the new initiative, which is linked to PARENT_PROJECT-932.

## Example 6: Minimal Template (Just 1 Epic)

Generate and customize for simple use cases:

```bash
# 1. Generate template
python create_tickets.py --generate-template simple.txt

# 2. Edit simple.txt - remove Epic 2-5, keep only Epic 1

# 3. Use it
python create_tickets.py --vars AREA=Hotfix --input simple.txt
```

Creates just 2 tickets: 1 initiative + 1 epic.

## Example 7: Validation and Error Checking

Check your template before creating tickets:

```bash
# Dry-run shows what would be created
python create_tickets.py --vars AREA=Test --dry-run --input my_template.txt

# Validation errors show line numbers:
# ❌ Line 15: EPIC_PROJECT Epic 1 is missing 'Title:' field
# ❌ Line 2: Invalid parent format: 'bad-format' (expected PROJECT-123)
```

## Example 8: Complex Multi-Variable Template

For sophisticated use cases with many variables:

```bash
python create_tickets.py \
  --vars AREA=Graphics \
  --vars PRODUCT=MyProduct \
  --vars VERSION=4.5 \
  --vars PLATFORM="Linux" \
  --vars CUSTOMER="Acme Corp" \
  --input enterprise_template.txt
```

Template can use all variables:
```
Title: {PRODUCT} {VERSION} - {AREA} for {CUSTOMER} on {PLATFORM}
```

## Example 9: Workflow Integration

Integrate with git workflow:

```bash
# Create feature branch
git checkout -b feature/graphics-gen45

# Generate template
python create_tickets.py --generate-template graphics_gen45.txt

# Edit and commit template
git add graphics_gen45.txt
git commit -m "Add Graphics Gen 4.5 initiative template"

# Create tickets
python create_tickets.py --vars AREA=Graphics --input graphics_gen45.txt

# Capture ticket keys
# (Output shows: PARENT_PROJECT-1234, EPIC_PROJECT-5678, etc.)

# Add to project documentation
echo "Initiative: PARENT_PROJECT-1234" >> docs/gen45_tracking.md
git add docs/gen45_tracking.md
git commit -m "Add Graphics Gen 4.5 initiative ticket"
```

## Example 10: Team Collaboration

Share templates across team:

```bash
# Create shared template directory
mkdir -p templates/initiatives/

# Generate template
python create_tickets.py --generate-template templates/initiatives/qc_integration.txt

# Team member 1 creates Graphics tickets
python create_tickets.py \
  --vars AREA=Graphics \
  --input templates/initiatives/qc_integration.txt

# Team member 2 creates Audio tickets
python create_tickets.py \
  --vars AREA=Audio \
  --input templates/initiatives/qc_integration.txt

# Same structure, different content!
```

## Example 11: After Creation - Managing Tickets

After creating tickets with create_tickets.py, use jira-tcktmngr.py to manage them:

```bash
# View the hierarchy
python jira-tcktmngr.py find PARENT_PROJECT-1234

# Add labels to all tickets
python jira-tcktmngr.py add-label PARENT_PROJECT-1234 "gen4.5" --include-children

# Set fix version
python jira-tcktmngr.py add-fix-version PARENT_PROJECT-1234 "4.5.0" --include-children

# Bulk close when done
python jira-tcktmngr.py close-ticket PARENT_PROJECT-1234 --include-children
```

## Troubleshooting Examples

### Fix: Invalid Parent Format

```bash
# Error: Line 2: Invalid parent format: 'PARENT_PROJECT-XXX'

# Fix 1: Use valid parent
sed -i 's/Parent: PARENT_PROJECT-XXX/Parent: PARENT_PROJECT-932/' my_template.txt

# Fix 2: Remove parent line entirely
sed -i '/^Parent:/d' my_template.txt
```

### Fix: Missing Goal Section

```bash
# Error: Missing 'h2. Goal' sections

# Add to all epic descriptions in template:
h2. Goal
* The goal of this epic is to...
```

### Fix: Wrong Markup

```bash
# Error shows numbered lists instead of headers in Jira

# Wrong (Markdown):
# Goal

# Correct (Jira):
h2. Goal
```

## See Also

- [CREATE_TICKETS_README.md](../CREATE_TICKETS_README.md) - Full documentation
- [TEMPLATE_GENERATION.md](../TEMPLATE_GENERATION.md) - Template guide
- [BATCH_PROCESSING.md](../BATCH_PROCESSING.md) - Batch processing details
- [JIRA_MARKUP.md](../JIRA_MARKUP.md) - Jira markup reference
