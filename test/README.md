# Test Input Files

This directory contains test input files for validating the `create_tickets.py` script.

## Valid Input Files

### `input_no_parent.txt`
A minimal valid template without a parent field (for top-level initiatives).

**Usage:**
```bash
python create_tickets.py --vars AREA=Audio --dry-run --input test/input_no_parent.txt
```

### `input_multi_vars.txt`
Demonstrates using multiple variables in a template.

**Usage:**
```bash
python create_tickets.py \
  --vars AREA=Graphics \
  --vars PRODUCT=MyProduct \
  --vars PLATFORM="Linux" \
  --vars VERSION=4.5 \
  --dry-run \
  --input test/input_multi_vars.txt
```

### `input_single_epic.txt`
Shows that you can have any number of epics (not limited to 5).

**Usage:**
```bash
python create_tickets.py --vars AREA=Camera --dry-run --input test/input_single_epic.txt
```

## Invalid Input Files (for testing validation)

### `input_broken.txt`
Missing required Goal and Acceptance criteria sections in descriptions.

**Expected result:** Validation fails with missing content errors.

```bash
python create_tickets.py --vars AREA=Test --dry-run --input test/input_broken.txt
```

### `input_bad_parent.txt`
Has an invalid parent format (should be `PROJECT-123`).

**Expected result:** Validation fails with parent format error and line number.

```bash
python create_tickets.py --vars AREA=Test --dry-run --input test/input_bad_parent.txt
```

### `input_missing_fields.txt`
Missing Title field in PARENT_PROJECT Initiative and EPIC_PROJECT Epic 1.

**Expected result:** Validation fails with specific line numbers for missing fields.

```bash
python create_tickets.py --vars AREA=Test --dry-run --input test/input_missing_fields.txt
```

### `input_wrong_structure.txt`
Completely invalid file with no proper sections.

**Expected result:** Validation fails immediately with missing PARENT_PROJECT section error.

```bash
python create_tickets.py --vars AREA=Test --dry-run --input test/input_wrong_structure.txt
```

### `input_detailed_errors.txt`
Multiple error types in one file (invalid parent, missing title, missing description).

**Expected result:** Shows multiple validation errors with line numbers.

```bash
python create_tickets.py --vars AREA=Test --dry-run --input test/input_detailed_errors.txt
```

## Validation Rules

The validator checks for:

1. **PARENT_PROJECT Initiative section** with:
   - `Title:` field (required)
   - `Description:` field (required)
   - `Parent:` field (optional, but if present must match format `PROJECT-123`)

2. **EPIC_PROJECT Epic sections** (any number, any numbering), each with:
   - `Title:` field (required)
   - `Description:` field (required)

3. **Required content** (blocks if missing):
   - `# Goal` sections must be present in descriptions
   - `# Acceptance criteria` sections must be present in descriptions
