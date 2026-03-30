# Batch Processing with create_tickets.py

The script supports batch processing to create multiple ticket sets in a single run using comma-separated values.

## Basic Batch Processing

### Single Variable, Multiple Values

Create tickets for Graphics, Audio, and Camera in one run:

```bash
python create_tickets.py --vars AREA=Graphics,Audio,Camera
```

This creates **3 sets** of tickets:
- Set 1: Graphics (6 tickets)
- Set 2: Audio (6 tickets)
- Set 3: Camera (6 tickets)

**Total: 18 tickets**

### Dry-Run Mode

Preview what would be created without actually creating tickets:

```bash
python create_tickets.py --vars AREA=Graphics,Audio,Camera --dry-run
```

Output:
```
======================================================================
Batch Processing: 3 ticket set(s) will be created
======================================================================

Set 1:
  AREA = Graphics
Set 2:
  AREA = Audio
Set 3:
  AREA = Camera

...processes each set...
```

## Advanced: Cartesian Product

### Multiple Variables with Multiple Values

When you specify multiple variables with multiple values each, the script creates **all combinations** (cartesian product):

```bash
python create_tickets.py \
  --vars AREA=Graphics,Audio \
  --vars VERSION=4.5,4.6 \
  --dry-run
```

This creates **4 sets** of tickets:
- Set 1: AREA=Graphics, VERSION=4.5
- Set 2: AREA=Graphics, VERSION=4.6
- Set 3: AREA=Audio, VERSION=4.5
- Set 4: AREA=Audio, VERSION=4.6

**Total: 24 tickets (6 tickets × 4 combinations)**

### Example with 3 Variables

```bash
python create_tickets.py \
  --vars AREA=Graphics,Audio \
  --vars PLATFORM="Linux","MyProduct" \
  --vars VERSION=4.5,4.6 \
  --input test/input_multi_vars.txt \
  --dry-run
```

This creates **8 sets** (2 × 2 × 2):
1. Graphics, Linux, 4.5
2. Graphics, Linux, 4.6
3. Graphics, MyProduct, 4.5
4. Graphics, MyProduct, 4.6
5. Audio, Linux, 4.5
6. Audio, Linux, 4.6
7. Audio, MyProduct, 4.5
8. Audio, MyProduct, 4.6

## Single Value (No Batch)

If you only provide a single value, it works normally without batch processing overhead:

```bash
python create_tickets.py --vars AREA=Graphics
```

Output:
```
======================================================================
Parsing input file: input
Variables:
  AREA = Graphics
======================================================================

Found 6 tickets to create:
...
```

No "Batch Processing" header is shown for a single set.

## Output Format

### Batch Processing Output

When creating multiple sets, the output shows:

1. **Validation** (once for the template)
2. **Batch Summary** (shows all sets that will be created)
3. **For each set:**
   - Set header (e.g., "Processing Set 2/3")
   - Variables for this set
   - Tickets found
   - Creation progress
   - Summary with ticket keys
4. **Final Summary** (total tickets across all sets)

### Example Output Structure

```
======================================================================
Validating input file structure...
======================================================================
✓ Input file structure is valid

======================================================================
Batch Processing: 3 ticket set(s) will be created
======================================================================
Set 1:
  AREA = Graphics
Set 2:
  AREA = Audio
Set 3:
  AREA = Camera

======================================================================
Processing Set 1/3
======================================================================
...creates 6 tickets...

======================================================================
Summary for Set 1
======================================================================
✓ Successfully created 6 tickets:
  - PARENT_PROJECT-1234
  - EPIC_PROJECT-5678
  ...

======================================================================
Processing Set 2/3
======================================================================
...creates 6 tickets...

======================================================================
Processing Set 3/3
======================================================================
...creates 6 tickets...

======================================================================
FINAL SUMMARY
======================================================================
✓ Successfully created 18 tickets across 3 set(s):
  - PARENT_PROJECT-1234
  - EPIC_PROJECT-5678
  ...
  (all 18 tickets listed)
```

## Use Cases

### 1. Multiple Areas for Same Project

Create initiatives for all areas in Gen 4.5:

```bash
python create_tickets.py --vars AREA=Graphics,Audio,Camera,Video,ML
```

### 2. Same Area Across Multiple Versions

Track Graphics across multiple releases:

```bash
python create_tickets.py \
  --vars AREA=Graphics \
  --vars VERSION=4.5,4.6,5.0 \
  --input version_template.txt
```

### 3. Matrix of Areas and Platforms

```bash
python create_tickets.py \
  --vars AREA=Graphics,Audio,Camera \
  --vars PLATFORM="Linux","MyProduct" \
  --input platform_template.txt
```

Creates 6 sets (3 areas × 2 platforms)

## Tips

1. **Start with --dry-run** to preview what will be created
2. **Watch API rate limits** - creating many tickets quickly may hit Jira rate limits
3. **Use descriptive variable names** in your template for clarity
4. **Check ticket hierarchy** after creation using `python jira-tcktmngr.py find PARENT_PROJECT-XXXX`
