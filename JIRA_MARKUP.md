# Jira Markup Quick Reference

Jira uses its own wiki markup syntax (not Markdown). Here's what you need for your ticket templates.

## Headers

```
h1. Biggest Heading
h2. Second Level Heading
h3. Third Level Heading
h4. Fourth Level Heading
h5. Fifth Level Heading
h6. Smallest Heading
```

**Note:** Space after the period is required!

## Lists

### Bullet Lists

```
* Item 1
* Item 2
** Sub-item 2.1
** Sub-item 2.2
* Item 3
```

### Numbered Lists

```
# First item
# Second item
## Sub-item 2.1
## Sub-item 2.2
# Third item
```

## Text Formatting

```
*bold text*
_italic text_
-strikethrough-
+underline+
{{monospace}}
{code}code block{code}
```

## Links

```
[Link text|https://example.com]
[PARENT_PROJECT-123]  (auto-links to ticket)
```

## Tables

```
||Header 1||Header 2||
|Cell 1|Cell 2|
|Cell 3|Cell 4|
```

## Blockquotes

```
{quote}
This is a quote
{quote}
```

## Code Blocks

```
{code:java}
public static void main(String[] args) {
    System.out.println("Hello");
}
{code}
```

Languages: `java`, `javascript`, `python`, `bash`, `sql`, etc.

## Common Template Patterns

### Goal Section

```
h2. Goal
* The goal of this epic is to track...
* Specific objective here
```

### Acceptance Criteria

```
h2. Acceptance criteria
* We have access to vendor documentation
* We can build the component
* Tests pass successfully
```

### Multi-level Structure

```
h2. Feature Overview

Vendor provides user-space software...

h2. Goals

The goal of this feature is to track...

At minimum we must:
* have vendor software works on our platform
* have documentations on how to make it work

As stretch goals we should:
* make it as easy as possible for our users to use these libraries
```

## Why Not Markdown?

Jira predates Markdown and uses its own markup. Some newer Jira instances support Markdown, but the classic wiki markup is more widely compatible.

**Markdown vs Jira:**
- Markdown: `# Header` → Jira: `h1. Header`
- Markdown: `- bullet` → Jira: `* bullet`
- Markdown: `**bold**` → Jira: `*bold*`
- Markdown: `*italic*` → Jira: `_italic_`

## In Your Templates

Your templates use `#` for the parser structure (not sent to Jira):
```
# PARENT_PROJECT Initiative     ← Parser marker (not sent to Jira)
Parent: PARENT_PROJECT-932
Title: My Title
Description:           ← Everything after this goes to Jira
h2. Feature Overview   ← Jira markup header
* Bullet point         ← Jira markup bullet
```

The parser looks for `# PARENT_PROJECT Initiative` and `# EPIC_PROJECT Epic N` to identify sections, but those aren't sent to Jira. Only the content in the `Description:` field uses Jira markup.

## Validation Requirements

The `create_tickets.py` script validates that your descriptions contain:
- `h2. Goal` (or `h1. Goal`)
- `h2. Acceptance criteria` (or `h1. Acceptance criteria`)

This ensures consistent structure across all tickets.

## More Information

Full Jira markup reference: https://jira.atlassian.com/secure/WikiRendererHelpAction.jspa
