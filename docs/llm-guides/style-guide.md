# ControlFlow Documentation Style Guide

This style guide ensures clear, consistent, and maintainable documentation for ControlFlow. It is primarily aimed at LLM agents that assist with writing documentation, but it may also be useful for other contributors.

## General Guidelines

- Use consistent terminology throughout the documentation. Always refer to the library as "ControlFlow".
- Link to related concepts, patterns, or API references when appropriate to help users navigate the documentation.
- Maintain a professional, technical tone. Avoid marketing language, hyperbole, or casual phrasing; this is technical documentation, not a blog.
- Write concisely and directly, focusing on technical accuracy and clarity.
- Do not end documentation with "Conclusions", "Best Practices", or other summary lists. Documentation is not a blog post.

## Code Examples

- Use `import controlflow as cf` instead of importing top-level classes and functions directly.
- Code examples should be complete, including all necessary imports, so that users can copy and paste them directly.
- For illustrative examples, provide simple, focused examples that demonstrate a specific concept or pattern.
- For "full" examples, provide realistic, practical examples that demonstrate actual use cases of ControlFlow.

## Tasks

- Ensure that example code reflects best practices for task definition, including suitable result types and instructions.
- Each task should have clear, unambiguous instructions, particularly when the task name doesn't fully convey the expected outcome.
- If placeholder tasks are required in examples, consider using a string result type with a comment to denote it's a placeholder, e.g., `result_type=str # Placeholder for actual result`.
- The default `result_type` is `str`, so there's no need to provide it explicitly for string results.

## Comparisons and Context

- When explaining new features or concepts, compare them to existing ControlFlow functionality rather than external concepts or "traditional" approaches.
- Frame new features as extensions or enhancements to existing ControlFlow capabilities.

## Documentation Structure

- Begin each major section with a brief introduction explaining its purpose and relevance to ControlFlow.
- Use clear, descriptive headings and subheadings to organize content logically.
- Provide code examples that demonstrate both basic and advanced usage of features.
- Avoid lengthy conclusions or summary sections. The documentation should focus on providing clear, actionable information.

## Mintlify-Specific Guidelines

- Mintlify components expect newlines before and after tags, e.g., `<Tip>\nThis is a tip\n</Tip>`.
- Mintlify displays the page's title as an H1 element, so there's no need to add an initial top-level header to any doc. Instead, add the title to the doc's frontmatter, e.g., `---\ntitle: My Title\n---`.
- Because the title is displayed as an H1, all subsequent headers should be H2 or lower.
- Use sentence case for all headers except the page title, e.g. `## Running your tasks` instead of `## Running Your Tasks`.