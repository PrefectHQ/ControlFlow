# Guide for Creating ControlFlow Examples

This guide outlines the process for creating clear, informative, and consistent examples for ControlFlow documentation. Follow these steps to produce high-quality examples that showcase ControlFlow's features and capabilities.

## 1. Example Structure

Each example should follow this general structure:

```markdown
---
title: [Concise Title]
description: [Brief description of what the example demonstrates]
icon: [FontAwesome icon name]
---

[1-2 sentence introduction explaining the task and its relevance]

## Code

[Brief explanation of what the code does]

```python
[Code block demonstrating the ControlFlow implementation]
```

[Usage examples in a CodeGroup, if applicable, as it creates a single tabbed view]

<CodeGroup>
```python Code
[Full, copyable code for the example, including prints]
```
```text Result
the result, including any intermiediate output that might be helpful
```
</CodeGroup>


ALTERNATIVELY, use a code block for a single example including a commented result

ALTERNATIVELY, use a CodeGroup for multiple examples with commented results



## Key concepts

[Explanation of key ControlFlow features demonstrated in the example]

1. **[Feature Name](/link-to-docs)**: [Brief explanation of the feature]

   ```python
   [Code snippet illustrating the feature]
   ```

[2-3 sentences wrapping up the example and its significance]
```

## 2. Title and Description

- Use a concise, descriptive title that clearly indicates the task or concept being demonstrated.
- Provide a brief (1-2 sentence) description that expands on the title and gives context.
- Choose an appropriate FontAwesome icon that represents the task or concept.

## 3. Introduction

- Begin with a 1-2 sentence introduction that explains the task or concept and its relevance in natural language processing or AI applications.
- If the example demonstrates multiple approaches or variations, briefly mention this.

## 4. Code Section

- Start with a brief explanation of what the code does and how it approaches the task.
- Present the main implementation code in a clear, well-commented Python code block.
- If there are multiple variations or approaches, present them sequentially, explaining the differences between each approach.
- Use type hints and follow PEP 8 style guidelines in the code.
- Import `controlflow as cf` at the beginning of each code block so it can be copied directly.
- Do not create agents unless you are demonstrating a specific feature (e.g. LLM selection, instructions, reusable tools, etc.)
- Try to write code that is as short as possible while still being clear and demonstrating the feature.
- Only use a flow if your tasks need to share history, otherwise just use a single task
- Do not import Dict or List from typing; use builtin dict or list instead

## 5. Usage Examples

- Provide 1-3 usage examples that demonstrate how to use the implemented function(s).
- Use the `<CodeGroup>` component to organize multiple examples.
- Include both the input and the expected output in the examples.
- Choose diverse and relevant examples that showcase different aspects of the implementation.

## 6. Key Concepts

- Identify 3-5 key ControlFlow features or concepts demonstrated in the example (if possible)
- For each concept:
  1. Provide a brief explanation of what the feature does and why it's important.
  2. Include a code snippet that illustrates the feature in use.
  3. Link to the relevant ControlFlow documentation for that feature.
- Arrange the concepts in order of importance or complexity.
- Do not consider controlflow basics like creating or running a task to be key concepts (or even "simple task creation")

## 7. Conclusion

- End with 2-3 sentences that wrap up the example, emphasizing its significance and how it showcases ControlFlow's capabilities.
- Mention any potential extensions or variations of the example that users might consider.

## 8. Style and Tone

- Use clear, concise language throughout the example.
- Maintain a professional and educational tone, avoiding overly casual language.
- Assume the reader has basic programming knowledge but may be new to ControlFlow.
- Use active voice and present tense where possible.

## 9. Consistency

- Ensure consistency in formatting, terminology, and style across all examples.
- Use the same naming conventions for variables and functions across related examples.
- Maintain a consistent level of detail and explanation across examples.

## 10. Review and Refinement

- After creating the example, review it for clarity, correctness, and completeness.
- Ensure all code is functional and produces the expected results.
- Check that all links to ControlFlow documentation are correct and up-to-date.
- Refine the language and explanations to be as clear and concise as possible.

By following this guide, you'll create informative, consistent, and high-quality examples that effectively showcase ControlFlow's features and capabilities. These examples will help users understand how to implement various NLP and AI tasks using ControlFlow, encouraging adoption and proper usage of the framework.