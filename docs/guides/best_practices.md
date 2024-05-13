# Best Practices

## Designing workflows
- Break down sequences into discrete tasks, even if they could be sent as a single prompt to an LLM. This forces the LLM to output intermediate results, which enhance the quality of the final output. This is akin to implementing "chain of thoughts" or similar techniques, but in a more controllable way.
- Use the task objective to describe the desired output; use task instructions to provide context and constraints. This helps the LLM understand the goal and the constraints it should adhere to.

## Agents
- An agent's `name` and `description` are visible to all other agents; its `instructions` are private and only visible to itself.