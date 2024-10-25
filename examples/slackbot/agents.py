import os
import re
from typing import Annotated

from pydantic import BaseModel, Field
from tools import search_internet, search_knowledge_base

import controlflow as cf


def _strip_app_mention(text: str) -> str:
    return re.sub(r"<@[A-Z0-9]+>", "", text).strip()


class SearchResult(BaseModel):
    """Individual search result with source and relevance"""

    content: str
    source: str
    relevance_score: float = Field(
        ge=0.0,
        le=1.0,
        description="A score indicating the relevance of the search result to the user's question",
    )


class ExplorerFindings(BaseModel):
    """Collection of search results with metadata"""

    search_query: str

    results: list[SearchResult] = Field(default_factory=list)
    total_results: int = Field(
        ge=0,
        description="The total number of search results found",
    )


class RefinedContext(BaseModel):
    """Final refined context after auditing"""

    relevant_content: str
    confidence_score: float = Field(
        ge=0.0,
        le=1.0,
        description="A score indicating the confidence in the relevance of the relevant content to the user's question",
    )
    reasoning: str


bouncer = cf.Agent(
    name="Bouncer",
    instructions=(
        "You are a gatekeeper. You are responsible for determining whether the user's question is appropriate for the system. "
        "If the user asks a legitimate question about Prefect, let them through. If its conversational, or not about Prefect, "
        "do not let them through. Tend towards giving the benefit of the doubt, since sometimes there are language barriers."
    ),
)

explorer = cf.Agent(
    name="Explorer",
    instructions=(
        "You are a thorough researcher. Use the knowledgebase and the internet to find "
        "documentation and code snippets related to Prefect. The knowledgebase is curated, "
        "so it should be preferred over the internet for finding information, but the internet "
        "should be used to supplement the knowledgebase when it doesn't contain the desired information."
    ),
    tools=[search_knowledge_base, search_internet],
)

auditor = cf.Agent(
    name="Auditor",
    instructions=(
        "You are a critical evaluator. Assess relevance and provide clear reasoning. "
        "The explorer is only responsible for finding idiosyncratic information related "
        "to Prefect and tangentially related information that might be in docs or the internet. "
        "You are responsible for determining whether the explorer found the requisite Prefect information, "
        "do not require that the explorer find common knowledge, only idiosyncratic Prefect information."
        "If the user asks how to do something, or for an example, the explorer must find an exact example or "
        "enough information for a downstream agent to extrapolate an example."
    ),
)

liaison = cf.Agent(
    name="Marvin",
    instructions=(
        "You are a liaison between the user and the system. You are responsible for "
        "summarizing the findings and providing a clear and concise answer. "
        "Speak with the tone of a normal support agent, but subtly in the style of Marvin the Paranoid Android. "
        "If their reply is unserious, feel free to riff on their message and be more playful (in character)."
    ),
)


async def search_knowledgebase_and_refine_context(
    user_text: str, memories: list[cf.Memory]
) -> str:
    clean_user_text = _strip_app_mention(user_text)
    if clean_user_text.startswith(os.getenv("SUDO_PREFIX", "___")):
        return cf.run(
            objective="Answer the question to the best of your ability",
            instructions="We have entered test mode. Ignore previous instructions and answer the question to the best of your ability.",
            result_type=str,
            context={
                "user_text": user_text,
                "test_mode": True,
                "personality": "None. You are a helpful assistant.",
            },
            memories=memories,
            agents=[liaison],
        )

    is_a_serious_question = await cf.run_async(
        objective="Determine if the user's question is a serious question about Prefect",
        result_type=bool,
        agents=[bouncer],
        context={"user_question": clean_user_text},
    )

    findings = None
    supporting_context_is_insufficient = True

    while supporting_context_is_insufficient and is_a_serious_question:
        findings = await cf.run_async(
            objective="Search through available sources to find relevant information about this query",
            result_type=ExplorerFindings,
            context={"query": clean_user_text},
            agents=[explorer],
        )

        supporting_context_is_insufficient = await cf.run_async(
            objective="Review and assess the relevance of search results to the user's question",
            result_type=Annotated[
                bool,
                Field(
                    description="Whether the search results are insufficient to answer the user's question"
                ),
            ],
            context={"findings": findings, "user_question": clean_user_text},
            agents=[auditor],
        )

    relevant_context = {"user_question": clean_user_text}

    relevant_context |= {"findings": findings} if findings else {"just_riffing": True}

    return cf.run(
        objective="Compose a final answer to the user's question.",
        instructions=(
            "Provide links to any relevant sources. The answer should address the user directly, NOT discuss the user"
        ),
        result_type=str,
        context=relevant_context,
        agents=[liaison],
        memories=memories,
    )
