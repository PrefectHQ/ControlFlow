from typing import Union

import pytest
from langchain_openai import AzureChatOpenAI, ChatOpenAI
from pydantic import BaseModel

from controlflow.llm.rules import OpenAIRules


class OpenAIFirst(BaseModel):
    model: Union[ChatOpenAI, AzureChatOpenAI]


class AzureFirst(BaseModel):
    model: Union[AzureChatOpenAI, ChatOpenAI]


class TestModelTypeValidation:
    """
    These tests document a bug in langchain's pydantic implementation where Union type
    validation depends on the order of types. This is tested as a canary since the
    behavior shouldn't affect controlflow - this test suite exists for reference only.
    """

    def test_openai_first_validation(self):
        """Test validation when ChatOpenAI is first in Union"""
        openai = ChatOpenAI(model="gpt-4")
        azure = AzureChatOpenAI(api_version="1", azure_endpoint="2")

        # OpenAI model should work
        model = OpenAIFirst(model=openai)
        assert isinstance(model.model, ChatOpenAI)

        # Azure model should fail
        with pytest.raises(Exception):
            OpenAIFirst(model=azure)

    def test_azure_first_validation(self):
        """Test validation when AzureChatOpenAI is first in Union"""
        openai = ChatOpenAI(model="gpt-4")
        azure = AzureChatOpenAI(api_version="1", azure_endpoint="2")

        # Azure model should work
        model = AzureFirst(model=azure)
        assert isinstance(model.model, AzureChatOpenAI)

        # OpenAI model should fail
        with pytest.raises(Exception):
            AzureFirst(model=openai)

    def test_controlflow_model_validation(self):
        """Test that controlflow's own typing accepts both model types"""
        openai = ChatOpenAI(model="gpt-4")
        azure = AzureChatOpenAI(api_version="1", azure_endpoint="2")

        # Both model types should work with OpenAIRules
        rules_openai = OpenAIRules(model=openai)
        assert isinstance(rules_openai.model, ChatOpenAI)

        rules_azure = OpenAIRules(model=azure)
        assert isinstance(rules_azure.model, AzureChatOpenAI)
