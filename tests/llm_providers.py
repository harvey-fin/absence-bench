from abc import ABC, abstractmethod
import os
import openai
import anthropic
from google import genai
from google.genai import types
from together import Together
from typing import Dict, Any, Optional, List

class LLMProvider(ABC):
    """Abstract base class for LLM providers"""
    
    @abstractmethod
    def get_response(self, system_prompt: str, user_message: str, model: str) -> str:
        """Get a response from the LLM model"""
        pass
    
    @classmethod
    def get_provider(cls, provider_name: str) -> 'LLMProvider':
        """Factory method to get the appropriate provider"""
        providers = {
            'openai': OpenAIProvider,
            'togetherai': TogetherAIProvider,
            'xai': XAIProvider,
            'anthropic': AnthropicProvider,
            'cohere': CohereProvider,
            'google': GoogleProvider
        }
        
        if provider_name not in providers:
            raise ValueError(f"Provider '{provider_name}' not supported. Available providers: {', '.join(providers.keys())}")
        
        return providers[provider_name]()


class OpenAIProvider(LLMProvider):
    """OpenAI-specific implementation"""
    
    def __init__(self):
        # API key is loaded from environment variable OPENAI_API_KEY by the openai library
        pass
    
    def get_response(self, system_prompt: str, user_message: str, model: str, thinking:bool=False) -> str:
        """Get a response from OpenAI model"""
        thinking_tokens = 0
        response = openai.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message}
            ],
        )

        if thinking:
            thinking_tokens = response.usage.completion_tokens_details.reasoning_tokens
        return [response.choices[0].message.content, thinking_tokens]
    

class XAIProvider(LLMProvider):
    """XAI implementation"""

    def __init__(self):
        # initialize the client. api key is accessed through the environment variable XAI_API_KEY
        self.client = openai.OpenAI(
            api_key=os.getenv("XAI_API_KEY"),
            base_url="https://api.x.ai/v1"
        )

    def get_response(self, system_prompt: str, user_message: str, model: str, thinking:bool=False) -> str:
        """Get a response through the XAI api"""
        thinking_tokens = 0
        response = self.client.chat.completions.create(
            model=model,
            reasoning_effort="low",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message}
            ],
        )
        if thinking:
            thinking_tokens = response.usage.completion_tokens_details.reasoning_tokens

        return [response.choices[0].message.content, thinking_tokens]


class TogetherAIProvider(LLMProvider):
    """TogetherAI-specific implementation"""

    def __init__(self):
        # API key is loaded from environment variable TOGETHER_API_KEY by the togetherai library
        self.client = Together()

    def get_response(self, system_prompt: str, user_message:str, model: str) -> str:
        """Get a response through TogetherAI api"""
        # Query image with Llama 4 Maverick model
        response = self.client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ],
        )

        return response.choices[0].message.content


class AnthropicProvider(LLMProvider):
    """Anthropic-specific implementation"""
    
    def __init__(self):
        self.client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
    
    def get_response(self, system_prompt: str, user_message: str, model: str, thinking: bool=False) -> str:
        """Get a response from Anthropic model"""
        if thinking: 
            response = self.client.messages.create(
                model=model,
                max_tokens=15000,
                thinking={
                    "type": "enabled",
                    "budget_tokens": 10000,
                },
                system=system_prompt,
                messages=[
                    {"role": "user", "content": user_message}
                ],
                temperature=1
            )
            thinking = response.content[0].thinking
            response_str = response.content[1].text
            thinking_count = self.client.messages.count_tokens(
                model=model,
                messages=[{
                    "role": "user",
                    "content": thinking
                }],
            )
            num_thinking = thinking_count.input_tokens
        else:
            response = self.client.messages.create(
                model=model,
                max_tokens=5000,
                system=system_prompt,
                messages=[
                    {"role": "user", "content": user_message}
                ],
                temperature=0.0
            )
            response_str = response.content[0].text
            num_thinking = 0

        return [response_str, num_thinking]
    
class GoogleProvider(LLMProvider):
    """Google-specific implementation"""

    def __init__(self):
        self.client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))
    
    def get_response(self, system_prompt: str, user_message: str, model: str, thinking: bool=False) -> str:
        """Get a response from Gemini"""
        if thinking:
            response = self.client.models.generate_content(
                model=model,
                config=types.GenerateContentConfig(
                    system_instruction=system_prompt),
                contents=user_message
            )
            return [response.text, response.usage_metadata.thoughts_token_count]
        else:
            response = self.client.models.generate_content(
                model=model,
                config=types.GenerateContentConfig(
                    system_instruction=system_prompt,
                    thinking_config=types.ThinkingConfig(thinking_budget=0)),
                contents=user_message
            )
            return [response.text, 0]


class CohereProvider(LLMProvider):
    """Cohere-specific implementation"""
    
    def __init__(self):
        try:
            import cohere
            self.client = cohere.Client(os.environ.get("COHERE_API_KEY"))
        except ImportError:
            raise ImportError("The 'cohere' package is not installed. Please install it with 'pip install cohere'")
    
    def get_response(self, system_prompt: str, user_message: str, model: str) -> str:
        """Get a response from Cohere model"""
        # Use chat endpoint for command models, generate for other models
        if model.startswith("command"):
            response = self.client.chat(
                model=model,
                message=user_message,
                preamble=system_prompt,
                temperature=0.0,
            )
            return response.text
        else:
            # Legacy generate API for older models
            response = self.client.generate(
                model=model,
                prompt=f"{system_prompt}\n\n{user_message}",
                max_tokens=1000,
                temperature=0.0,
            )
            return response.generations[0].text