"""
LLM Client for SalesLLM evaluation.

Provides unified interface for interacting with different LLM providers
(OpenAI, Azure OpenAI, vLLM) in a chat-based conversation format.
"""

from typing import List, Dict, Any, Optional, Union, Callable
import time
import logging
import asyncio
from openai import OpenAI, AzureOpenAI, AsyncOpenAI, AsyncAzureOpenAI

# Setup module logger
logger = logging.getLogger(__name__)


class SalesLLMClient:
    """
    Unified client for SalesLLM evaluation with support for:
    - OpenAI API compatible endpoints
    - Azure OpenAI
    - vLLM server endpoints
    - Chat-based conversation format
    """
    
    def __init__(
        self,
        api_base: str,
        api_key: str,
        model_name: str,
        api_type: str = "openai",  # "openai", "azure", "vllm"
        api_version: Optional[str] = None,  # For Azure OpenAI
        max_retries: int = 3,
        retry_delay: float = 1.0,
    ):
        """
        Initialize the SalesLLM client.
        
        Args:
            api_base: API endpoint base URL
            api_key: API key for authentication
            model_name: Model name to use
            api_type: Type of API ("openai", "azure", "vllm")
            api_version: Azure OpenAI API version (required for azure type)
            max_retries: Maximum number of retries on failure
            retry_delay: Delay between retries in seconds
        """
        self.api_base = api_base
        self.api_key = api_key
        self.model_name = model_name
        self.api_type = api_type
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        
        # Initialize the appropriate client
        if api_type == "azure":
            if not api_version:
                raise ValueError("api_version is required for Azure OpenAI")
            
            # Parse azure_endpoint and azure_deployment from api_base
            import re
            match = re.match(r"(https?://[^/]+)(?:/openai/deployments/([^/]+))?", api_base)
            if not match:
                raise ValueError(f"Invalid Azure API base URL format: {api_base}")
            
            azure_endpoint = match.group(1)
            azure_deployment = match.group(2)
            
            if not azure_deployment:
                raise ValueError(f"Azure deployment name not found in API base URL: {api_base}")

            self.client = AzureOpenAI(
                api_version=api_version,
                api_key=api_key,
                azure_endpoint=azure_endpoint,
                azure_deployment=azure_deployment
            )
            self.async_client = AsyncAzureOpenAI(
                api_version=api_version,
                api_key=api_key,
                azure_endpoint=azure_endpoint,
                azure_deployment=azure_deployment
            )
        else:  # openai or vllm
            self.client = OpenAI(
                base_url=api_base,
                api_key=api_key
            )
            self.async_client = AsyncOpenAI(
                base_url=api_base,
                api_key=api_key
            )
    
    def chat_completion(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.8,
        top_p: float = 0.99,
        max_tokens: Optional[int] = 2048,
        reasoning_effort: str = "minimal",
        **kwargs
    ) -> Optional[str]:
        """
        Generate a chat completion response.
        
        Args:
            messages: List of conversation messages with role and content
            temperature: Sampling temperature
            top_p: Top-p sampling parameter
            max_tokens: Maximum tokens to generate
            reasoning_effort: Reasoning effort parameter
                           Options: "none" (disable reasoning), "minimal", "medium", "high"
            **kwargs: Additional parameters to pass to the API
            
        Returns:
            Generated text response, or None if all retries failed
        """
        for attempt in range(self.max_retries):
            try:
                # Prepare the API call parameters
                params = {
                    "model": self.model_name,
                    "messages": messages,
                    "temperature": temperature,
                    "top_p": top_p,
                    "max_tokens": max_tokens,
                    **kwargs
                }
                
                # Add reasoning parameters based on API type
                if reasoning_effort:
                    if self.api_type == "openai":
                        # OpenAI API supports reasoning_effort parameter
                        if reasoning_effort != "minimal":
                            params["reasoning_effort"] = reasoning_effort
                    elif self.api_type == "azure":
                        # Azure OpenAI may support reasoning_effort depending on model
                        if reasoning_effort != "minimal":
                            params["reasoning_effort"] = reasoning_effort
                    # other include vllm or other openai api compatible endpoints (qwen,deepseek)
                    elif self.api_type == "other":
                        # vLLM uses extra_body for reasoning/thinking parameters
                        if reasoning_effort:
                            params["extra_body"] = {"enable_thinking": True}
                        else:
                            params["extra_body"] = {"enable_thinking": False}

                
                response = self.client.chat.completions.create(**params)
                return response.choices[0].message.content.strip()
                
            except Exception as e:
                logger.warning(f"Attempt {attempt + 1} failed: {str(e)}")
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay * (attempt + 1))
                else:
                    logger.error(f"Failed to generate response after {self.max_retries} attempts: {str(e)}")
                    return None

    async def chat_completion_async(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.8,
        top_p: float = 0.99,
        max_tokens: Optional[int] = 2048,
        reasoning_effort: str = "minimal",
        **kwargs
    ) -> Optional[str]:
        """
        Generate a chat completion response asynchronously.
        
        Args:
            messages: List of conversation messages with role and content
            temperature: Sampling temperature
            top_p: Top-p sampling parameter
            max_tokens: Maximum tokens to generate
            reasoning_effort: Reasoning effort parameter
            **kwargs: Additional parameters to pass to the API
            
        Returns:
            Generated text response, or None if all retries failed
        """
        for attempt in range(self.max_retries):
            try:
                # Prepare the API call parameters
                params = {
                    "model": self.model_name,
                    "messages": messages,
                    "temperature": temperature,
                    "top_p": top_p,
                    "max_tokens": max_tokens,
                    **kwargs
                }
                
                # Add reasoning parameters based on API type
                if reasoning_effort:
                    if self.api_type == "openai":
                        # OpenAI API supports reasoning_effort parameter
                        if reasoning_effort != "minimal":
                            params["reasoning_effort"] = reasoning_effort
                    elif self.api_type == "azure":
                        # Azure OpenAI may support reasoning_effort depending on model
                        if reasoning_effort != "minimal":
                            params["reasoning_effort"] = reasoning_effort
                    # other include vllm or other openai api compatible endpoints (qwen,deepseek)
                    elif self.api_type == "other":
                        # vLLM uses extra_body for reasoning/thinking parameters
                        if reasoning_effort:
                            params["extra_body"] = {"enable_thinking": True}
                        else:
                            params["extra_body"] = {"enable_thinking": False}

                
                response = await self.async_client.chat.completions.create(**params)
                return response.choices[0].message.content.strip()
                
            except Exception as e:
                logger.warning(f"Attempt {attempt + 1} failed: {str(e)}")
                if attempt < self.max_retries - 1:
                    await asyncio.sleep(self.retry_delay * (attempt + 3))
                else:
                    logger.error(f"Failed to generate response after {self.max_retries} attempts: {str(e)}")
                    return None

    def batch_chat_completion(
        self,
        conversations: List[List[Dict[str, str]]],
        temperature: float = 0.8,
        top_p: float = 0.99,
        max_tokens: Optional[int] = 2048,
        reasoning_effort: str = "minimal",
        **kwargs
    ) -> List[Optional[str]]:
        """
        Generate chat completions for multiple conversations.
        
        Args:
            conversations: List of conversation message lists
            temperature: Sampling temperature
            top_p: Top-p sampling parameter
            max_tokens: Maximum tokens to generate
            reasoning_effort: Reasoning effort parameter
            **kwargs: Additional parameters to pass to the API
            
        Returns:
            List of generated text responses
        """
        responses = []
        for messages in conversations:
            response = self.chat_completion(
                messages=messages,
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens,
                reasoning_effort=reasoning_effort,
                **kwargs
            )
            responses.append(response)
        return responses


class SalesLLMClientManager:
    """
    Manager for multiple SalesLLM clients (assistant, user, judge models).
    """
    
    def __init__(self):
        self.clients = {}
    
    def add_client(
        self,
        client_name: str,
        api_base: str,
        api_key: str,
        model_name: str,
        api_type: str = "openai",
        api_version: Optional[str] = None,
        **kwargs
    ):
        """
        Add a new client to the manager.
        
        Args:
            client_name: Name to identify this client (e.g., "assistant", "user", "judge")
            api_base: API endpoint base URL
            api_key: API key for authentication
            model_name: Model name to use
            api_type: Type of API ("openai", "azure", "vllm")
            api_version: Azure OpenAI API version
            **kwargs: Additional parameters for the client
        """
        self.clients[client_name] = SalesLLMClient(
            api_base=api_base,
            api_key=api_key,
            model_name=model_name,
            api_type=api_type,
            api_version=api_version,
            **kwargs
        )
    
    def get_client(self, client_name: str) -> Optional[SalesLLMClient]:
        """Get a client by name."""
        return self.clients.get(client_name)
    
    def chat_with_role(
        self,
        role: str,
        messages: List[Dict[str, str]],
        temperature: float = 0.8,
        top_p: float = 0.99,
        max_tokens: Optional[int] = 2048,
        reasoning_effort: str = "minimal",
        **kwargs
    ) -> Optional[str]:
        """
        Chat with a specific role/client.
        
        Args:
            role: Client name ("assistant", "user", "judge", etc.)
            messages: List of conversation messages
            temperature: Sampling temperature
            top_p: Top-p sampling parameter
            max_tokens: Maximum tokens to generate
            reasoning_effort: Reasoning effort parameter
            **kwargs: Additional parameters
            
        Returns:
            Generated text response, or None if client not found or request failed
        """
        client = self.get_client(role)
        if client is None:
            logger.error(f"Client '{role}' not found")
            return None
        
        return client.chat_completion(
            messages=messages,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            reasoning_effort=reasoning_effort,
            **kwargs
        )

    async def chat_with_role_async(
        self,
        role: str,
        messages: List[Dict[str, str]],
        temperature: float = 0.8,
        top_p: float = 0.99,
        max_tokens: Optional[int] = 2048,
        reasoning_effort: str = "minimal",
        **kwargs
    ) -> Optional[str]:
        """
        Chat with a specific role/client asynchronously.
        
        Args:
            role: Client name ("assistant", "user", "judge", etc.)
            messages: List of conversation messages
            temperature: Sampling temperature
            top_p: Top-p sampling parameter
            max_tokens: Maximum tokens to generate
            reasoning_effort: Reasoning effort parameter
            **kwargs: Additional parameters
            
        Returns:
            Generated text response, or None if client not found or request failed
        """
        client = self.get_client(role)
        if client is None:
            logger.error(f"Client '{role}' not found")
            return None
        
        return await client.chat_completion_async(
            messages=messages,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            reasoning_effort=reasoning_effort,
            **kwargs
        )


# Backward compatibility - keep the original LLMClient class name
class LLMClient(SalesLLMClient):
    """Backward compatibility wrapper for the original LLMClient class name."""
    pass


# Utility functions for conversation management
def create_conversation_message(role: str, content: str) -> Dict[str, str]:
    """Create a conversation message."""
    return {"role": role, "content": content}

def create_system_message(content: str) -> Dict[str, str]:
    """Create a system message."""
    return create_conversation_message("system", content)

def create_user_message(content: str) -> Dict[str, str]:
    """Create a user message."""
    return create_conversation_message("user", content)

def create_assistant_message(content: str) -> Dict[str, str]:
    """Create an assistant message."""
    return create_conversation_message("assistant", content)


def build_conversation_history(
    system_message: Optional[str] = None,
    user_messages: Optional[List[str]] = None,
    assistant_messages: Optional[List[str]] = None
) -> List[Dict[str, str]]:
    """
    Build a conversation history from individual messages.
    
    Args:
        system_message: Optional system message
        user_messages: List of user messages
        assistant_messages: List of assistant messages
        
    Returns:
        List of conversation messages
    """
    messages = []
    
    if system_message:
        messages.append(create_system_message(system_message))
    
    if user_messages and assistant_messages:
        # Interleave user and assistant messages
        for user_msg, assistant_msg in zip(user_messages, assistant_messages):
            messages.append(create_user_message(user_msg))
            messages.append(create_assistant_message(assistant_msg))
        
        # Add remaining messages if lengths differ
        if len(user_messages) > len(assistant_messages):
            for msg in user_messages[len(assistant_messages):]:
                messages.append(create_user_message(msg))
        elif len(assistant_messages) > len(user_messages):
            for msg in assistant_messages[len(user_messages):]:
                messages.append(create_assistant_message(msg))
    elif user_messages:
        for msg in user_messages:
            messages.append(create_user_message(msg))
    elif assistant_messages:
        for msg in assistant_messages:
            messages.append(create_assistant_message(msg))
    
    return messages