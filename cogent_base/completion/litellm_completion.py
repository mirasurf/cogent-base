"""
Copyright (c) 2025 Mirasurf
Copyright (c) 2023-2025 morphik/morphik-core
Original code from https://github.com/morphik/morphik-core
"""

import logging
from typing import Any, AsyncGenerator, Dict, List, Optional, Tuple, Union

import litellm
from pydantic import BaseModel

from cogent_base.config import get_cogent_config
from cogent_base.models.completion import CompletionRequest, CompletionResponse

from .base_completion import BaseCompletionModel

logger = logging.getLogger(__name__)


def get_system_message() -> Dict[str, str]:
    """Return the standard system message for Cogent's query agent."""
    return {
        "role": "system",
        "content": """You are Cogent's powerful query agent. Your role is to:

1. Analyze the provided context chunks from objects carefully
2. Use the context to answer questions accurately and comprehensively
3. Be clear and concise in your answers
4. When relevant, cite specific parts of the context to support your answers
5. For image-based queries, analyze the visual content in conjunction with any text context provided
6. Format your responses using Markdown.

Remember: Your primary goal is to provide accurate, context-aware responses that help users understand
and utilize the information in their objects effectively.""",
    }


def _process_context_chunks_for_litellm(context_chunks: List[str]) -> Tuple[List[str], List[str]]:
    """
    Process context chunks and separate text from images for LiteLLM.

    Args:
        context_chunks: List of context chunks which may include images

    Returns:
        Tuple of (context_text, image_urls)
    """
    context_text = []
    image_urls = []  # For LiteLLM models (full data URI)

    for chunk in context_chunks:
        if chunk.startswith("data:image/"):
            image_urls.append(chunk)
        else:
            context_text.append(chunk)

    return context_text, image_urls


def format_user_content(context_text: List[str], query: str, prompt_template: Optional[str] = None) -> str:
    """
    Format the user content based on context and query.

    Args:
        context_text: List of context text chunks
        query: The user query
        prompt_template: Optional template to format the content

    Returns:
        Formatted user content string
    """
    context = "\n" + "\n\n".join(context_text) + "\n\n" if context_text else ""

    if prompt_template:
        return prompt_template.format(
            context=context,
            question=query,
            query=query,
        )
    elif context_text:
        return f"Context: {context} Question: {query}"
    else:
        return query


def create_dynamic_model_from_schema(schema: Union[type, Dict]) -> Optional[type]:
    """
    Create a dynamic Pydantic model from a schema definition.

    Args:
        schema: Either a Pydantic BaseModel class or a JSON schema dict

    Returns:
        A Pydantic model class or None if schema format is not recognized
    """
    from pydantic import create_model

    if isinstance(schema, type) and issubclass(schema, BaseModel):
        return schema
    elif isinstance(schema, dict) and "properties" in schema:
        # Create a dynamic model from JSON schema
        field_definitions = {}
        schema_dict = schema

        # Type mapping for JSON schema to Python types
        type_mapping = {
            "string": str,
            "number": float,
            "integer": int,
            "boolean": bool,
            "array": list,
            "object": dict,
        }

        for field_name, field_info in schema_dict.get("properties", {}).items():
            if isinstance(field_info, dict) and "type" in field_info:
                field_type = field_info.get("type")
                # Convert schema types to Python types using mapping
                python_type = type_mapping.get(field_type, Any)
                field_definitions[field_name] = (python_type, None)

        # Create the dynamic model
        return create_model("DynamicQueryModel", **field_definitions)
    else:
        logger.warning(f"Unrecognized schema format: {schema}")
        return None


class LiteLLMCompletionModel(BaseCompletionModel):
    """
    LiteLLM completion model implementation that provides unified access to various LLM providers.
    Uses registered models from the config file.
    """

    def __init__(self, model_key: str) -> None:
        """
        Initialize LiteLLM completion model with a model key from registered_models.

        Args:
            model_key: The key of the model in the registered_models config
        """
        settings = get_cogent_config()
        self.model_key = model_key

        # Get the model configuration from registered_models
        if not hasattr(settings.llm, "registered_models") or model_key not in settings.llm.registered_models:
            raise ValueError(f"Model '{model_key}' not found in registered_models configuration")

        self.model_config = settings.llm.registered_models[model_key]

        logger.info(f"Initialized LiteLLM completion model with model_key={model_key}, " f"config={self.model_config}")

    async def _handle_structured_litellm(
        self,
        dynamic_model: type,
        system_message: Dict[str, str],
        user_content: str,
        image_urls: List[str],
        request: CompletionRequest,
        history_messages: List[Dict[str, str]],
        model_config: Optional[Dict[str, Any]] = None,
    ) -> CompletionResponse:
        """Handle structured output generation with LiteLLM."""
        import instructor

        try:
            # Use instructor with litellm
            client = instructor.from_litellm(litellm.acompletion, mode=instructor.Mode.JSON)

            # Create content list with text and images
            content_list = [{"type": "text", "text": user_content}]

            # Add images if available
            if image_urls:
                NUM_IMAGES = min(5, len(image_urls))
                for img_url in image_urls[:NUM_IMAGES]:
                    content_list.append({"type": "image_url", "image_url": {"url": img_url}})

            # Create messages for instructor
            messages = [system_message] + history_messages + [{"role": "user", "content": content_list}]

            # Extract model configuration
            config = model_config or self.model_config
            model = config.get("model", config.get("model_name", ""))
            model_params = {k: v for k, v in config.items() if k not in ["model", "model_name"]}

            # Add model kwargs from request if provided
            if request.model_kwargs:
                model_params.update(request.model_kwargs)

            # Override with completion request parameters
            if request.temperature is not None:
                model_params["temperature"] = request.temperature
            if request.max_tokens is not None:
                model_params["max_tokens"] = request.max_tokens

            # Add format forcing for structured output
            model_params["response_format"] = {"type": "json_object"}

            # Call instructor with litellm
            response = await client.chat.completions.create(
                model=model,
                messages=messages,
                response_model=dynamic_model,
                **model_params,
            )

            # Get token usage from response
            completion_tokens = model_params.get("response_tokens", 0)
            prompt_tokens = model_params.get("prompt_tokens", 0)

            return CompletionResponse(
                completion=response,
                usage={
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": prompt_tokens + completion_tokens,
                },
                finish_reason="stop",
            )

        except Exception as e:
            logger.error(f"Error using instructor with LiteLLM: {e}")
            # Fall back to standard completion if instructor fails
            logger.warning("Falling back to standard LiteLLM completion without structured output")
            return None

    async def _handle_standard_litellm(
        self,
        user_content: str,
        image_urls: List[str],
        request: CompletionRequest,
        history_messages: List[Dict[str, str]],
        model_config: Optional[Dict[str, Any]] = None,
    ) -> CompletionResponse:
        """Handle standard (non-structured) output generation with LiteLLM."""
        # Use provided model_config or fall back to instance config
        config = model_config or self.model_config
        model_name = config.get("model", config.get("model_name", ""))

        logger.debug(f"Using LiteLLM for model: {model_name}")
        # Build messages for LiteLLM
        content_list = [{"type": "text", "text": user_content}]
        include_images = image_urls  # Use the collected full data URIs

        if include_images:
            NUM_IMAGES = min(5, len(image_urls))
            for img_url in image_urls[:NUM_IMAGES]:
                content_list.append({"type": "image_url", "image_url": {"url": img_url}})

        # LiteLLM uses list content format
        user_message = {"role": "user", "content": content_list}
        # Use the system prompt defined earlier
        litellm_messages = [get_system_message()] + history_messages + [user_message]

        # Prepare LiteLLM parameters
        model_params = {
            "model": model_name,
            "messages": litellm_messages,
            "max_tokens": request.max_tokens,
            "temperature": request.temperature,
            "num_retries": 3,
        }

        # Add additional parameters from config
        for key, value in config.items():
            if key not in ["model", "model_name"]:
                model_params[key] = value

        # Add model kwargs from request if provided
        if request.model_kwargs:
            model_params.update(request.model_kwargs)

        logger.debug(f"Calling LiteLLM with params: {model_params}")
        response = await litellm.acompletion(**model_params)

        return CompletionResponse(
            completion=response.choices[0].message.content,
            usage={
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens,
            },
            finish_reason=response.choices[0].finish_reason,
        )

    async def _handle_streaming_litellm(
        self,
        user_content: str,
        image_urls: List[str],
        request: CompletionRequest,
        history_messages: List[Dict[str, str]],
        model_config: Optional[Dict[str, Any]] = None,
    ) -> AsyncGenerator[str, None]:
        """Handle streaming output generation with LiteLLM."""
        # Use provided model_config or fall back to instance config
        config = model_config or self.model_config
        model_name = config.get("model", config.get("model_name", ""))

        logger.debug(f"Using LiteLLM streaming for model: {model_name}")
        # Build messages for LiteLLM
        content_list = [{"type": "text", "text": user_content}]
        include_images = image_urls  # Use the collected full data URIs

        if include_images:
            NUM_IMAGES = min(5, len(image_urls))
            for img_url in image_urls[:NUM_IMAGES]:
                content_list.append({"type": "image_url", "image_url": {"url": img_url}})

        # LiteLLM uses list content format
        user_message = {"role": "user", "content": content_list}
        # Use the system prompt defined earlier
        litellm_messages = [get_system_message()] + history_messages + [user_message]

        # Prepare LiteLLM parameters
        model_params = {
            "model": model_name,
            "messages": litellm_messages,
            "max_tokens": request.max_tokens,
            "temperature": request.temperature,
            "stream": True,  # Enable streaming
            "num_retries": 3,
        }

        # Add additional parameters from config
        for key, value in config.items():
            if key not in ["model", "model_name"]:
                model_params[key] = value

        # Add model kwargs from request if provided
        if request.model_kwargs:
            model_params.update(request.model_kwargs)

        logger.debug(f"Calling LiteLLM streaming with params: {model_params}")
        response = await litellm.acompletion(**model_params)

        # Stream the response chunks
        async for chunk in response:
            if chunk.choices and chunk.choices[0].delta and chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content

    async def complete(self, request: CompletionRequest) -> Union[CompletionResponse, AsyncGenerator[str, None]]:
        """
        Generate completion using LiteLLM.

        Args:
            request: CompletionRequest object containing query, context, and parameters

        Returns:
            CompletionResponse object with the generated text and usage statistics or
            AsyncGenerator for streaming responses
        """
        # Use llm_config from request if provided, otherwise use instance config
        model_config = request.llm_config if request.llm_config else self.model_config

        # Process context chunks and handle images
        context_text, image_urls = _process_context_chunks_for_litellm(request.context_chunks)

        # Format user content
        user_content = format_user_content(context_text, request.query, request.prompt_template)

        history_messages = [{"role": m.role, "content": m.content} for m in (request.chat_history or [])]

        # Check if structured output is requested
        structured_output = request.schema is not None

        # Streaming is not supported with structured output
        if request.stream_response and structured_output:
            logger.warning("Streaming is not supported with structured output. Falling back to non-streaming.")
            request.stream_response = False

        # If streaming is requested and no structured output
        if request.stream_response and not structured_output:
            return self._handle_streaming_litellm(
                user_content,
                image_urls,
                request,
                history_messages,
                model_config,
            )

        # If structured output is requested, use instructor to handle it
        if structured_output:
            # Get dynamic model from schema
            dynamic_model = create_dynamic_model_from_schema(request.schema)

            # If schema format is not recognized, log warning and fall back to text completion
            if not dynamic_model:
                logger.warning(f"Unrecognized schema format: {request.schema}. Falling back to text completion.")
                structured_output = False
            else:
                logger.info(f"Using structured output with model: {dynamic_model.__name__}")

                # Create system and user messages with enhanced instructions for structured output
                system_message = {
                    "role": "system",
                    "content": get_system_message()["content"]
                    + "\n\nYou MUST format your response according to the required schema.",
                }

                # Create enhanced user message that includes schema information
                enhanced_user_content = (
                    user_content + "\n\nPlease format your response according to the required schema."
                )

                # Try structured output with LiteLLM
                response = await self._handle_structured_litellm(
                    dynamic_model,
                    system_message,
                    enhanced_user_content,
                    image_urls,
                    request,
                    history_messages,
                    model_config,
                )
                if response:
                    return response
                structured_output = False  # Fall back if structured output failed

        # If we're here, either structured output wasn't requested or instructor failed
        # Proceed with standard completion
        return await self._handle_standard_litellm(
            user_content,
            image_urls,
            request,
            history_messages,
            model_config,
        )
