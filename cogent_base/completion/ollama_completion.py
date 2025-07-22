"""
Copyright (c) 2025 Mirasurf
Copyright (c) 2023-2025 morphik/morphik-core
Original code from https://github.com/morphik/morphik-core
"""

import logging
from typing import Any, AsyncGenerator, Dict, List, Optional, Tuple, Union

try:
    import ollama
except ImportError:
    ollama = None  # Make ollama import optional

from pydantic import BaseModel

from cogent_base.config import get_cogent_config
from cogent_base.models.completion import CompletionRequest, CompletionResponse
from cogent_base.ollama import initialize_ollama_model

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


def _process_context_chunks_for_ollama(context_chunks: List[str]) -> Tuple[List[str], List[str]]:
    """
    Process context chunks and separate text from images for Ollama.

    Args:
        context_chunks: List of context chunks which may include images

    Returns:
        Tuple of (context_text, ollama_image_data)
    """
    context_text = []
    ollama_image_data = []  # For Ollama models (raw base64)

    for chunk in context_chunks:
        if chunk.startswith("data:image/"):
            # For Ollama, strip the data URI prefix and just keep the base64 data
            try:
                base64_data = chunk.split(",", 1)[1]
                ollama_image_data.append(base64_data)
            except IndexError:
                logger.warning(f"Could not parse base64 data from image chunk: {chunk[:50]}...")
        else:
            context_text.append(chunk)

    return context_text, ollama_image_data


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


class OllamaCompletionModel(BaseCompletionModel):
    """
    Ollama completion model implementation that provides direct access to Ollama.
    Uses registered models from the config file with direct Ollama client.
    """

    def __init__(self, model_key: str) -> None:
        """
        Initialize Ollama completion model with a model key from registered_models.

        Args:
            model_key: The key of the model in the registered_models config
        """
        if ollama is None:
            raise ImportError("Ollama library not installed. Please install it with: pip install ollama")

        settings = get_cogent_config()
        self.model_key = model_key

        # Get the model configuration from registered_models
        if not hasattr(settings.llm, "registered_models") or model_key not in settings.llm.registered_models:
            raise ValueError(f"Model '{model_key}' not found in registered_models configuration")

        self.model_config = settings.llm.registered_models[model_key]

        # Initialize Ollama configuration using utility function
        self.is_ollama, self.ollama_api_base, self.ollama_base_model_name = initialize_ollama_model(
            model_key, self.model_config
        )

        if not self.is_ollama:
            raise ValueError(f"Model '{model_key}' is not configured as an Ollama model")

        if not self.ollama_api_base or not self.ollama_base_model_name:
            raise ValueError(f"Invalid Ollama configuration for model '{model_key}'")

        logger.info(
            f"Initialized Ollama completion model with model_key={model_key}, "
            f"config={self.model_config}, api_base={self.ollama_api_base}, "
            f"model_name={self.ollama_base_model_name}"
        )

    async def _handle_structured_ollama(
        self,
        dynamic_model: type,
        system_message: Dict[str, str],
        user_content: str,
        ollama_image_data: List[str],
        request: CompletionRequest,
        history_messages: List[Dict[str, str]],
    ) -> CompletionResponse:
        """Handle structured output generation with Ollama."""
        try:
            client = ollama.AsyncClient(host=self.ollama_api_base)

            # Add images directly to content if available
            content_data = user_content
            if ollama_image_data and len(ollama_image_data) > 0:
                # Ollama image handling is limited; we can use only the first image
                content_data = {
                    "content": user_content,
                    "images": [ollama_image_data[0]],
                }

            # Create messages for Ollama
            messages = [system_message] + history_messages + [{"role": "user", "content": content_data}]

            # Get the JSON schema from the dynamic model
            format_schema = dynamic_model.model_json_schema()

            # Call Ollama directly with format parameter
            response = await client.chat(
                model=self.ollama_base_model_name,
                messages=messages,
                format=format_schema,
                options={
                    "temperature": request.temperature or 0.1,  # Lower temperature for structured output
                    "num_predict": request.max_tokens,
                },
            )

            # Parse the response into the dynamic model
            parsed_response = dynamic_model.model_validate_json(response["message"]["content"])

            # Extract token usage information
            usage = {
                "prompt_tokens": response.get("prompt_eval_count", 0),
                "completion_tokens": response.get("eval_count", 0),
                "total_tokens": response.get("prompt_eval_count", 0) + response.get("eval_count", 0),
            }

            return CompletionResponse(
                completion=parsed_response,
                usage=usage,
                finish_reason=response.get("done_reason", "stop"),
            )

        except Exception as e:
            logger.error(f"Error using Ollama for structured output: {e}")
            # Fall back to standard completion if structured output fails
            logger.warning("Falling back to standard Ollama completion without structured output")
            return None

    async def _handle_standard_ollama(
        self,
        user_content: str,
        ollama_image_data: List[str],
        request: CompletionRequest,
        history_messages: List[Dict[str, str]],
    ) -> CompletionResponse:
        """Handle standard (non-structured) output generation with Ollama."""
        logger.debug(f"Using direct Ollama client for model: {self.ollama_base_model_name}")
        client = ollama.AsyncClient(host=self.ollama_api_base)

        # Construct Ollama messages
        system_message = {
            "role": "system",
            "content": get_system_message()["content"],
        }
        user_message_data = {"role": "user", "content": user_content}

        # Add images directly to the user message if available
        if ollama_image_data:
            # Add all images to the user message
            user_message_data["images"] = ollama_image_data

        ollama_messages = [system_message] + history_messages + [user_message_data]

        # Construct Ollama options
        options = {
            "temperature": request.temperature,
            "num_predict": (
                request.max_tokens if request.max_tokens is not None else -1
            ),  # Default to model's default if None
        }

        try:
            response = await client.chat(
                model=self.ollama_base_model_name,
                messages=ollama_messages,
                options=options,
            )

            # Map Ollama response to CompletionResponse
            prompt_tokens = response.get("prompt_eval_count", 0)
            completion_tokens = response.get("eval_count", 0)

            return CompletionResponse(
                completion=response["message"]["content"],
                usage={
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": prompt_tokens + completion_tokens,
                },
                finish_reason=response.get("done_reason", "unknown"),  # Map done_reason if available
            )

        except Exception as e:
            logger.error(f"Error during direct Ollama call: {e}")
            raise

    async def _handle_streaming_ollama(
        self,
        user_content: str,
        ollama_image_data: List[str],
        request: CompletionRequest,
        history_messages: List[Dict[str, str]],
    ) -> AsyncGenerator[str, None]:
        """Handle streaming output generation with Ollama."""
        logger.debug(f"Using direct Ollama streaming for model: {self.ollama_base_model_name}")
        client = ollama.AsyncClient(host=self.ollama_api_base)

        # Construct Ollama messages
        system_message = {
            "role": "system",
            "content": get_system_message()["content"],
        }
        user_message_data = {"role": "user", "content": user_content}

        # Add images directly to the user message if available
        if ollama_image_data:
            # Add all images to the user message
            user_message_data["images"] = ollama_image_data

        ollama_messages = [system_message] + history_messages + [user_message_data]

        # Construct Ollama options
        options = {
            "temperature": request.temperature,
            "num_predict": (
                request.max_tokens if request.max_tokens is not None else -1
            ),  # Default to model's default if None
        }

        try:
            response = await client.chat(
                model=self.ollama_base_model_name,
                messages=ollama_messages,
                options=options,
                stream=True,  # Enable streaming
            )

            async for chunk in response:
                if chunk.get("message", {}).get("content"):
                    yield chunk["message"]["content"]

        except Exception as e:
            logger.error(f"Error during direct Ollama streaming call: {e}")
            raise

    async def complete(self, request: CompletionRequest) -> Union[CompletionResponse, AsyncGenerator[str, None]]:
        """
        Generate completion using direct Ollama client.

        Args:
            request: CompletionRequest object containing query, context, and parameters

        Returns:
            CompletionResponse object with the generated text and usage statistics or
            AsyncGenerator for streaming responses
        """
        # Process context chunks and handle images
        context_text, ollama_image_data = _process_context_chunks_for_ollama(request.context_chunks)

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
            return self._handle_streaming_ollama(user_content, ollama_image_data, request, history_messages)

        # If structured output is requested, use Ollama's format parameter
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

                # Try structured output with Ollama
                response = await self._handle_structured_ollama(
                    dynamic_model,
                    system_message,
                    enhanced_user_content,
                    ollama_image_data,
                    request,
                    history_messages,
                )
                if response:
                    return response
                structured_output = False  # Fall back if structured output failed

        # If we're here, either structured output wasn't requested or it failed
        # Proceed with standard completion
        return await self._handle_standard_ollama(user_content, ollama_image_data, request, history_messages)
