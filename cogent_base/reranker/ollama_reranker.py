"""
Copyright (c) 2025 Mirasurf
"""

import logging
from typing import List, Union

try:
    import ollama
except ImportError:
    ollama = None  # Make ollama import optional

from cogent_base.config import get_cogent_config
from cogent_base.models.chunk import ObjectChunk
from cogent_base.ollama import initialize_ollama_model
from cogent_base.reranker.base_reranker import BaseReranker

logger = logging.getLogger(__name__)


class OllamaReranker(BaseReranker):
    """
    Ollama reranker implementation that provides direct access to Ollama.
    Uses registered models from the config file with direct Ollama client.
    """

    def __init__(self, model_key: str) -> None:
        """
        Initialize Ollama reranker with a model key from registered_rerankers.

        Args:
            model_key: The key of the model in the registered_rerankers config
        """
        if ollama is None:
            raise ImportError("Ollama library not installed. Please install it with: pip install ollama")

        settings = get_cogent_config()
        self.model_key = model_key

        # Get the model configuration from registered_rerankers
        if (
            not hasattr(settings.reranker, "registered_rerankers")
            or model_key not in settings.reranker.registered_rerankers
        ):
            raise ValueError(f"Reranker '{model_key}' not found in registered_rerankers configuration")

        self.model_config = settings.reranker.registered_rerankers[model_key]

        # Initialize Ollama configuration using utility function
        (
            self.is_ollama,
            self.ollama_api_base,
            self.ollama_base_model_name,
        ) = initialize_ollama_model(model_key, self.model_config)

        if not self.is_ollama:
            raise ValueError(f"Reranker '{model_key}' is not configured as an Ollama model")

        if not self.ollama_api_base or not self.ollama_base_model_name:
            raise ValueError(f"Invalid Ollama configuration for reranker '{model_key}'")

        logger.info(
            f"Initialized Ollama reranker with model_key={model_key}, "
            f"config={self.model_config}, api_base={self.ollama_api_base}, "
            f"model_name={self.ollama_base_model_name}"
        )

    async def rerank(
        self,
        query: str,
        chunks: List[ObjectChunk],
    ) -> List[ObjectChunk]:
        """Rerank chunks based on their relevance to the query"""
        if not chunks:
            return []

        # Get scores for all chunks
        passages = [chunk.content for chunk in chunks]
        scores = await self.compute_score(query, passages)

        # Update scores and sort chunks
        for chunk, score in zip(chunks, scores):
            chunk.score = float(score)

        return sorted(chunks, key=lambda x: x.score, reverse=True)

    async def compute_score(
        self,
        query: str,
        text: Union[str, List[str]],
    ) -> Union[float, List[float]]:
        """Compute relevance scores between query and text"""
        if isinstance(text, str):
            text = [text]
            single_text = True
        else:
            single_text = False

        scores = await self._compute_score_ollama(query, text)

        return scores[0] if single_text and len(scores) == 1 else scores

    async def _compute_score_ollama(self, query: str, texts: List[str]) -> List[float]:
        """Compute scores using direct Ollama client"""
        try:
            client = ollama.AsyncClient(host=self.ollama_api_base)

            scores = []
            for text in texts:
                # Create a prompt for reranking
                prompt = (
                    f"Query: {query}\nPassage: {text}\n\n"
                    "Please provide a relevance score between 0 and 1, "
                    "where 1 is most relevant. Only return the number."
                )

                response = await client.chat(
                    model=self.ollama_base_model_name,
                    messages=[{"role": "user", "content": prompt}],
                    options={"temperature": 0.0},  # Use deterministic output
                )

                # Extract score from response
                try:
                    score_text = response["message"]["content"].strip()
                    score = float(score_text)
                    # Ensure score is between 0 and 1
                    score = max(0.0, min(1.0, score))
                    scores.append(score)
                except (ValueError, AttributeError, KeyError):
                    logger.warning(f"Could not parse score from Ollama response: {response}")
                    scores.append(0.0)

            return scores

        except Exception as e:
            logger.error(f"Error computing scores with Ollama: {e}")
            # Fallback to uniform scores
            return [0.5] * len(texts)
