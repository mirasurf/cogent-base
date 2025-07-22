"""
Copyright (c) 2025 Mirasurf
"""

import logging
from typing import List, Union

import litellm

from cogent_base.config import get_cogent_config
from cogent_base.models.chunk import ObjectChunk
from cogent_base.reranker.base_reranker import BaseReranker

logger = logging.getLogger(__name__)


class LiteLLMReranker(BaseReranker):
    """
    LiteLLM reranker implementation that provides unified access to various
    reranker providers. Uses registered models from the config file.
    """

    def __init__(self, model_key: str) -> None:
        """
        Initialize LiteLLM reranker with a model key from registered_rerankers.

        Args:
            model_key: The key of the model in the registered_rerankers config
        """
        settings = get_cogent_config()
        self.model_key = model_key

        # Get the model configuration from registered_rerankers
        if (
            not hasattr(settings.reranker, "registered_rerankers")
            or model_key not in settings.reranker.registered_rerankers
        ):
            raise ValueError(f"Reranker '{model_key}' not found in registered_rerankers configuration")

        self.model_config = settings.reranker.registered_rerankers[model_key]

        logger.info(f"Initialized LiteLLM reranker with model_key={model_key}, " f"config={self.model_config}")

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

        scores = await self._compute_score_litellm(query, text)

        return scores[0] if single_text and len(scores) == 1 else scores

    async def _compute_score_litellm(self, query: str, texts: List[str]) -> List[float]:
        """Compute scores using LiteLLM"""
        try:
            # For reranking, we typically use a cross-encoder approach
            # This is a simplified implementation - in practice, you might want to use
            # a dedicated reranking model through LiteLLM

            scores = []
            for text in texts:
                # Create a prompt for scoring
                prompt = (
                    f"Query: {query}\nPassage: {text}\n\n"
                    "Rate the relevance of this passage to the query on a scale of 0 to 1, "
                    "where 1 is most relevant. Only respond with the number."
                )

                response = await litellm.acompletion(
                    model=self.model_config["model_name"],
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.0,  # Use deterministic output
                    max_tokens=10,  # We only need a short response
                    **{k: v for k, v in self.model_config.items() if k != "model_name"},
                )

                # Extract score from response
                try:
                    score_text = response.choices[0].message.content.strip()
                    score = float(score_text)
                    # Ensure score is between 0 and 1
                    score = max(0.0, min(1.0, score))
                    scores.append(score)
                except (ValueError, AttributeError, IndexError):
                    logger.warning(f"Could not parse score from LiteLLM response: {response}")
                    scores.append(0.0)

            return scores

        except Exception as e:
            logger.error(f"Error computing scores with LiteLLM: {e}")
            # Fallback to uniform scores
            return [0.5] * len(texts)
