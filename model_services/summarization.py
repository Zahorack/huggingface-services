import pydantic
import torch
from transformers import pipeline

import pubsub
from model_registry import ModelRegistry
from model_services import base
from model_services.schemas import (
    EntitySummarizationInputSchema,
    EntitySummarizationResultSchema,
    SummarizedWikipediaEntity,
)


@ModelRegistry.register
class EntitySummarizationModelService(base.ModelService):
    """A model service which provides text summarization using BART."""

    _NAME: str = "entity-summarization"
    _MODEL_NAME: str = "facebook/bart-large-cnn"
    _MAX_LENGTH: int = 130
    _MIN_LENGTH: int = 30

    def __init__(self, pubsub_client: pubsub.PubSubClient):
        super().__init__(model_name=self._NAME, pubsub_client=pubsub_client)
        self._summarizer = None

    def _load_model(self):
        """Lazy load the summarization pipeline to keep startup times fast."""
        if self._summarizer is None:
            if torch.backends.mps.is_available():
                device = "mps"
            elif torch.cuda.is_available():
                device = "cuda"
            else:
                device = "cpu"

            self._summarizer = pipeline("summarization", model=self._MODEL_NAME, device=device)

    def get_input_schema(self) -> type[pydantic.BaseModel]:
        """Get the input schema for the model service."""
        return EntitySummarizationInputSchema

    def get_output_schema(self) -> type[pydantic.BaseModel]:
        """Get the output schema for the model service."""
        return EntitySummarizationResultSchema

    def _summarize_text(self, text: str) -> str:
        """Summarize text using the pipeline."""
        summary = self._summarizer(
            text,
            max_length=self._MAX_LENGTH,
            min_length=self._MIN_LENGTH,
            do_sample=False,
        )
        return summary[0]["summary_text"]

    async def _predict(
        self, model_input: EntitySummarizationInputSchema
    ) -> EntitySummarizationResultSchema:
        """Summarize text content and extract entities."""
        self._load_model()

        summarized_entities = []
        for entity in model_input.linked_entities:
            if entity.wikipedia_entries_found:
                entity_dict = entity.model_dump()
                entity_dict["summary"] = self._summarize_text(entity.content)
                summarized_entities.append(SummarizedWikipediaEntity.model_validate(entity_dict))
            else:
                summarized_entities.append(entity)

        return EntitySummarizationResultSchema(
            id=model_input.id,
            text=model_input.text,
            summarized_entities=summarized_entities,
        )
