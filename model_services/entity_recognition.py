import pydantic
import transformers

import pubsub
from model_registry import ModelRegistry
from model_services import base
from model_services.entity_linking import EntityLinkingModelService
from model_services.schemas import (
    Entity,
    EntityRecognitionInputSchema,
    EntityRecognitionResultSchema,
)


@ModelRegistry.register
@ModelRegistry.chain_to(EntityLinkingModelService)
class EntityRecognitionModelService(base.ModelService):
    """A model service which provides Named Entity Recognition (NER) capabilities."""

    _NAME: str = "entity-recognition"
    _HF_MODEL: str = "dslim/bert-base-NER"

    _ner_pipeline: transformers.pipeline

    def __init__(self, pubsub_client: pubsub.PubSubClient):
        super().__init__(model_name=self._NAME, pubsub_client=pubsub_client)
        self._ner_pipeline = transformers.pipeline(
            "ner",
            model=self._HF_MODEL,
            aggregation_strategy="simple",
        )

    def get_input_schema(self) -> type[pydantic.BaseModel]:
        """Get the input schema for the model service."""
        return EntityRecognitionInputSchema

    def get_output_schema(self) -> type[pydantic.BaseModel]:
        """Get the output schema for the model service."""
        return EntityRecognitionResultSchema

    async def _predict(
        self, model_input: EntityRecognitionInputSchema
    ) -> EntityRecognitionResultSchema:
        """Perform inference on the model with the given input."""
        ner_results = self._ner_pipeline(model_input.text)
        entities = [
            Entity(text=result["word"], label=result["entity_group"]) for result in ner_results
        ]
        output = EntityRecognitionResultSchema(
            id=model_input.id, entities=entities, text=model_input.text
        )

        return output
