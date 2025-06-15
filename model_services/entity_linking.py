import logging

import pydantic
import wikipedia
from wikipedia.exceptions import DisambiguationError, PageError

import pubsub
from model_registry import ModelRegistry
from model_services import base
from model_services.schemas import (
    EntityLinkingInputSchema,
    EntityLinkingResultSchema,
    WikipediaEntity,
)
from model_services.summarization import EntitySummarizationModelService


@ModelRegistry.register
@ModelRegistry.chain_to(EntitySummarizationModelService)
class EntityLinkingModelService(base.ModelService):
    """A model service which provides entity linking capabilities."""

    _NAME: str = "entity-linking"
    _MAX_CONTENT_LENGTH: int = 1024

    def __init__(self, pubsub_client: pubsub.PubSubClient):
        super().__init__(model_name=self._NAME, pubsub_client=pubsub_client)

    def get_input_schema(self) -> type[pydantic.BaseModel]:
        """Get the input schema for the model service."""
        return EntityLinkingInputSchema

    def get_output_schema(self) -> type[pydantic.BaseModel]:
        """Get the output schema for the model service."""
        return EntityLinkingResultSchema

    async def _predict(self, model_input: EntityLinkingInputSchema) -> EntityLinkingResultSchema:

        linked_entities = []
        for entity in model_input.entities:
            entry = {"text": entity.text, "wikipedia_entries_found": False}

            try:
                page = wikipedia.page(entity.text, auto_suggest=False)
                entry.update(
                    {
                        "wikipedia_entries_found": True,
                        "title": page.title,
                        "url": page.url,
                        "content": page.content[: self._MAX_CONTENT_LENGTH] + "...",
                    }
                )
            except DisambiguationError as e:
                logging.warning(f"DisambiguationError: {e}")
                first_option = e.options[0]
                page = wikipedia.page(first_option, auto_suggest=False)
                entry.update(
                    {
                        "wikipedia_entries_found": True,
                        "title": page.title,
                        "url": page.url,
                        "content": page.content[: self._MAX_CONTENT_LENGTH] + "...",
                    }
                )

            except PageError as e:
                logging.error(f"PageError: {e}")
                entry["wikipedia_entries_found"] = False

            except Exception as e:
                logging.error(f"Exception: {e}")
                entry["wikipedia_entries_found"] = False

            entity_adapter = pydantic.TypeAdapter(WikipediaEntity)
            linked_entities.append(entity_adapter.validate_python(entry))

        return EntityLinkingResultSchema(
            id=model_input.id,
            linked_entities=linked_entities,
            text=model_input.text,
        )
