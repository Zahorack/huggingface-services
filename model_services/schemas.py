import uuid
from typing import Literal

import pydantic


class Entity(pydantic.BaseModel):
    """Entity schema for Named Entity Recognition."""

    text: str
    label: str


class WikipediaEntityFound(pydantic.BaseModel):
    """Schema for entities with found Wikipedia entries."""

    text: str
    wikipedia_entries_found: Literal[True]
    title: str
    url: str
    content: str


class SummarizedWikipediaEntity(WikipediaEntityFound):
    """Schema for entities with summarized Wikipedia entries."""

    summary: str


class WikipediaEntityNotFound(pydantic.BaseModel):
    """Schema for entities without Wikipedia entries."""

    text: str
    wikipedia_entries_found: Literal[False]


WikipediaEntity = WikipediaEntityFound | WikipediaEntityNotFound


class EntityRecognitionInputSchema(pydantic.BaseModel):
    """Input schema for the NER model service."""

    id: str = pydantic.Field(default_factory=lambda: str(uuid.uuid4()))
    text: str


class EntityRecognitionResultSchema(pydantic.BaseModel):
    """Output schema for the NER model service."""

    id: str = pydantic.Field(default_factory=lambda: str(uuid.uuid4()))
    text: str
    entities: list[Entity]


class EntityLinkingInputSchema(EntityRecognitionResultSchema):
    """Input schema for the entity linking model service."""

    pass


class EntityLinkingResultSchema(pydantic.BaseModel):
    """Output schema for the entity linking model service."""

    id: str
    text: str
    linked_entities: list[WikipediaEntity]


class EntitySummarizationInputSchema(EntityLinkingResultSchema):
    """Input schema for the entity summarization model service."""

    pass


class EntitySummarizationResultSchema(pydantic.BaseModel):
    """Output schema for the entity summarization model service."""

    id: str
    text: str
    summarized_entities: list[SummarizedWikipediaEntity | WikipediaEntityNotFound]
