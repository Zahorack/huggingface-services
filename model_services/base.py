import abc
import json
import logging

import pydantic

import pubsub


class ModelService(abc.ABC):
    """Base class for a model service.

    This abstraction represents a model service that can serve inference on-demand and
    as part of a Pub/Sub-triggered pipeline.
    """

    _model_name: str
    _trigger_topic: str
    _pubsub_client: pubsub.PubSubClient

    def __init__(self, model_name: str, pubsub_client: pubsub.PubSubClient):
        self._model_name = model_name
        self._trigger_topic = f"{model_name}-trigger"
        self._pubsub_client = pubsub_client

    @abc.abstractmethod
    def get_input_schema(self) -> type[pydantic.BaseModel]:
        """Get the input schema for the model.

        Returns:
            pydantic.BaseModel: Input schema for the model.
        """
        pass

    @abc.abstractmethod
    def get_output_schema(self) -> type[pydantic.BaseModel]:
        """Get the output schema for the model.

        Returns:
            pydantic.BaseModel: Output schema for the model.
        """
        pass

    @abc.abstractmethod
    async def _predict(self, model_input: pydantic.BaseModel) -> pydantic.BaseModel:
        """Perform inference on the model with the given input.

        Args:
            model_input (dict): Input data for the model.

        Returns:
            dict: Model prediction output.
        """
        pass

    async def predict(self, model_input: dict) -> dict:
        """Run the model prediction.

        Args:
            model_input (dict): Input data for the model.

        Returns:
            dict: Model prediction output.
        """
        logging.info(f"Running prediction for model: {self._model_name}")
        logging.info(f"Model input: {json.dumps(model_input)}")
        schematized_input = self.get_input_schema()(**model_input)
        schematized_output = await self._predict(schematized_input)
        raw_output = schematized_output.model_dump()
        logging.info(f"Model output: {json.dumps(raw_output)}")
        return raw_output

    @property
    def name(self) -> str:
        """Get the name of the model service."""
        return self._model_name

    @property
    def trigger_topic(self) -> str:
        """Get the Pub/Sub topic for triggering the model."""
        return self._trigger_topic

    @property
    def pubsub_client(self) -> pubsub.PubSubClient:
        """Get the Pub/Sub client for the model service."""
        return self._pubsub_client
