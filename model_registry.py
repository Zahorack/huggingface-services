import logging
from typing import Any, Callable, ClassVar, Type

import pubsub
from model_services import base


class ModelRegistry:
    """A model service registry.

    The registry also configures the Pub/Sub topics for each model service.
    """

    _model_services: ClassVar[dict[str, Type[base.ModelService]]] = {}
    _pubsub_client: pubsub.PubSubClient | None = None
    _chains: ClassVar[dict[str, str]] = {}

    @classmethod
    def register(cls, model_service_class: Type[base.ModelService]) -> Type[base.ModelService]:
        """Decorator to register a model service class.

        Args:
            model_service_class: The ModelService class to register

        Returns:
            The same class, for use as a decorator
        """
        if not issubclass(model_service_class, base.ModelService):
            raise TypeError(f"{model_service_class.__name__} must inherit from ModelService")

        # Create a temporary instance to get the name
        temp_instance = model_service_class(pubsub_client=None)  # type: ignore
        cls._model_services[temp_instance.name] = model_service_class
        return model_service_class

    @classmethod
    def chain_to(
        cls, target_service_class: Type[base.ModelService]
    ) -> Callable[[Type[base.ModelService]], Type[base.ModelService]]:
        """Decorator to chain one model service to another.

        Args:
            target_service_class: The ModelService class to chain to

        Returns:
            A decorator that registers the chaining relationship
        """

        def decorator(
            source_service_class: Type[base.ModelService],
        ) -> Type[base.ModelService]:
            if not issubclass(source_service_class, base.ModelService):
                raise TypeError(f"{source_service_class.__name__} must inherit from ModelService")

            # Create temporary instances to get the names
            source_instance = source_service_class(pubsub_client=None)  # type: ignore
            target_instance = target_service_class(pubsub_client=None)  # type: ignore

            cls._chains[source_instance.name] = target_instance.name
            return source_service_class

        return decorator

    def __init__(self, pubsub_client: pubsub.PubSubClient):
        """Initialize the registry with a PubSub client.

        Args:
            pubsub_client: The PubSub client to use for model service communication
        """
        self._service_instances = {}
        self._pubsub_client = pubsub_client
        self._initialize_services()

    def _create_chained_callback(
        self,
        source_service: base.ModelService,
        target_service: base.ModelService,
    ) -> Callable[[dict], Any]:
        """Create a callback that chains one service to another.

        Args:
            source_service: The source model service
            target_service: The target model service to chain to

        Returns:
            An async callback function that runs the source service and chains to the target
        """

        async def chained_callback(model_input: dict) -> dict:
            logging.info(
                f"Running {source_service.name} and chaining to {target_service.name} for input: {model_input}"
            )
            source_output = await source_service.predict(model_input)
            await self._pubsub_client.publish(
                topic=target_service.trigger_topic, message=source_output
            )
            return source_output

        return chained_callback

    def _initialize_services(self):
        """Initialize all registered model services and set up their Pub/Sub topics."""
        # Create instances of all registered services
        self._service_instances = {}
        for service_class in self._model_services.values():
            service = service_class(pubsub_client=self._pubsub_client)
            self._service_instances[service.name] = service

        # Set up topics and chaining
        for service_name, service in self._service_instances.items():
            if service_name in self._chains:
                # This service is chained to another service
                target_service = self._service_instances[self._chains[service_name]]
                callback = self._create_chained_callback(service, target_service)
            else:
                # This service runs independently
                callback = service.predict

            self._pubsub_client.register_topic(topic=service.trigger_topic, callback=callback)

    def get_model(self, model_name: str) -> base.ModelService:
        """Get a model service by name.

        Args:
            model_name (str): The name of the model service.

        Returns:
            ModelService: The requested model service.
        """
        if model_name not in self._service_instances:
            raise ValueError(f"Model service '{model_name}' not found.")

        return self._service_instances[model_name]

    @property
    def model_names(self) -> list[str]:
        """Get the names of all registered model services."""
        return list(self._model_services.keys())
