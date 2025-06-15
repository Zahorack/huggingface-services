import json
import logging
from typing import Any, Callable


class PubSubClient:
    _message_queues: dict[str, Callable[[dict], Any] | None]
    _last_responses: dict[str, dict]

    def __init__(self):
        self._message_queues = {}
        self._last_responses = {}

    def register_topic(self, topic: str, callback: Callable[[dict], Any]) -> None:
        """Register a callback for a specific topic."""
        if topic not in self._message_queues:
            logging.info(f"Registering topic: {topic}")
            self._message_queues[topic] = None
        else:
            logging.info(f"Skipping registration for existing topic: {topic}")
        self._message_queues[topic] = callback

    async def publish(
        self,
        topic: str,
        message: dict,
        message_id: str = "no-id",
    ) -> dict:
        """Publish a message to a specific topic."""
        if topic not in self._message_queues:
            raise ValueError(f"Topic {topic} not registered.")
        callback = self._message_queues[topic]
        if callback is None:
            raise ValueError(f"No callback registered for topic {topic}.")
        response = await callback(message)
        logging.info(f"Message {message_id} published to topic {topic}")
        logging.info(json.dumps(response))
        self._last_responses[topic] = response
        return response

    def get_last_response(self, topic: str) -> dict:
        """Get the last response for a specific topic."""
        return self._last_responses.get(topic, {})
