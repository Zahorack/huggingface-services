import pytest

import model_registry
import pubsub
from model_services.entity_recognition import EntityRecognitionModelService


@pytest.mark.asyncio
async def test_ner_model_predicts_entities():
    pubsub_client = pubsub.PubSubClient()
    model_registry.ModelRegistry(pubsub_client=pubsub_client)
    service = EntityRecognitionModelService(pubsub_client=pubsub_client)
    input_data = {
        "id": "doc1",
        "text": "Barack Obama visited Microsoft headquarters in Seattle.",
    }
    output = await service.predict(input_data)

    assert output["id"] == "doc1"
    assert any(e["label"] == "PER" for e in output["entities"])
    assert any("Barack Obama" in e["text"] for e in output["entities"])
