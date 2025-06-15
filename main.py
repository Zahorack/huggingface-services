import asyncio
import json
import logging
import uuid
from datetime import datetime, timezone

import fastapi
import pydantic
import uvicorn

import model_registry
import pubsub

# Import model services here to ensure they are registered with the model registry
from model_services.entity_linking import EntityLinkingModelService  # noqa:
from model_services.entity_recognition import EntityRecognitionModelService  # noqa:
from model_services.summarization import EntitySummarizationModelService  # noqa:

logging.basicConfig(level=logging.INFO)


app = fastapi.FastAPI()
pubsub_client = pubsub.PubSubClient()
model_registry = model_registry.ModelRegistry(pubsub_client)


class PublishRequest(pydantic.BaseModel):
    """Request model for publishing messages to a topic."""

    topic: str
    payload: dict
    ts: datetime = datetime.now(tz=timezone.utc)


@app.get("/ping")
def ping():
    """Health check endpoint."""
    return {"message": "Pong!"}


@app.get("/models")
async def list_models():
    """Endpoint to list all available models."""
    logging.info("Received request to list models.")
    try:
        return {"models": model_registry.model_names}
    except Exception as e:
        logging.error(f"Error listing models: {str(e)}")
        raise fastapi.HTTPException(status_code=500, detail="Internal Server Error")


@app.post("/{model_name}/predict")
async def predict(model_name: str, request: fastapi.Request):
    """Endpoint to trigger model prediction."""
    logging.info(f"Received prediction request for model: {model_name}")
    model_input = await request.json()

    try:
        model_service = model_registry.get_model(model_name)
        response = await model_service.predict(model_input)
        return response
    except ValueError as e:
        logging.error(f"Model not found: {model_name}")
        raise fastapi.HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logging.error(f"Error during prediction: {str(e)}")
        raise fastapi.HTTPException(status_code=500, detail="Internal Server Error")


@app.post("/publish")
async def publish(request: PublishRequest):
    """Publish a message to the Pub/Sub topic."""
    message_id = uuid.uuid4()
    topic = request.topic
    payload = request.payload
    ts = request.ts

    logging.info(f"Publishing message with ID: {message_id} to topic: {topic} at {str(ts)}")
    logging.info(json.dumps(payload))

    try:
        await pubsub_client.publish(topic=topic, message=payload)
        return {
            "message_id": str(message_id),
            "topic": topic,
            "payload": payload,
            "timestamp": ts.isoformat(),
        }
    except ValueError as e:
        logging.error(f"Error publishing message: {str(e)}")
        raise fastapi.HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logging.error(f"Error during publish: {str(e)}")
        raise fastapi.HTTPException(status_code=500, detail="Internal Server Error")


@app.get("/{topic}/last_response")
def last_response(topic: str):
    """Get the last response for a specific topic."""
    logging.info(f"Fetching last response for topic: {topic}")
    try:
        response = pubsub_client.get_last_response(topic)
        if not response:
            raise fastapi.HTTPException(status_code=404, detail="No response found")
        return response
    except Exception as e:
        logging.error(f"Error fetching last response: {str(e)}")
        raise fastapi.HTTPException(status_code=500, detail="Internal Server Error")


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
