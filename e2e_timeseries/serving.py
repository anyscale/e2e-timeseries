"""
Online serving script for DLinear model on ETT dataset using Ray Serve.
"""

import argparse
import asyncio
import os

import aiohttp
import numpy as np
import requests
import torch
from fastapi import FastAPI
from models import DLinear
from ray import serve
from starlette.requests import Request

os.environ["RAY_TRAIN_V2_ENABLED"] = "1"

DEPLOYMENT_NAME = "dlinear-ett-server"

# Create a FastAPI app that we can use to add endpoints to our Serve deployment
app = FastAPI(title="DLinear", description="predict future oil temperatures", version="0.1")


# FIXME: update GPU usage later
@serve.deployment(num_replicas=1, ray_actor_options={"num_cpus": 1, "num_gpus": 0})
@serve.ingress(app)
class DLinearModelServe:
    def __init__(self, model_checkpoint_path: str | None = None):
        checkpoint = torch.load(model_checkpoint_path, map_location=torch.device("cpu"))  # Load to CPU first
        self.args = checkpoint["train_args"]
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        # Load model from checkpoint
        self.model = DLinear.Model(self.args).float()
        self.model.load_state_dict(checkpoint["model_state_dict"])
        print(f"Model loaded successfully from {model_checkpoint_path}")

        self.model.to(self.device)
        self.model.eval()

    @serve.batch(max_batch_size=32, batch_wait_timeout_s=0.1)
    async def predict_batch(self, input_data_list: list[dict]) -> list[list[float]]:
        """
        Expects a list of dictionaries, each with a "series" key.
        e.g., [{"series": [0.1, 0.2, ..., 0.N]}, {"series": [0.3, 0.4, ..., 0.M]}]
        where N and M must be equal to self.args.seq_len.
        """
        print(f"Processing batch of size: {len(input_data_list)}")

        batch_series = []
        for item in input_data_list:
            series = item.get("series")
            if series is None or len(series) != self.args["seq_len"]:
                print(f"Warning: Skipping invalid input. Expected series of length {self.args['seq_len']}, got {len(series) if series else 'None'}")

            batch_series.append(series)

        if not batch_series:
            return []

        batch_x_np = np.array(batch_series, dtype=np.float32)
        if self.args["enc_in"] == 1 and batch_x_np.ndim == 2:  # (batch_size, seq_len)
            batch_x_np = np.expand_dims(batch_x_np, axis=-1)  # -> (batch_size, seq_len, 1)

        batch_x = torch.from_numpy(batch_x_np).float().to(self.device)

        with torch.no_grad():
            outputs = self.model(batch_x)

        outputs = outputs[:, -self.args["pred_len"] :, :]

        # Use self.args for features
        if self.args["features"] == "S" and outputs.shape[-1] == 1:
            outputs = outputs.squeeze(-1)

        outputs_list = outputs.cpu().numpy().tolist()
        return outputs_list

    @app.post("/predict")
    async def predict_endpoint(self, request: Request):
        """
        Expects a JSON body with a "series" key.
        e.g., {"series": [0.1, 0.2, ..., 0.N]}
        where N must be equal to self.args.seq_len.
        """
        try:
            input_data = await request.json()
            if not isinstance(input_data, dict) or "series" not in input_data:
                return {"error": "Invalid input. JSON object with 'series' key expected."}
            if not isinstance(input_data["series"], list):
                return {"error": "Invalid input. 'series' must be a list."}
            # Use self.args for seq_len
            if len(input_data["series"]) != self.args["seq_len"]:
                return {"error": f"Invalid series length. Expected {self.args['seq_len']}, got {len(input_data['series'])}."}

        except Exception as e:
            return {"error": f"Failed to parse JSON request: {str(e)}"}

        prediction_list = await self.predict_batch([input_data])
        # If predict_batch can return an empty list or error indicators, handle them.
        # For now, assuming it returns a list with one item for a single input.
        return prediction_list[0] if prediction_list else {"error": "Prediction failed for the input."}

    # Expose get_seq_len as a GET endpoint
    @app.get("/seq_len")
    async def get_sequence_length(self):
        return {"seq_len": self.args["seq_len"]}


def serve_model(model_checkpoint_path_arg: str):
    dlinear_app = DLinearModelServe.bind(model_checkpoint_path=model_checkpoint_path_arg)

    # The route_prefix will apply to all routes within the FastAPI app
    serve.run(dlinear_app, name=DEPLOYMENT_NAME, route_prefix="/predict_dlinear")
    print(f"DLinear model deployment '{DEPLOYMENT_NAME}' is running with FastAPI app.")
    print("  Prediction endpoint: http://127.0.0.1:8000/predict_dlinear/predict")
    print("  Sequence length endpoint: http://127.0.0.1:8000/predict_dlinear/seq_len")

    print("\nTo stop the server, press Ctrl+C in the terminal where it's running.")


def test_serve():
    # --- Example Client Code (can be run in a separate script or after serve starts) ---

    # Base URL for the service
    base_url = "http://127.0.0.1:8000/predict_dlinear"
    seq_len_url = f"{base_url}/seq_len"
    predict_url = f"{base_url}/predict"

    # Get seq_len from the new endpoint
    seq_len = None
    print("\n--- Fetching Sequence Length ---")
    try:
        response = requests.get(seq_len_url)
        response.raise_for_status()
        seq_len_data = response.json()
        seq_len = seq_len_data.get("seq_len")
        if seq_len is None:
            print(f"Error: 'seq_len' not found in response from {seq_len_url}. Response: {seq_len_data}")
            return  # Cannot proceed without seq_len
        print(f"Successfully fetched seq_len: {seq_len}")
    except requests.exceptions.RequestException as e:
        print(f"Error fetching seq_len from {seq_len_url}: {e}")
        if hasattr(e, "response") and e.response is not None:
            print(f"Response content: {e.response.text}")
        return  # Cannot proceed

    # FIXME: load real data from the ETTh2 dataset

    sample_input_series = [0.5 + np.sin(i / 10) for i in range(seq_len)]
    sample_request_body = {"series": sample_input_series}

    print("\n--- Sending Single Synchronous Request to /predict endpoint ---")
    try:
        response = requests.post(predict_url, json=sample_request_body)
        response.raise_for_status()  # Raise an exception for HTTP errors
        prediction = response.json()
        print(f"Prediction (first {min(5, len(prediction))} values): {prediction[:5] if isinstance(prediction, list) else prediction}")
        print(f"Full prediction length: {len(prediction) if isinstance(prediction, list) else 'N/A (error likely)'}")
    except requests.exceptions.RequestException as e:
        print(f"Error sending request to {predict_url}: {e}")
        if hasattr(e, "response") and e.response is not None:
            print(f"Response content: {e.response.text}")

    # --- Example for Batch Asynchronous Requests (using aiohttp) ---
    # This also needs to be run against an active serve deployment.

    print("\n--- Sending Batch Asynchronous Requests to /predict endpoint ---")
    sample_input_list = [sample_request_body] * 5  # Batch of 5 identical requests

    async def fetch(session, url, data):
        async with session.post(url, json=data) as response:
            if response.status != 200:
                print(f"Error from server ({url}): {response.status}, {await response.text()}")
                return None
            return await response.json()

    async def fetch_all_concurrently(requests_to_send: list):
        async with aiohttp.ClientSession() as session:
            # Use the correct predict_url for async requests
            tasks = [fetch(session, predict_url, input_data) for input_data in requests_to_send]
            responses = await asyncio.gather(*tasks, return_exceptions=True)
            return responses

    async def run_async_requests():
        responses = await fetch_all_concurrently(sample_input_list)
        for i, res in enumerate(responses):
            if isinstance(res, Exception):
                print(f"Request {i + 1} failed: {res}")
            elif res is None:
                print(f"Request {i + 1} returned no response (check server logs).")
            else:
                print(f"Response {i + 1} (first {min(5, len(res))} values): {res[:5] if isinstance(res, list) else res}")

    asyncio.run(run_async_requests())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Serve DLinear model with Ray Serve.")
    parser.add_argument("--checkpoint_path", type=str, required=True, help="Path to the model checkpoint file (.pt).")
    args = parser.parse_args()

    serve_model(args.checkpoint_path)
    test_serve()
