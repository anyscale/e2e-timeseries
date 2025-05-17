"""
Online serving script for DLinear model on ETT dataset using Ray Serve.
"""

import argparse
import asyncio
import os

import aiohttp
import numpy as np
import pandas as pd
import requests
import torch
from fastapi import FastAPI
from ray import serve
from starlette.requests import Request

from e2e_timeseries.models import DLinear

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
    async def predict_batch(self, batch_x: list[list[float]]) -> list[list[float]]:
        """
        Expects a list of series, where each series is a 1D list of floats/integers.
        e.g., [[0.1, 0.2, ..., 0.N], [0.3, 0.4, ..., 0.M]]
        Each series is a 1D list of floats/integers.
        """

        # Convert list of 1D series to a 2D numpy array (batch_size, seq_len)
        batch_x = np.array(batch_x, dtype=np.float32)
        batch_x = torch.from_numpy(batch_x).float().to(self.device)

        # Ensure batch_x is 3D: (batch_size, seq_len, num_features)
        # For univariate 'S' models, num_features will be 1.
        if batch_x.ndim == 2:
            batch_x = batch_x.unsqueeze(-1)

        with torch.no_grad():
            outputs = self.model(batch_x)
            # model output: (batch_size, pred_len, features_out)

        # Slice to get the prediction length part of the output.
        # The [:, :, :] part takes all output features.
        # For 'S' (single-feature) forecasting, DLinear typically outputs 1 feature.
        # For 'M' (multi-feature) forecasting, DLinear typically outputs multiple features.
        outputs = outputs[:, -self.args["pred_len"] :, :]

        # If 'S' (single feature forecasting) and the model's output for that single
        # feature has an explicit last dimension of 1, squeeze it.
        # This makes the output a list of 1D series (list of lists of floats).
        if outputs.shape[-1] == 1:
            outputs = outputs.squeeze(-1)  # Shape: (batch_size, pred_len)

        outputs_list = outputs.cpu().numpy().tolist()
        return outputs_list

    @app.post("/predict")
    async def predict_endpoint(self, request: Request):
        """
        Expects a JSON body which is a list of floats/integers.
        e.g., [0.1, 0.2, ..., 0.N]
        where N must be equal to self.args.seq_len.
        """
        try:
            input_data = await request.json()
            if not isinstance(input_data, list):
                return {"error": "Invalid input. JSON list of numbers expected."}
            # Use self.args for seq_len
            if len(input_data) != self.args["seq_len"]:
                return {"error": f"Invalid series length. Expected {self.args['seq_len']}, got {len(input_data)}."}

        except Exception as e:
            return {"error": f"Failed to parse JSON request: {str(e)}"}

        # Pass the single list input_data, wrapped in another list, to predict_batch.
        # Ray Serve's @serve.batch will handle collecting these into a batch for predict_batch.
        # The await call will return the specific result for this input_data.
        single_prediction_output = await self.predict_batch(input_data)

        # single_prediction_output is expected to be a list[float] (the prediction for one series)
        return single_prediction_output

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

    # Get the proper seq_len for the deployed model
    response = requests.get(seq_len_url)
    response.raise_for_status()
    seq_len_data = response.json()
    seq_len = seq_len_data.get("seq_len")

    # Load sample data for demonstration purposes
    df = pd.read_csv("e2e_timeseries/dataset/ETTh2.csv")
    ot_series = df["OT"].tolist()

    # Create a single sample request from the loaded data
    sample_input_series = ot_series[:seq_len]
    sample_request_body = sample_input_series

    print("\n--- Sending Single Synchronous Request to /predict endpoint ---")
    response = requests.post(predict_url, json=sample_request_body)
    response.raise_for_status()
    prediction = response.json()
    print(f"Prediction (first 5 values): {prediction[:5]}")

    print("\n--- Sending Batch Asynchronous Requests to /predict endpoint ---")
    sample_input_list = [sample_input_series] * 100  # Use identical requests

    async def fetch(session, url, data):
        async with session.post(url, json=data) as response:
            return await response.json()

    async def fetch_all_concurrently(requests_to_send: list):
        async with aiohttp.ClientSession() as session:
            tasks = [fetch(session, predict_url, input_data) for input_data in requests_to_send]
            responses = await asyncio.gather(*tasks, return_exceptions=True)
            return responses

    asyncio.run(fetch_all_concurrently(sample_input_list))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Serve DLinear model with Ray Serve.")
    parser.add_argument("--checkpoint_path", type=str, required=True, help="Path to the model checkpoint file (.pt).")
    args = parser.parse_args()

    serve_model(args.checkpoint_path)
    test_serve()
