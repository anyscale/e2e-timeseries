"""
Online serving script for DLinear model on ETT dataset using Ray Serve.
"""
import asyncio
import os
import argparse

import aiohttp
import numpy as np
import requests
import torch
from models import DLinear
from ray import serve
from starlette.requests import Request

os.environ["RAY_TRAIN_V2_ENABLED"] = "1"

DEPLOYMENT_NAME = "dlinear-ett-server"

# FIXME: update GPU usage later
@serve.deployment(num_replicas=1, ray_actor_options={"num_cpus": 1, "num_gpus": 0})
class DLinearModelServe:
    def __init__(self, model_checkpoint_path: str | None = None):
        checkpoint = torch.load(model_checkpoint_path, map_location=torch.device("cpu")) # Load to CPU first
        loaded_train_args = checkpoint["train_args"]
        self._seq_len = loaded_train_args["seq_len"]
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        # Load model from checkpoint
        self.model = DLinear.Model(loaded_train_args).float()
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
            if series is None or len(series) != self.args.seq_len:
                # Handle error or return a specific response for malformed input
                # For now, we'll raise an error or skip.
                # In a production system, you might return an error code.
                print(f"Warning: Skipping invalid input. Expected series of length {self.args.seq_len}, got {len(series) if series else 'None'}")
                # Placeholder for actual prediction list to maintain batch structure
                # Or, filter out invalid inputs before batching (more complex with serve.batch)
                # For simplicity, we'll try to predict even if it might fail for this item.
                # A robust way is to return error markers.
                # For this example, let's assume valid inputs or make it fail clearly.
                if series is None: 
                    series = [0.0] * self.args.seq_len # Dummy to avoid immediate crash
                elif len(series) != self.args.seq_len:
                    series = series[:self.args.seq_len] + [0.0]*(self.args.seq_len - len(series)) # Pad/truncate

            batch_series.append(series)

        if not batch_series:
            return []

        # Convert list of series to a PyTorch tensor
        # Input shape for DLinear: (batch_size, seq_len, num_features)
        # For univariate 'S' features, num_features is 1.
        # The input "series" is just the time sequence.
        # We need to reshape it to (batch_size, seq_len, 1) if enc_in is 1
        
        batch_x_np = np.array(batch_series, dtype=np.float32)
        if self.args.enc_in == 1 and batch_x_np.ndim == 2: # (batch_size, seq_len)
            batch_x_np = np.expand_dims(batch_x_np, axis=-1) # -> (batch_size, seq_len, 1)
        
        batch_x = torch.from_numpy(batch_x_np).float().to(self.device)

        with torch.no_grad():
            # Model expects (Batch, Input Length, Input Features)
            outputs = self.model(batch_x)  # Shape (Batch, Pred Length, Output Features)

        # Extract predictions based on model config
        # For DLinear, f_dim is usually -1 for MS and 0 for S,
        # but the model's output is typically (Batch, Pred Len, NumTargetFeatures)
        # If 'S' and c_out=1, then (Batch, Pred Len, 1)
        outputs = outputs[:, -self.args.pred_len:, :] # Get the prediction part

        if self.args.features == "S" and outputs.shape[-1] == 1:
            outputs = outputs.squeeze(-1) # Shape (Batch, Pred Len)
        
        outputs_list = outputs.cpu().numpy().tolist()
        return outputs_list

    async def __call__(self, request: Request):
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
            if len(input_data["series"]) != self.args.seq_len:
                return {"error": f"Invalid series length. Expected {self.args.seq_len}, got {len(input_data['series'])}."}

        except Exception as e:
            return {"error": f"Failed to parse JSON request: {str(e)}"}
        
        # predict_batch expects a list of dictionaries
        prediction_list = await self.predict_batch([input_data])
        return prediction_list[0] # Return the single prediction list
    
    
    def get_seq_len(self):
        return self._seq_len


def serve_model(model_checkpoint_path_arg: str):
    dlinear_deployment = DLinearModelServe.bind(
        model_checkpoint_path=model_checkpoint_path_arg
    )
    
    serve.run(dlinear_deployment, name=DEPLOYMENT_NAME, route_prefix="/predict_dlinear")
    print(f"DLinear model deployment '{dlinear_deployment.name}' is running at route prefix '/predict_dlinear'")

    print("\nTo stop the server, press Ctrl+C in the terminal where it's running.")

def test_serve():
    # FIXME: get a real datapoint from the ETTh2 dataset
    # --- Example Client Code (can be run in a separate script or after serve starts) ---


    dlinear_deployment = serve.get_deployment(DEPLOYMENT_NAME)
    seq_len = dlinear_deployment.get_seq_len()


    sample_input_series = [0.5 + np.sin(i / 10) for i in range(seq_len)]
    sample_request_body = {"series": sample_input_series}

    # The URL for the deployed model
    url = "http://127.0.0.1:8000/predict_dlinear"

    # asyncio.run(asyncio.sleep(1)) # Give serve time to start

    # Example with a single request (synchronous call using requests)
    # This would typically be run *after* the serve application has started.
    # If you run this script directly, serve.run() is blocking.
    # You'd need to run the client part in a separate process/terminal or use `serve.start()`.
    
    print("\n--- Sending Single Synchronous Request ---")
    try:
        response = requests.post(url, json=sample_request_body)
        response.raise_for_status() # Raise an exception for HTTP errors
        prediction = response.json()
        print(f"Prediction (first {min(5, len(prediction))} values): {prediction[:5] if isinstance(prediction, list) else prediction}")
        print(f"Full prediction length: {len(prediction) if isinstance(prediction, list) else 'N/A (error likely)'}")
    except requests.exceptions.RequestException as e:
        print(f"Error sending request: {e}")
        if hasattr(e, 'response') and e.response is not None:
            print(f"Response content: {e.response.text}")


    # --- Example for Batch Asynchronous Requests (using aiohttp) ---
    # This also needs to be run against an active serve deployment.
    
    # print("\n--- Sending Batch Asynchronous Requests ---")
    sample_input_list = [sample_request_body] * 5 # Batch of 5 identical requests

    async def fetch(session, url, data):
        async with session.post(url, json=data) as response:
            if response.status != 200:
                print(f"Error from server: {response.status}, {await response.text()}")
                return None
            return await response.json()

    async def fetch_all_concurrently(requests_to_send: list):
        async with aiohttp.ClientSession() as session:
            tasks = [fetch(session, url, input_data) for input_data in requests_to_send]
            responses = await asyncio.gather(*tasks, return_exceptions=True)
            return responses

    async def run_async_requests():
        responses = await fetch_all_concurrently(sample_input_list)
        for i, res in enumerate(responses):
            if isinstance(res, Exception):
                print(f"Request {i+1} failed: {res}")
            elif res is None:
                print(f"Request {i+1} returned no response (check server logs).")
            else:
                print(f"Response {i+1} (first {min(5, len(res))} values): {res[:5] if isinstance(res, list) else res}")
    
    try:
        asyncio.run(run_async_requests())
    except RuntimeError as e:
        if "cannot be called from a running event loop" in str(e):
            print("Note: Async client example cannot run directly if main event loop is already running (e.g. in Jupyter).")
            print("Run the client part in a separate script or environment.")
        else:
            raise

    


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Serve DLinear model with Ray Serve.")
    parser.add_argument(
        "--model_checkpoint_path",
        type=str,
        required=True,
        help="Path to the model checkpoint file (.pt)."
    )
    args = parser.parse_args()

    serve_model(args.model_checkpoint_path)
    test_serve()

