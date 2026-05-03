#!/usr/bin/env python3
"""
Plan B: Colab cloud inference server for SmolVLA.
Run this on a Colab T4 GPU, expose via ngrok, and call from local machine.

Usage (on Colab):
    !pip install pyngrok fastapi uvicorn
    !python cloud_server.py --model-path /content/drive/MyDrive/smolvla_cup/checkpoint_20000

Usage (local client):
    curl -X POST http://<ngrok-url>/predict \
      -H "Content-Type: application/json" \
      -d '{"image_b64": "...", "state": [...], "task": "Pick up the water cup..."}'
"""

import argparse
import base64
import io
import json
import time

import numpy as np
import torch
from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn


app = FastAPI(title="SmolVLA Cloud Inference")
policy = None


class PredictRequest(BaseModel):
    image_b64: str  # Base64-encoded JPEG image
    state: list  # Robot joint positions [6 floats]
    task: str  # Task instruction string


class PredictResponse(BaseModel):
    actions: list  # Predicted action chunk
    inference_ms: float


@app.post("/predict", response_model=PredictResponse)
async def predict(req: PredictRequest):
    """Run SmolVLA inference on GPU and return action chunk."""
    t0 = time.perf_counter()

    # Decode image
    from PIL import Image
    img_bytes = base64.b64decode(req.image_b64)
    image = Image.open(io.BytesIO(img_bytes))
    image_array = np.array(image)

    # Build observation dict
    observation = {
        "front": image_array,
        "state": np.array(req.state, dtype=np.float32),
        "task": req.task,
    }

    # Run inference
    with torch.no_grad():
        action = policy.predict(observation)

    dt_ms = (time.perf_counter() - t0) * 1000

    return PredictResponse(
        actions=action.tolist() if hasattr(action, "tolist") else action,
        inference_ms=round(dt_ms, 1),
    )


@app.get("/health")
async def health():
    return {"status": "ok", "model_loaded": policy is not None}


def main():
    global policy

    parser = argparse.ArgumentParser(description="SmolVLA cloud inference server")
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--ngrok-token", type=str, default=None,
                        help="ngrok auth token (or set NGROK_AUTH_TOKEN env var)")
    args = parser.parse_args()

    # Load model on GPU
    import os
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading model on {device}...")

    from lerobot.common.policies import make_policy
    policy = make_policy(policy_path=args.model_path, device=device)
    policy.eval()
    print("Model loaded.")

    # Setup ngrok tunnel
    ngrok_token = args.ngrok_token or os.environ.get("NGROK_AUTH_TOKEN")
    if ngrok_token:
        from pyngrok import ngrok
        ngrok.set_auth_token(ngrok_token)
        tunnel = ngrok.connect(args.port)
        print(f"\n{'='*50}")
        print(f"  PUBLIC URL: {tunnel.public_url}")
        print(f"{'='*50}\n")
        print("Use this URL on your local machine to send requests.")
    else:
        print(f"\nRunning on http://0.0.0.0:{args.port}")
        print("Set --ngrok-token or NGROK_AUTH_TOKEN for public access.")

    uvicorn.run(app, host="0.0.0.0", port=args.port)


if __name__ == "__main__":
    main()
