from fastapi import FastAPI, Request
from prometheus_client import Counter, Histogram, generate_latest
from prometheus_client import CONTENT_TYPE_LATEST
from fastapi import FastAPI
from prometheus_fastapi_instrumentator import Instrumentator
import time
import pickle
import numpy as np

# Initialize FastAPI app
app = FastAPI()

# Load your trained model
with open("app/model.pkl", "rb") as f:
    model = pickle.load(f)

# Define Prometheus metrics
REQUEST_COUNT = Counter('request_count', 'Total number of requests')
REQUEST_LATENCY = Histogram('request_latency_seconds', 'Latency of requests in seconds')


app = FastAPI()

@app.get("/")
def read_root():
    return {"message": "Hello, world"}

# Enable metrics
Instrumentator().instrument(app).expose(app)


# Inference endpoint
@app.post("/predict")
async def predict(request: Request):
    REQUEST_COUNT.inc()
    start_time = time.time()

    data = await request.json()
    features = np.array(data["features"]).reshape(1, -1)  # Expecting {"features": [..]}

    prediction = model.predict(features)
    latency = time.time() - start_time
    REQUEST_LATENCY.observe(latency)

    return {"prediction": prediction.tolist()}

# Expose metrics for Prometheus
@app.get("/metrics")
async def metrics():
    return generate_latest(), 200, {'Content-Type': CONTENT_TYPE_LATEST}
