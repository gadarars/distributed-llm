# server.py
from fastapi import FastAPI
from pydantic import BaseModel
import multiprocessing as mp

from workers.gpu_worker import ProcessWorkerHandle
from lb.load_balancer import LoadBalancer, Strategy
from master.scheduler import Scheduler
from master.metrics import MetricsCollector
from common.models import Request

app = FastAPI(title="Distributed LLM API")


# Define what the client sends us
class QueryRequest(BaseModel):
    query: str
    user_id: int


# Global state variables
scheduler = None
workers = []
metrics = None


@app.on_event("startup")
def startup_event():
    global scheduler, workers, metrics
    print("Starting up GPU Workers...")

    # Required for Windows multiprocessing
    mp.set_start_method("spawn", force=True)

    # Initialize 4 workers and least-connections routing
    workers = [ProcessWorkerHandle(i) for i in range(4)]
    metrics = MetricsCollector()
    lb = LoadBalancer(workers, strategy=Strategy.LEAST_CONNECTIONS)
    scheduler = Scheduler(lb, metrics)


@app.on_event("shutdown")
def shutdown_event():
    print("Shutting down workers and exporting metrics...")
    if metrics:
        metrics.export_to_excel("API_Server_Run")
    for w in workers:
        w.shutdown()


@app.post("/generate")
def generate_text(req: QueryRequest):
    # 1. Convert the incoming HTTP JSON into your internal Request object
    internal_req = Request(query=req.query, user_id=req.user_id)

    # 2. Let your load balancer and workers process it
    response = scheduler.handle_request(internal_req)

    # 3. Send the result back to the client over the network
    return {
        "request_id": response.request_id,
        "worker_id": response.worker_id,
        "status": response.status.value,
        "latency_ms": response.latency_ms
    }