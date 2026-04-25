# master/scheduler.py
# Control-plane entry point. Receives requests and sends them to the load balancer.

from common.models    import Request, Response
from common.logger    import get_logger
from lb.load_balancer import LoadBalancer
from master.metrics   import MetricsCollector

log = get_logger("Master.Scheduler")


class Scheduler:

    def __init__(self, load_balancer: LoadBalancer, metrics: MetricsCollector):
        self.lb = load_balancer
        self.metrics = metrics

    def handle_request(self, request: Request) -> Response:
        log.debug(
            "Received req=%s user=%d alive_workers=%d/%d",
            request.request_id, request.user_id,
            self.lb.alive_count(), self.lb.total_count(),
        )
        response = self.lb.dispatch(request)
        self.metrics.record(response)
        return response
