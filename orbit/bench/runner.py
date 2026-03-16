from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from dataclasses import asdict, dataclass, field

import httpx

from orbit.bench.workloads import WORKLOAD_GENERATORS, WorkloadConfig
from orbit.common.schemas import InferenceRequest, InferenceResponse

logger = logging.getLogger("orbit.bench")


@dataclass
class RequestResult:
    request_idx: int
    router_name: str
    replica_id: str
    total_ms: float
    ttft_ms: float  # time to first token (prefill + queue)
    prefill_ms: float
    decode_ms: float
    queue_ms: float
    cached_tokens: int
    prompt_tokens: int
    completion_tokens: int
    success: bool
    error: str = ""


@dataclass
class BenchmarkResult:
    router_name: str
    workload_name: str
    num_requests: int
    total_time_s: float
    results: list[RequestResult] = field(default_factory=list)


async def reset_all_caches(replica_urls: list[str], router_url: str = "") -> None:
    """Reset KV caches on all replicas and clear router cache registry."""
    async with httpx.AsyncClient(timeout=10.0) as client:
        tasks = []
        for url in replica_urls:
            tasks.append(client.post(f"{url}/v1/cache/reset"))
        # Also clear router's cache registry if available
        if router_url:
            tasks.append(client.post(f"{router_url}/v1/cache/reset"))
        results = await asyncio.gather(*tasks, return_exceptions=True)
        for i, r in enumerate(results):
            if isinstance(r, Exception):
                logger.debug(f"Cache reset failed for endpoint {i}: {r}")
            else:
                logger.info(f"Cache reset: {r.status_code}")


async def run_single_request(
    client: httpx.AsyncClient,
    router_url: str,
    request: InferenceRequest,
    idx: int,
    router_name: str,
) -> RequestResult:
    start = time.monotonic()
    try:
        resp = await client.post(
            f"{router_url}/v1/chat/completions",
            json=request.model_dump(),
        )
        resp.raise_for_status()
        data = resp.json()
        elapsed = (time.monotonic() - start) * 1000

        response = InferenceResponse(**data)

        return RequestResult(
            request_idx=idx,
            router_name=router_name,
            replica_id=response.orbit_replica_id or "unknown",
            total_ms=elapsed,
            ttft_ms=response.orbit_prefill_ms + response.orbit_queue_ms,
            prefill_ms=response.orbit_prefill_ms,
            decode_ms=response.orbit_decode_ms,
            queue_ms=response.orbit_queue_ms,
            cached_tokens=response.orbit_cached_tokens,
            prompt_tokens=response.usage.prompt_tokens,
            completion_tokens=response.usage.completion_tokens,
            success=True,
        )

    except Exception as e:
        elapsed = (time.monotonic() - start) * 1000
        return RequestResult(
            request_idx=idx,
            router_name=router_name,
            replica_id="error",
            total_ms=elapsed,
            ttft_ms=0,
            prefill_ms=0,
            decode_ms=0,
            queue_ms=0,
            cached_tokens=0,
            prompt_tokens=0,
            completion_tokens=0,
            success=False,
            error=str(e),
        )


async def set_router_strategy(router_url: str, strategy: str) -> None:
    """Tell the router which routing strategy to use."""
    async with httpx.AsyncClient(timeout=10.0) as client:
        resp = await client.post(
            f"{router_url}/v1/strategy",
            json={"strategy": strategy},
        )
        resp.raise_for_status()
        logger.info(f"Router strategy set to: {strategy}")


async def run_benchmark(
    requests: list[InferenceRequest],
    router_url: str,
    workload_name: str,
    router_name: str = "orbit",
    arrival_rate: float = 10.0,
) -> BenchmarkResult:
    """Run a benchmark with all requests going through the router."""
    logger.info(f"Running {router_name}/{workload_name}: {len(requests)} requests")

    start = time.monotonic()

    async with httpx.AsyncClient(timeout=120.0) as client:
        interval = 1.0 / arrival_rate if arrival_rate > 0 else 0
        tasks: list[asyncio.Task] = []

        for i, req in enumerate(requests):
            task = asyncio.create_task(
                run_single_request(client, router_url, req, i, router_name)
            )
            tasks.append(task)

            if interval > 0 and i < len(requests) - 1:
                await asyncio.sleep(interval)

        results = await asyncio.gather(*tasks)

    total_time = time.monotonic() - start

    return BenchmarkResult(
        router_name=router_name,
        workload_name=workload_name,
        num_requests=len(requests),
        total_time_s=total_time,
        results=list(results),
    )


STRATEGIES_TO_BENCH = ["orbit", "round_robin", "random", "hash_based", "least_loaded"]


async def run_single_strategy(
    router_url: str,
    replica_urls: list[str],
    strategy: str,
    workload_name: str = "chat",
    workload_config: WorkloadConfig | None = None,
    output_dir: str = "results",
) -> BenchmarkResult:
    """Run benchmark for a single strategy.

    Caller is responsible for ensuring caches are cold (e.g. restarting
    llama.cpp instances) before calling this.
    """
    if workload_config is None:
        workload_config = WorkloadConfig()

    workload_dir = os.path.join(output_dir, workload_name)
    os.makedirs(workload_dir, exist_ok=True)

    generator = WORKLOAD_GENERATORS[workload_name]
    requests = generator(workload_config)
    logger.info(
        f"Generated {len(requests)} requests for workload '{workload_name}' "
        f"(seed={workload_config.seed})"
    )

    # Reset shadow caches (router registry + replica shadow caches)
    await reset_all_caches(replica_urls, router_url)
    await asyncio.sleep(0.5)

    # Switch router strategy
    await set_router_strategy(router_url, strategy)

    result = await run_benchmark(
        requests, router_url, workload_name, strategy,
        arrival_rate=workload_config.arrival_rate,
    )

    # Save result
    path = os.path.join(workload_dir, f"{strategy}.json")
    with open(path, "w") as f:
        json.dump(
            {
                "router_name": result.router_name,
                "workload_name": result.workload_name,
                "num_requests": result.num_requests,
                "total_time_s": result.total_time_s,
                "results": [asdict(r) for r in result.results],
            },
            f,
            indent=2,
        )
    logger.info(f"Saved results to {path}")

    return result


async def main():
    logging.basicConfig(level=logging.INFO)

    router_url = os.environ.get("ORBIT_ROUTER_URL", "http://orbit-router:8000")
    replica_urls_str = os.environ.get(
        "ORBIT_REPLICA_URLS",
        "http://orbit-replica-1:8001,http://orbit-replica-2:8001,"
        "http://orbit-replica-3:8001,http://orbit-replica-4:8001",
    )
    replica_urls = [u.strip() for u in replica_urls_str.split(",") if u.strip()]
    workload = os.environ.get("ORBIT_WORKLOAD", "chat")
    strategy = os.environ.get("ORBIT_STRATEGY", "orbit")
    num_requests = int(os.environ.get("ORBIT_NUM_REQUESTS", "100"))
    arrival_rate = float(os.environ.get("ORBIT_ARRIVAL_RATE", "10.0"))
    seed = int(os.environ.get("ORBIT_SEED", "42"))

    config = WorkloadConfig(
        num_requests=num_requests,
        arrival_rate=arrival_rate,
        seed=seed,
    )

    output_dir = "/app/results" if os.path.exists("/app") else "results"

    await run_single_strategy(
        router_url=router_url,
        replica_urls=replica_urls,
        strategy=strategy,
        workload_name=workload,
        workload_config=config,
        output_dir=output_dir,
    )


if __name__ == "__main__":
    asyncio.run(main())
