"""
FastAPI server for Aegis-Router.

Provides HTTP endpoints for:
- Request routing
- Worker management
- Health checks
- Metrics
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import TYPE_CHECKING, Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from aegis_router.router.cache_router import CacheRouter, RoutingDecision

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator


# Global router instance
_router: CacheRouter | None = None


class RouteRequest(BaseModel):
    """Request to route a prompt."""

    token_ids: list[int] = Field(..., description="Token sequence of the prompt")
    priority: int = Field(0, description="Request priority")
    preferred_workers: list[str] | None = Field(None, description="Preferred worker IDs")


class RouteResponse(BaseModel):
    """Response from routing request."""

    worker_id: str
    worker_url: str
    strategy_used: str
    cache_hit_ratio: float
    matched_tokens: int
    total_tokens: int
    estimated_tokens_to_compute: int
    confidence: str
    metadata: dict[str, Any]


class WorkerRegistration(BaseModel):
    """Worker registration request."""

    worker_id: str
    host: str
    port: int


class WorkerHeartbeat(BaseModel):
    """Worker heartbeat request."""

    load: float | None = Field(None, ge=0.0, le=1.0)
    queue_depth: int | None = Field(None, ge=0)


class CacheUpdate(BaseModel):
    """Cache update request."""

    token_sequences: list[list[int]]


def get_router() -> CacheRouter:
    """Get the global router instance."""
    if _router is None:
        raise RuntimeError("Router not initialized")
    return _router


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Manage router lifecycle."""
    global _router
    _router = CacheRouter()
    _router.start()
    yield
    _router.stop()


def create_app() -> FastAPI:
    """Create the FastAPI application."""
    app = FastAPI(
        title="Aegis-Router",
        description="Cache-aware LLM request router with Radix Trie matching",
        version="0.1.0",
        lifespan=lifespan,
    )

    @app.post("/route", response_model=RouteResponse)
    async def route_request(req: RouteRequest) -> RouteResponse:
        """Route a request to the best worker."""
        router = get_router()

        try:
            decision = router.route_request(
                token_ids=tuple(req.token_ids),
                priority=req.priority,
                preferred_workers=set(req.preferred_workers) if req.preferred_workers else None,
            )

            return RouteResponse(
                worker_id=decision.worker_id,
                worker_url=decision.worker_url,
                strategy_used=decision.strategy_used,
                cache_hit_ratio=decision.cache_hit_ratio,
                matched_tokens=decision.matched_tokens,
                total_tokens=decision.total_tokens,
                estimated_tokens_to_compute=decision.estimated_tokens_to_compute,
                confidence=decision.confidence,
                metadata=decision.metadata,
            )
        except Exception as e:
            raise HTTPException(status_code=503, detail=str(e))

    @app.post("/workers/register")
    async def register_worker(req: WorkerRegistration) -> dict:
        """Register a new worker."""
        router = get_router()
        worker = router.register_worker(
            worker_id=req.worker_id,
            host=req.host,
            port=req.port,
        )
        return {"status": "registered", "worker": worker.to_dict()}

    @app.post("/workers/{worker_id}/heartbeat")
    async def worker_heartbeat(worker_id: str, req: WorkerHeartbeat) -> dict:
        """Update worker heartbeat."""
        router = get_router()
        router.update_worker_heartbeat(
            worker_id=worker_id,
            load=req.load,
            queue_depth=req.queue_depth,
        )
        return {"status": "ok"}

    @app.post("/workers/{worker_id}/cache")
    async def update_cache(worker_id: str, req: CacheUpdate) -> dict:
        """Update worker cache."""
        router = get_router()
        sequences = [tuple(seq) for seq in req.token_sequences]
        router.update_worker_cache(worker_id, sequences)
        return {"status": "updated", "sequences": len(sequences)}

    @app.delete("/workers/{worker_id}")
    async def unregister_worker(worker_id: str) -> dict:
        """Unregister a worker."""
        router = get_router()
        success = router.unregister_worker(worker_id)
        if not success:
            raise HTTPException(status_code=404, detail="Worker not found")
        return {"status": "unregistered"}

    @app.get("/workers")
    async def list_workers() -> dict:
        """List all workers."""
        router = get_router()
        workers = [w.to_dict() for w in router.get_all_workers()]
        return {"workers": workers}

    @app.get("/stats")
    async def get_stats() -> dict:
        """Get router statistics."""
        router = get_router()
        return router.get_stats()

    @app.get("/health")
    async def health_check() -> dict:
        """Health check endpoint."""
        router = get_router()
        healthy = len(router.get_healthy_workers())
        total = len(router.get_all_workers())
        return {
            "status": "healthy" if healthy > 0 else "degraded",
            "workers_healthy": healthy,
            "workers_total": total,
        }

    return app
