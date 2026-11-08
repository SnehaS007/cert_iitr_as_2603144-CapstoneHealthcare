"""Task 11 — FastAPI deployment.

  POST /ask           AskRequest -> SupportResponse
  POST /add-document  AddDocumentRequest -> AddDocumentResponse
  GET  /health
  WS   /ws/chat       real-time multi-turn chat; a client disconnect is caught and logged, and the
                      server keeps serving other clients.

Run (single worker: memory, cache and index are in-process):
  uvicorn app.api.main:app --port 8000
"""

from __future__ import annotations

import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.concurrency import run_in_threadpool
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import ValidationError

import app.config as config
from app.governance.budget import BudgetExceededError
from app.guardrails.input_guard import InputRejectedError
from app.logging_utils import new_trace_id
from app.pipeline import DocumentConflictError, SupportPipeline
from app.schemas import AddDocumentRequest, AddDocumentResponse, AskRequest, ErrorResponse, SupportResponse


def _error(status: int, trace_id: str, error: str, detail: str) -> JSONResponse:
    return JSONResponse(status_code=status, content=ErrorResponse(trace_id=trace_id, error=error, detail=detail).model_dump())


def _safe_validation_detail(errors: list[dict]) -> str:
    """Field locations and error types only. FastAPI's default 422 body echoes the raw input, which
    could contain an unmasked phone number; this never does."""
    return "; ".join(f"{'.'.join(str(p) for p in e.get('loc', []))}: {e.get('type')}" for e in errors)


def create_app(pipeline: SupportPipeline | None = None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if pipeline is None:
            from app.rag.embeddings import get_default_embedder

            app.state.pipeline = SupportPipeline(get_default_embedder())
        else:
            app.state.pipeline = pipeline
        yield

    app = FastAPI(title="Practo Domain Support Agent", version="1.0.0", lifespan=lifespan)

    @app.exception_handler(RequestValidationError)
    async def validation_handler(request: Request, exc: RequestValidationError):
        trace_id = new_trace_id()
        request.app.state.pipeline.logger.log_request(
            trace_id=trace_id, endpoint=request.url.path, outcome="rejected_validation",
            validation_errors=_safe_validation_detail(exc.errors()),
        )
        return _error(422, trace_id, "validation_error", _safe_validation_detail(exc.errors()))

    @app.get("/health")
    async def health(request: Request) -> dict:
        p: SupportPipeline = request.app.state.pipeline
        return {"status": "ok", "mock_llm": config.MOCK_LLM, "collection": p.collection, "threshold": p.threshold}

    @app.post("/ask", response_model=SupportResponse, responses={400: {"model": ErrorResponse}, 413: {"model": ErrorResponse}})
    async def ask(body: AskRequest, request: Request):
        trace_id = new_trace_id()
        p: SupportPipeline = request.app.state.pipeline
        try:
            return await run_in_threadpool(p.ask, body.query, body.session_id, "/ask", trace_id)
        except InputRejectedError as exc:
            return _error(400, trace_id, "input_rejected", f"Request blocked by guardrail: {', '.join(exc.result.injection_rules)}")
        except BudgetExceededError as exc:
            return _error(413, trace_id, "budget_exceeded", str(exc))

    @app.post("/add-document", response_model=AddDocumentResponse,
              responses={400: {"model": ErrorResponse}, 409: {"model": ErrorResponse}, 422: {"model": ErrorResponse}})
    async def add_document(body: AddDocumentRequest, request: Request):
        trace_id = new_trace_id()
        p: SupportPipeline = request.app.state.pipeline
        try:
            return await run_in_threadpool(p.add_document, body.doc_id, body.title, body.text, trace_id)
        except InputRejectedError as exc:
            return _error(400, trace_id, "input_rejected", f"Document blocked by guardrail: {', '.join(exc.result.injection_rules)}")
        except DocumentConflictError as exc:
            return _error(409, trace_id, "conflict", str(exc))
        except ValueError as exc:
            return _error(422, trace_id, "validation_error", str(exc))

    @app.websocket("/ws/chat")
    async def ws_chat(websocket: WebSocket):
        await websocket.accept()
        p: SupportPipeline = websocket.app.state.pipeline
        session_id = websocket.query_params.get("session_id") or f"ws-{uuid.uuid4().hex[:12]}"
        await websocket.send_json({"type": "session", "session_id": session_id})
        try:
            while True:
                message = await websocket.receive_json()
                trace_id = new_trace_id()
                try:
                    body = AskRequest(session_id=session_id, query=message.get("query", ""))
                    response = await run_in_threadpool(p.ask, body.query, session_id, "/ws/chat", trace_id)
                    await websocket.send_json({"type": "answer", **response.model_dump()})
                except ValidationError as exc:
                    await websocket.send_json({"type": "error", "trace_id": trace_id, "error": "validation_error",
                                               "detail": _safe_validation_detail(exc.errors())})
                except InputRejectedError as exc:
                    await websocket.send_json({"type": "error", "trace_id": trace_id, "error": "input_rejected",
                                               "detail": ", ".join(exc.result.injection_rules)})
                except BudgetExceededError as exc:
                    await websocket.send_json({"type": "error", "trace_id": trace_id, "error": "budget_exceeded",
                                               "detail": str(exc)})
        except WebSocketDisconnect as exc:
            # Client left mid-conversation: log it and end this handler only. The server keeps running.
            p.logger.log_request(trace_id=new_trace_id(), endpoint="/ws/chat", session_id=session_id,
                                 outcome="client_disconnected", close_code=exc.code)

    return app


app = create_app()
