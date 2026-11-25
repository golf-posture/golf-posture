"""Run with: pixi run uvicorn api.main:app --host 0.0.0.0 --port 8000 --workers 1."""
from __future__ import annotations

import asyncio
import json
from contextlib import asynccontextmanager, suppress
from pathlib import Path
from uuid import UUID

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from starlette.datastructures import UploadFile
from starlette.formparsers import MultiPartException, MultiPartParser
from starlette.requests import ClientDisconnect

from api.jobs import JobStore, MAX_BYTES
from pipeline.event_analysis import AnalysisCancelled, MAX_SECONDS

ROOT = Path(__file__).resolve().parents[1]


def create_app(store=None):
    uploads = {}

    @asynccontextmanager
    async def lifespan(app):
        app.state.jobs = store or JobStore(ROOT / "event_detection/models/swingnet_1800.pth.tar")

        async def sweep():
            while True:
                await asyncio.sleep(30)
                app.state.jobs.expire()

        sweeper = asyncio.create_task(sweep())
        try:
            yield
        finally:
            sweeper.cancel()
            with suppress(asyncio.CancelledError):
                await sweeper
            active = list(uploads.values())
            for task in active:
                task.cancel()
            if active:
                await asyncio.gather(*active, return_exceptions=True)
            await asyncio.to_thread(app.state.jobs.close)

    app = FastAPI(title="KUGOLF events", lifespan=lifespan)
    app.add_middleware(CORSMiddleware, allow_origin_regex=r"https?://(localhost|127\.0\.0\.1|10\.[\d.]+|192\.168\.[\d.]+|172\.(1[6-9]|2\d|3[01])\.[\d.]+)(:\d+)?",
                       allow_methods=["GET", "POST", "PUT"], allow_headers=["Content-Type"])

    def jobs():
        return app.state.jobs

    def lookup(job_id):
        try:
            return jobs().get(str(job_id))
        except KeyError:
            raise HTTPException(404, "작업이 만료되었거나 서버가 재시작되었습니다. 다시 분석해주세요.")

    @app.get("/health")
    def health():
        return {"status": "ok", "max_bytes": MAX_BYTES, "max_seconds": MAX_SECONDS}

    @app.post("/event-jobs", status_code=201)
    def create(job_id: UUID | None = None):
        return jobs().snapshot(jobs().create(str(job_id) if job_id else None))

    @app.get("/event-jobs/{job_id}")
    def status(job_id: UUID):
        return jobs().snapshot(lookup(job_id))

    @app.post("/event-jobs/{job_id}/cancel")
    async def cancel(job_id: UUID):
        job = lookup(job_id)
        first_cancel = not job.cancel.is_set()
        jobs().cancel(job)
        task = uploads.get(job.id)
        if task and first_cancel and job.cancel.is_set():
            task.cancel()
        return jobs().snapshot(job)

    @app.put("/event-jobs/{job_id}/video")
    async def upload(job_id: UUID, request: Request):
        job = lookup(job_id)
        if not jobs().claim_upload(job):
            raise HTTPException(409, "이미 업로드했거나 종료된 작업입니다.")
        uploads[job.id] = asyncio.current_task()
        form = None
        failure = None
        event_frames = None
        try:
            if not request.headers.get("content-type", "").startswith("multipart/form-data"):
                raise ValueError("영상 파일을 multipart 형식으로 전송해주세요.")

            async def limited_stream():
                received = 0
                async for chunk in request.stream():
                    jobs().check(job)
                    received += len(chunk)
                    if received > MAX_BYTES + 65536:
                        raise ValueError("200MB 이하의 영상을 선택해주세요.")
                    yield chunk

            form = await MultiPartParser(request.headers, limited_stream(), max_files=1, max_fields=1).parse()
            file = form.get("file")
            if not isinstance(file, UploadFile) or set(form) - {"file", "event_frames"}:
                raise ValueError("영상 파일 한 개를 선택해주세요.")
            if "event_frames" in form:
                try:
                    event_frames = json.loads(form["event_frames"])
                except (ValueError, TypeError):
                    raise ValueError("수정할 이벤트 프레임 형식이 올바르지 않습니다.") from None
                if (not isinstance(event_frames, list) or len(event_frames) != 8 or
                        any(f is not None and (type(f) is not int or f < 0) for f in event_frames)):
                    raise ValueError("8개 이벤트의 프레임 번호를 확인해주세요.")
            if Path(file.filename or "").suffix.lower() not in {".mp4", ".mov"}:
                raise ValueError("MP4 또는 MOV 영상을 선택해주세요.")
            if not file.size or file.size > MAX_BYTES:
                raise ValueError("크기가 0보다 크고 200MB 이하인 영상을 선택해주세요.")
            with jobs().path(job).open("wb") as target:
                while chunk := await file.read(1024 * 1024):
                    jobs().check(job)
                    target.write(chunk)
            jobs().check(job)
        except (asyncio.CancelledError, AnalysisCancelled):
            job.cancel.set()
        except ClientDisconnect:
            failure = "영상 전송이 끊겼습니다. 다시 분석해주세요."
        except (ValueError, MultiPartException) as exc:
            failure = str(exc) or getattr(exc, "message", "영상을 업로드할 수 없습니다.")
        except Exception:
            failure = "영상을 업로드할 수 없습니다. 다시 시도해주세요."
        finally:
            if form is not None:
                await form.close()
            uploads.pop(job.id, None)
        if job.cancel.is_set():
            jobs().finish(job, "cancelled")
        elif failure:
            jobs().finish(job, "failed", error=failure)
        else:
            jobs().enqueue(job, event_frames)
        return jobs().snapshot(job)

    return app


app = create_app()
