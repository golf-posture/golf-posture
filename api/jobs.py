"""Single-server, single-inference job lifecycle. Results expire after 30 minutes."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
import logging
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Event, RLock
import time
from uuid import uuid4

from pipeline.event_analysis import AnalysisCancelled, InvalidVideo, analyze_events, apply_event_frames

MAX_BYTES = 200_000_000
TTL_SECONDS = 30 * 60
TERMINAL = {"completed", "cancelled", "failed"}
logger = logging.getLogger(__name__)


@dataclass
class Job:
    id: str
    status: str = "created"
    stage: str = "created"
    processed_frames: int = 0
    total_frames: int | None = None
    result: dict | None = None
    error: str | None = None
    created_at: float = field(default_factory=time.monotonic)
    finished_at: float | None = None
    cancel: Event = field(default_factory=Event)
    future: object = None


class JobStore:
    def __init__(self, checkpoint: Path, analyzer=analyze_events):
        self.checkpoint, self.analyzer = checkpoint, analyzer
        self.jobs: dict[str, Job] = {}
        self.lock = RLock()
        self.pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="events")
        self.temp = TemporaryDirectory(prefix="kugolf-events-")

    def path(self, job: Job):
        return Path(self.temp.name) / f"{job.id}.video"

    def create(self, job_id=None):
        # An optional client-generated UUID makes creation retries idempotent.
        with self.lock:
            self.expire()
            job_id = job_id or str(uuid4())
            if job_id not in self.jobs:
                self.jobs[job_id] = Job(job_id)
            return self.jobs[job_id]

    def get(self, job_id):
        with self.lock:
            self.expire()
            return self.jobs[job_id]

    def snapshot(self, job):
        with self.lock:
            return {"id": job.id, "status": job.status, "stage": job.stage,
                    "processed_frames": job.processed_frames, "total_frames": job.total_frames,
                    "result": job.result, "error": job.error}

    def claim_upload(self, job):
        with self.lock:
            if job.status != "created":
                return False
            job.status = job.stage = "uploading"
            return True

    def check(self, job):
        if job.cancel.is_set():
            raise AnalysisCancelled()

    def update(self, job, **values):
        with self.lock:
            self.check(job)
            for key, value in values.items():
                setattr(job, key, value)

    def finish(self, job, status, result=None, error=None):
        with self.lock:
            if job.status in TERMINAL:
                return
            # Caller has closed every file/decoder before reaching this point.
            self.path(job).unlink(missing_ok=True)
            job.status = job.stage = "cancelled" if job.cancel.is_set() else status
            job.result = result if job.status == "completed" else None
            job.error = error if job.status == "failed" else None
            job.finished_at = time.monotonic()

    def enqueue(self, job, event_frames=None):
        with self.lock:
            if job.cancel.is_set():
                self.finish(job, "cancelled")
                return
            job.status = job.stage = "queued"
            job.future = self.pool.submit(self._run, job, event_frames)

    def _run(self, job, event_frames):
        try:
            self.update(job, status="running", stage="preparing")
            def update(**values):
                self.update(job, **values)

            def check():
                self.check(job)

            if event_frames is None:
                result = self.analyzer(self.path(job), self.checkpoint, update, check)
            else:
                result = apply_event_frames(self.path(job), event_frames, update, check)
            self.check(job)
            if job.processed_frames != result["total_frames"]:
                raise InvalidVideo("디코딩한 프레임 수와 영상 시각 정보가 일치하지 않습니다.")
            self.finish(job, "completed", result=result)
        except AnalysisCancelled:
            self.finish(job, "cancelled")
        except Exception as exc:
            logger.exception("Event job %s failed", job.id)
            message = str(exc) if isinstance(exc, InvalidVideo) else "이벤트 검출 중 오류가 발생했습니다. 다시 시도해주세요."
            self.finish(job, "failed", error=message)

    def cancel(self, job):
        with self.lock:
            if job.status in TERMINAL:
                return
            previous = job.status
            job.cancel.set()
            job.status = job.stage = "cancelling"
            if previous == "created" or (previous == "queued" and job.future.cancel()):
                self.finish(job, "cancelled")

    def expire(self):
        with self.lock:
            now = time.monotonic()
            for key, job in list(self.jobs.items()):
                since = job.finished_at if job.finished_at is not None else job.created_at
                if (job.status in TERMINAL or job.status == "created") and now - since >= TTL_SECONDS:
                    self.path(job).unlink(missing_ok=True)
                    del self.jobs[key]

    def close(self):
        for job in list(self.jobs.values()):
            self.cancel(job)
        self.pool.shutdown(wait=True, cancel_futures=True)
        self.temp.cleanup()
