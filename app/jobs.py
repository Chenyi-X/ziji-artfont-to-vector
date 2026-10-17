"""Bounded conversion jobs with cancellation and terminal-only retention."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import threading
import time
import uuid


class JobCancelled(Exception):
    pass


class JobQueueFull(Exception):
    pass


class JobManager:
    def __init__(self, workers=2, capacity=10, retained=64, ttl_seconds=3600):
        self._executor = ThreadPoolExecutor(max_workers=workers, thread_name_prefix='vector-job')
        self._slots = threading.BoundedSemaphore(capacity)
        self._lock = threading.RLock()
        self._jobs = {}
        self.retained = retained
        self.ttl_seconds = ttl_seconds

    def _prune(self):
        now = time.monotonic()
        terminal = [j for j in self._jobs.values() if j['status'] in {'done','error','cancelled'}]
        terminal.sort(key=lambda j:j['finished'])
        for index, job in enumerate(terminal):
            if len(terminal)-index > self.retained or now-job['finished'] > self.ttl_seconds:
                self._jobs.pop(job['id'], None)

    def submit(self, operation):
        if not self._slots.acquire(blocking=False):
            raise JobQueueFull('转换队列已满，请稍后重试')
        job = {'id':uuid.uuid4().hex,'status':'queued','stage':'queued',
               'created':time.monotonic(),'result':None,'error':None,'cancel':False}
        with self._lock:
            self._prune()
            self._jobs[job['id']] = job

        def progress(stage):
            with self._lock:
                if job['cancel']:
                    raise JobCancelled()
                job['stage'] = stage

        def run():
            terminal_status = 'error'
            try:
                progress('preprocess')
                with self._lock:
                    job['status'] = 'running'
                    job['started'] = time.monotonic()
                payload = operation(progress)
                with self._lock:
                    if job['cancel']:
                        raise JobCancelled()
                    job['result'] = payload
                    terminal_status = 'error' if payload.get('error') else 'done'
                    job['error'] = payload.get('error')
            except JobCancelled:
                terminal_status = 'cancelled'
            except Exception as exc:
                with self._lock:
                    job['error'] = f'转换失败：{exc}'
            finally:
                with self._lock:
                    if job['cancel'] and terminal_status == 'done':
                        terminal_status = 'cancelled'
                    job['finished'] = time.monotonic()
                    self._slots.release()
                    job['status'] = terminal_status
                    self._prune()
        try:
            self._executor.submit(run)
        except Exception:
            with self._lock:
                self._jobs.pop(job['id'],None)
            self._slots.release()
            raise
        return job['id']

    def get(self, job_id):
        with self._lock:
            self._prune()
            job = self._jobs.get(job_id)
            if job is None:
                raise KeyError(job_id)
            end = job.get('finished', time.monotonic())
            out = {'id':job_id,'status':job['status'],'stage':job['stage'],
                   'elapsed_ms':round((end-job['created'])*1000,1),'error':job['error'],
                   'queue_ms':round((job.get('started',end)-job['created'])*1000,1)}
            if job['status']=='done':
                out['result'] = deepcopy(job['result'])
            return out

    def cancel(self, job_id):
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                raise KeyError(job_id)
            if job['status'] in {'queued','running'}:
                job['cancel'] = True
            return self.get(job_id)

    def shutdown(self):
        with self._lock:
            for job in self._jobs.values():
                if job['status'] in {'queued','running'}:
                    job['cancel'] = True
        self._executor.shutdown(wait=True)
