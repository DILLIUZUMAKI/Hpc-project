"""Bounded subprocess capture. Never invokes a shell."""
import os
import selectors
import signal
import subprocess
import time


def run_process(args, *, timeout=30, cwd=None, env=None, input_bytes=None, limit=2_000_000):
    start = time.perf_counter()
    proc = subprocess.Popen(args, cwd=cwd, env=env, stdin=subprocess.PIPE if input_bytes is not None else subprocess.DEVNULL,
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE, start_new_session=True)
    selector = selectors.DefaultSelector()
    outputs = {"stdout": bytearray(), "stderr": bytearray()}
    for name in outputs:
        selector.register(getattr(proc, name), selectors.EVENT_READ, name)
    payload = memoryview(input_bytes or b"")
    if input_bytes is not None:
        os.set_blocking(proc.stdin.fileno(), False)
        selector.register(proc.stdin, selectors.EVENT_WRITE, "stdin")
    failure = None
    try:
        while selector.get_map():
            if time.perf_counter() - start > timeout:
                failure = "timeout"
                break
            for key, _ in selector.select(timeout=0.05):
                if key.data == "stdin":
                    try:
                        count = os.write(key.fileobj.fileno(), payload[:65536]) if payload else 0
                        payload = payload[count:]
                    except BrokenPipeError:
                        payload = memoryview(b"")
                    if not payload:
                        selector.unregister(key.fileobj)
                        key.fileobj.close()
                    continue
                chunk = os.read(key.fileobj.fileno(), 65536)
                if not chunk:
                    selector.unregister(key.fileobj)
                    key.fileobj.close()
                    continue
                remaining = max(0, limit - sum(map(len, outputs.values())))
                outputs[key.data].extend(chunk[:remaining])
                if len(chunk) > remaining:
                    failure = "output_limit"
                    break
            if failure:
                break
        if not failure:
            try:
                proc.wait(timeout=max(0.01, timeout - (time.perf_counter() - start)))
            except subprocess.TimeoutExpired:
                failure = "timeout"
    finally:
        # Kill any descendants, including a background child that inherited a pipe.
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        proc.wait()
        for key in list(selector.get_map().values()):
            key.fileobj.close()
        selector.close()
    return {"command": list(args), "returncode": proc.returncode, "failure": failure,
            "seconds": time.perf_counter() - start,
            **{k: bytes(v).decode("utf-8", errors="replace") for k, v in outputs.items()}}
