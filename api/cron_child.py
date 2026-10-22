"""Profile-pinned child-process boundary for cron job bodies (#7295, #6753).

``cron.jobs`` / ``cron.scheduler`` still resolve their store and config from
process-global state (``HERMES_HOME`` plus module constants), so the WebUI pins
those globals with ``api.profiles.cron_profile_context*`` under one
process-wide ``_cron_env_lock``.  Running a scheduled job body inside that
context held the lock for the entire agent run and froze every ``/api/crons*``
reader until the job returned.

This module runs ONLY the job body (``cron.scheduler.run_job``) in a fresh
interpreter whose environment is pinned to the job's profile home.  The parent
keeps everything the Agent's lifecycle owns: the fire claim and its heartbeat,
the execution ledger row, output persistence, delivery (with any live
adapters/event loop), and terminal bookkeeping.  The parent never holds
``_cron_env_lock`` while it waits for the child.

Transport (POSIX only; other platforms fall back to the legacy in-process path):

* ``subprocess.Popen`` with ``start_new_session=True`` so the child leads its
  own process group and cancellation/timeout can kill its whole tree;
* ``env=`` pins ``HERMES_HOME`` for the child without mutating the parent's
  ``os.environ``;
* stdin carries two pickles: the parent's ``sys.path`` then the payload;
* a private result pipe carries ``("progress", execution_id)`` relays and the
  final ``("ok", result, job)`` / ``("error", ...)`` frame;
* a private control pipe carries cancellation (one byte) and, by EOF, parent
  death: the child kills its process group when the parent goes away, so a
  WebUI crash or restart ends the job body exactly as an in-process run would.
"""
from __future__ import annotations

import importlib
import logging
import os
import pickle
import selectors
import signal
import struct
import subprocess
import sys
import threading
import time
import traceback
from pathlib import Path

logger = logging.getLogger(__name__)

# Seconds a child may keep running after cancellation was forwarded before it
# is killed (the Agent's own watchdog normally returns well inside this).
CHILD_CANCEL_GRACE_SECONDS = 30.0
# Seconds between SIGTERM and SIGKILL when the parent kills a child group.
CHILD_TERMINATE_GRACE_SECONDS = 5.0
# A job-declared timeout gets this much headroom before the parent backstop
# kills the child, never less than the floor (matches the manual-run path).
CHILD_TIMEOUT_GRACE_SECONDS = 30.0
CHILD_TIMEOUT_FLOOR_SECONDS = 60.0
# After the result arrives, a child that is still finishing deferred teardown
# (for example a wedged agent worker after an inactivity timeout) is reaped in
# the background and killed once this grace expires.
CHILD_EXIT_GRACE_SECONDS = 120.0
_POLL_SECONDS = 0.2

# run_job keyword arguments that hold live parent-process objects.
#  - defer_agent_teardown: a list the caller fills with the live agent; the
#    child tears its agent down inline instead, so the caller's list stays empty.
#  - cancel_event: replaced by a child-side event driven over the control pipe.
_NON_TRANSFERABLE_KWARGS = ("defer_agent_teardown", "cancel_event")

_CANCEL_BYTE = b"c"

_in_cron_job_child = False


class ChildBoundaryUnavailable(RuntimeError):
    """The call cannot cross the process boundary; no child was started."""


def mark_cron_job_child() -> None:
    """Mark this interpreter as a cron job child so run_job never re-spawns."""
    global _in_cron_job_child
    _in_cron_job_child = True


def in_cron_job_child() -> bool:
    return _in_cron_job_child


def job_timeout_seconds(job, default=None):
    """Parent-side backstop derived from a job-declared timeout, else *default*."""
    for key in ("timeout_seconds", "max_runtime_seconds", "timeout"):
        raw = (job or {}).get(key)
        if raw in (None, ""):
            continue
        try:
            value = float(raw)
        except (TypeError, ValueError):
            continue
        if value > 0:
            return max(CHILD_TIMEOUT_FLOOR_SECONDS, value + CHILD_TIMEOUT_GRACE_SECONDS)
    return default


def _unwrap(fn):
    seen = set()
    while fn is not None and id(fn) not in seen:
        seen.add(id(fn))
        inner = getattr(fn, "_webui_original_run_job", None)
        if inner is None:
            return fn
        fn = inner
    return fn


def _resolve_target(module_name: str, qualname: str):
    obj = importlib.import_module(module_name)
    for part in qualname.split("."):
        obj = getattr(obj, part)
    return obj


def _callable_target(fn) -> tuple[str, str]:
    module_name = getattr(fn, "__module__", None)
    qualname = getattr(fn, "__qualname__", None)
    if not module_name or not qualname or "<" in qualname:
        raise ChildBoundaryUnavailable(f"{fn!r} is not importable by name")
    try:
        resolved = _resolve_target(module_name, qualname)
    except Exception as exc:
        raise ChildBoundaryUnavailable(f"{module_name}.{qualname} is not importable: {exc}") from exc
    if resolved is not fn and _unwrap(resolved) is not fn:
        raise ChildBoundaryUnavailable(f"{module_name}.{qualname} does not resolve to the wrapped run_job")
    return module_name, qualname


def _current_cron_execution():
    """The Agent's bound cron execution identity for this fire, if any."""
    module = sys.modules.get("cron.execution_identity")
    var = getattr(module, "_CURRENT", None) if module is not None else None
    if var is None:
        return None
    try:
        return var.get()
    except Exception:
        return None


def _snapshot_fire_context() -> dict:
    """Launch inputs the child needs to rebuild the Agent's per-fire scopes.

    ``run_one_job`` binds the fire's identity, secret scope and terminal policy
    as ContextVars before it calls ``run_job``.  ContextVars do not cross a
    process boundary, so the child re-installs them for its pinned home with
    the Agent's own helpers (each guarded so older Agents simply skip it).
    """
    identity = _current_cron_execution()
    snapshot = {"owned_fire": identity is not None}
    if identity is not None:
        try:
            pickle.dumps(identity)
            snapshot["identity"] = identity
        except Exception:
            logger.debug("cron child: execution identity is not transferable", exc_info=True)
    return snapshot


def _routing_process_home():
    """The parent's launch-home identity for routed-profile decisions.

    WebUI pins its own served profile (``pin_process_hermes_home``).  The child
    pins the same identity so the Agent decides "is this fire routed to another
    profile?" exactly as it would have in the parent: a fire for a non-launch
    profile keeps multiplex semantics and a files-only secret scope instead of
    falling back to the launch process environment.
    """
    module = sys.modules.get("hermes_constants")
    getter = getattr(module, "get_routing_process_hermes_home", None) if module is not None else None
    if not callable(getter):
        return None
    try:
        return str(getter())
    except Exception:
        return None


def _pin_routing_process_home(path) -> None:
    if not path:
        return
    try:
        module = importlib.import_module("hermes_constants")
    except Exception:
        return
    pin = getattr(module, "pin_process_hermes_home", None)
    if callable(pin):
        pin(path)


def _interpreter_flags() -> list[str]:
    try:
        return list(subprocess._args_from_interpreter_flags())  # same as multiprocessing spawn
    except Exception:
        return []


_BOOTSTRAP = (
    "import pickle, sys\n"
    "sys.path[:] = pickle.load(sys.stdin.buffer)\n"
    "from api.cron_child import _child_entry\n"
    "_child_entry()\n"
)


class JobChild:
    """One prepared profile-pinned child for one ``run_job`` call."""

    def __init__(self, fn, job, home, args=(), kwargs=None, *, timeout=None):
        if os.name != "posix" or not hasattr(os, "killpg"):
            raise ChildBoundaryUnavailable("profile-pinned cron child needs POSIX process groups")
        kwargs = dict(kwargs or {})
        self.parent_cancel = kwargs.pop("cancel_event", None)
        kwargs.pop("defer_agent_teardown", None)
        self.job = job
        self.home = Path(home)
        self.timeout = timeout
        payload = {
            "target": _callable_target(fn),
            "home": str(self.home),
            "job": job,
            "args": tuple(args),
            "kwargs": kwargs,
            "wants_cancel": self.parent_cancel is not None,
            "fire_context": _snapshot_fire_context(),
            "routing_process_home": _routing_process_home(),
            "log_level": logging.getLogger().getEffectiveLevel(),
        }
        try:
            self._payload_bytes = pickle.dumps(payload, protocol=pickle.HIGHEST_PROTOCOL)
            self._path_bytes = pickle.dumps(list(sys.path), protocol=pickle.HIGHEST_PROTOCOL)
        except Exception as exc:
            raise ChildBoundaryUnavailable(f"run_job arguments are not transferable: {exc}") from exc
        self.process = None
        self._selector = None
        self._result_r = None
        self._control_w = None
        self._buffer = b""
        self._cancel_forwarded_at = None

    # ── lifecycle ────────────────────────────────────────────────────────────
    def start(self) -> None:
        """Spawn the child. Raises ChildBoundaryUnavailable if nothing started."""
        result_r = result_w = control_r = control_w = None
        try:
            result_r, result_w = os.pipe()
            control_r, control_w = os.pipe()
            env = dict(os.environ)
            env["HERMES_HOME"] = str(self.home)
            env["HERMES_WEBUI_CRON_CHILD_FDS"] = f"{result_w},{control_r}"
            self.process = subprocess.Popen(
                [sys.executable, *_interpreter_flags(), "-c", _BOOTSTRAP],
                stdin=subprocess.PIPE,
                env=env,
                pass_fds=(result_w, control_r),
                start_new_session=True,
                close_fds=True,
            )
        except Exception as exc:
            for fd in (result_r, result_w, control_r, control_w):
                if fd is not None:
                    try:
                        os.close(fd)
                    except OSError:
                        pass
            raise ChildBoundaryUnavailable(f"could not start cron job child: {exc}") from exc
        os.close(result_w)
        os.close(control_r)
        self._result_r = result_r
        self._control_w = control_w
        # selectors (epoll/poll), not select.select: a busy server can hold fds > FD_SETSIZE.
        self._selector = selectors.DefaultSelector()
        self._selector.register(result_r, selectors.EVENT_READ)
        try:
            self.process.stdin.write(self._path_bytes)
            self.process.stdin.write(self._payload_bytes)
            self.process.stdin.close()
        except (BrokenPipeError, OSError):
            # The child died before reading its payload; wait() reports it.
            logger.debug("cron child exited before reading its payload", exc_info=True)
        self._payload_bytes = self._path_bytes = None

    def wait(self, on_progress=None):
        """Wait for the child; return run_job's result or raise its exception."""
        try:
            message = self._wait_for_final_message(on_progress)
        except BaseException:
            self._kill()
            self._close()
            raise
        self._reap_in_background()
        kind = message[0]
        child_job = message[2] if len(message) > 2 else None
        _sync_job(self.job, child_job)
        if kind == "ok":
            return message[1]
        raise _rebuild_exception(message[1])

    # ── internals ────────────────────────────────────────────────────────────
    def _wait_for_final_message(self, on_progress):
        deadline = None if self.timeout is None else time.monotonic() + float(self.timeout)
        eof = False
        while True:
            frame = self._next_frame()
            while frame is not None:
                if frame[0] == "progress":
                    if on_progress is not None:
                        try:
                            on_progress(frame[1])
                        except Exception:
                            logger.debug("cron child progress relay failed", exc_info=True)
                else:
                    return frame
                frame = self._next_frame()
            if eof:
                try:
                    code = self.process.wait(timeout=CHILD_TERMINATE_GRACE_SECONDS)
                except subprocess.TimeoutExpired:
                    code = "unknown (still running)"
                raise RuntimeError(
                    f"cron job child exited with code {code} before returning a result")
            self._check_cancel_and_deadline(deadline)
            if self._selector.select(timeout=_POLL_SECONDS):
                chunk = os.read(self._result_r, 1 << 16)
                if chunk:
                    self._buffer += chunk
                else:
                    eof = True

    def _next_frame(self):
        if len(self._buffer) < 4:
            return None
        (size,) = struct.unpack("!I", self._buffer[:4])
        if len(self._buffer) < 4 + size:
            return None
        body, self._buffer = self._buffer[4:4 + size], self._buffer[4 + size:]
        return pickle.loads(body)

    def _check_cancel_and_deadline(self, deadline) -> None:
        now = time.monotonic()
        cancel = self.parent_cancel
        if self._cancel_forwarded_at is None and cancel is not None:
            try:
                cancelled = bool(cancel.is_set())
            except Exception:
                cancelled = False
            if cancelled:
                self._cancel_forwarded_at = now
                try:
                    os.write(self._control_w, _CANCEL_BYTE)
                except OSError:
                    logger.debug("cron child cancel forward failed", exc_info=True)
        if (self._cancel_forwarded_at is not None
                and now - self._cancel_forwarded_at > CHILD_CANCEL_GRACE_SECONDS):
            raise RuntimeError(
                f"cron job child did not stop within {CHILD_CANCEL_GRACE_SECONDS:g}s "
                "after cancellation and was killed")
        if deadline is not None and now > deadline:
            raise RuntimeError(
                f"cron job child timed out after {float(self.timeout or 0):g}s and was killed")

    def _signal_group(self, sig) -> None:
        if self.process is None:
            return
        try:
            os.killpg(self.process.pid, sig)
        except (ProcessLookupError, PermissionError):
            pass
        except OSError:
            logger.debug("cron child killpg(%s) failed", sig, exc_info=True)

    def _kill(self) -> None:
        process = self.process
        if process is None:
            return
        if process.poll() is None:
            self._signal_group(signal.SIGTERM)
            try:
                process.wait(timeout=CHILD_TERMINATE_GRACE_SECONDS)
            except subprocess.TimeoutExpired:
                pass
        # Always sweep the group: descendants may outlive the leader.
        self._signal_group(signal.SIGKILL)
        try:
            process.wait(timeout=CHILD_TERMINATE_GRACE_SECONDS)
        except subprocess.TimeoutExpired:
            logger.warning("cron job child pid=%s did not exit after SIGKILL", process.pid)

    def _close(self) -> None:
        selector = getattr(self, "_selector", None)
        if selector is not None:
            self._selector = None
            try:
                selector.close()
            except Exception:
                pass
        for name in ("_result_r", "_control_w"):
            fd = getattr(self, name)
            if fd is not None:
                setattr(self, name, None)
                try:
                    os.close(fd)
                except OSError:
                    pass

    def _reap_in_background(self) -> None:
        def _reap():
            try:
                try:
                    self.process.wait(timeout=CHILD_EXIT_GRACE_SECONDS)
                except subprocess.TimeoutExpired:
                    logger.warning(
                        "cron job child pid=%s still running %.0fs after returning its result; killing it",
                        self.process.pid, CHILD_EXIT_GRACE_SECONDS)
                    self._kill()
                else:
                    # The leader exited; sweep any descendants it left in its group.
                    self._signal_group(signal.SIGKILL)
            finally:
                self._close()

        threading.Thread(target=_reap, name=f"cron-child-reap-{self.process.pid}", daemon=True).start()


def _sync_job(target, source) -> None:
    """Mirror run_job's in-place job-dict mutations back onto the parent's dict."""
    if not isinstance(target, dict) or not isinstance(source, dict):
        return
    for key in [k for k in target if k not in source]:
        del target[key]
    for key, value in source.items():
        try:
            same = key in target and target[key] == value
        except Exception:
            same = False
        if not same:
            target[key] = value


def _rebuild_exception(info) -> BaseException:
    exc = None
    blob = info.get("pickled") if isinstance(info, dict) else None
    if blob is not None:
        try:
            exc = pickle.loads(blob)
        except Exception:
            exc = None
    if not isinstance(exc, BaseException):
        exc = RuntimeError(f"{info.get('type', 'Exception')}: {info.get('message', '')}")
    tb_text = info.get("traceback") if isinstance(info, dict) else ""
    if tb_text:
        logger.debug("cron job child traceback:\n%s", tb_text)
        add_note = getattr(exc, "add_note", None)
        if callable(add_note):
            try:
                add_note("Raised in the profile-pinned cron job child:\n" + tb_text)
            except Exception:
                pass
    return exc


def run_job_in_child(fn, job, home, args=(), kwargs=None, *, timeout=None, on_progress=None):
    """Prepare, start, and wait for one child (convenience for callers/tests)."""
    child = JobChild(fn, job, home, args, kwargs, timeout=timeout)
    child.start()
    return child.wait(on_progress=on_progress)


# ── child side ──────────────────────────────────────────────────────────────

class _ChildChannel:
    def __init__(self, fd: int):
        self._fd = fd
        self._lock = threading.Lock()

    def send(self, frame) -> None:
        body = pickle.dumps(frame, protocol=pickle.HIGHEST_PROTOCOL)
        data = struct.pack("!I", len(body)) + body
        with self._lock:
            view = memoryview(data)
            while view:
                written = os.write(self._fd, view)
                view = view[written:]


def _kill_own_group_and_exit(code: int = 1) -> None:  # pragma: no cover - exits
    try:
        os.killpg(0, signal.SIGKILL)
    finally:
        os._exit(code)


def _watch_control_pipe(fd: int, cancel_event: threading.Event) -> None:
    """Cancel byte -> set the child cancel event; EOF -> the parent is gone."""
    while True:
        try:
            data = os.read(fd, 1)
        except InterruptedError:
            continue
        except OSError:
            data = b""
        if not data:
            _kill_own_group_and_exit(1)
            return
        cancel_event.set()


def _install_progress_relay(channel: _ChildChannel) -> None:
    """Relay the Agent's ledger progress stamps to the owning parent process.

    ``cron.executions.touch_execution_progress`` is fenced to the process that
    owns the execution row (``process_id``/``pid``).  The parent created and
    owns that row, so the child forwards each stamp and the parent writes it.
    """
    try:
        executions = importlib.import_module("cron.executions")
    except Exception:
        return
    original = getattr(executions, "touch_execution_progress", None)
    if not callable(original):
        return

    def _relay_touch_execution_progress(execution_id, *args, **kwargs):
        try:
            channel.send(("progress", str(execution_id)))
            return True
        except Exception:
            return False

    executions.touch_execution_progress = _relay_touch_execution_progress
    for name, module in list(sys.modules.items()):
        if (name == "cron" or name.startswith("cron.")) and getattr(
                module, "touch_execution_progress", None) is original:
            setattr(module, "touch_execution_progress", _relay_touch_execution_progress)


def _install_fire_context(fire_context: dict, home: Path) -> list:
    """Re-install the Agent's per-fire ContextVars for the pinned home."""
    resets = []
    if not fire_context.get("owned_fire"):
        return resets
    identity = fire_context.get("identity")
    if identity is not None:
        try:
            execution_identity = importlib.import_module("cron.execution_identity")
            var = getattr(execution_identity, "_CURRENT")
            token = var.set(identity)
            resets.append(lambda: var.reset(token))
        except Exception:
            logger.debug("cron child: could not bind execution identity", exc_info=True)
    scheduler = importlib.import_module("cron.scheduler")
    install_scope = getattr(scheduler, "_install_fire_secret_scope", None)
    reset_scope = getattr(scheduler, "_reset_fire_secret_scope", None)
    if callable(install_scope) and callable(reset_scope):
        tokens = install_scope()
        resets.append(lambda: reset_scope(tokens))
    try:
        terminal_scope = importlib.import_module("tools.terminal_scope")
    except Exception:
        terminal_scope = None
    install_terminal = getattr(terminal_scope, "install_profile_terminal_scope", None)
    reset_terminal = getattr(terminal_scope, "reset_terminal_scope", None)
    if callable(install_terminal) and callable(reset_terminal):
        terminal_token = install_terminal(home)
        resets.append(lambda: reset_terminal(terminal_token))
    return resets


def _exception_info(exc: BaseException) -> dict:
    info = {
        "type": type(exc).__name__,
        "message": str(exc),
        "traceback": "".join(traceback.format_exception(type(exc), exc, exc.__traceback__)),
        "pickled": None,
    }
    try:
        blob = pickle.dumps(exc, protocol=pickle.HIGHEST_PROTOCOL)
        pickle.loads(blob)
        info["pickled"] = blob
    except Exception:
        pass
    return info


def _raise_system_exit(signum, frame):  # pragma: no cover - signal path
    raise SystemExit(128 + signum)


def _child_entry() -> None:  # pragma: no cover - runs in the child interpreter
    mark_cron_job_child()
    result_fd, control_fd = (int(x) for x in os.environ.pop("HERMES_WEBUI_CRON_CHILD_FDS").split(","))
    os.set_inheritable(result_fd, False)
    os.set_inheritable(control_fd, False)
    channel = _ChildChannel(result_fd)
    cancel_event = threading.Event()
    threading.Thread(
        target=_watch_control_pipe, args=(control_fd, cancel_event),
        name="cron-child-control", daemon=True).start()
    signal.signal(signal.SIGTERM, _raise_system_exit)

    job = None
    try:
        payload = pickle.load(sys.stdin.buffer)
        try:
            sys.stdin.close()
            devnull = os.open(os.devnull, os.O_RDONLY)
            os.dup2(devnull, 0)
            os.close(devnull)
        except OSError:
            pass
        if not logging.getLogger().handlers:
            logging.basicConfig(
                level=payload.get("log_level", logging.WARNING),
                format="%(asctime)s %(levelname)s [cron-child %(process)d] %(name)s: %(message)s")
        job = payload["job"]
        home = Path(payload["home"])
        _pin_routing_process_home(payload.get("routing_process_home"))
        from api.profiles import cron_profile_context_for_home

        with cron_profile_context_for_home(home):
            _install_progress_relay(channel)
            resets = _install_fire_context(payload.get("fire_context") or {}, home)
            try:
                fn = _unwrap(_resolve_target(*payload["target"]))
                kwargs = dict(payload.get("kwargs") or {})
                if payload.get("wants_cancel"):
                    kwargs["cancel_event"] = cancel_event
                result = fn(job, *payload.get("args", ()), **kwargs)
            finally:
                for reset in reversed(resets):
                    try:
                        reset()
                    except Exception:
                        logger.debug("cron child: scope reset failed", exc_info=True)
        try:
            channel.send(("ok", result, job))
        except Exception as exc:
            channel.send(("error", _exception_info(
                RuntimeError(f"cron job result is not transferable: {exc}")), job))
    except BaseException as exc:
        try:
            try:
                channel.send(("error", _exception_info(exc), job))
            except Exception:
                channel.send(("error", _exception_info(exc), None))
        except Exception:
            pass
        if not isinstance(exc, Exception):
            raise
    finally:
        # Flush what the job printed; normal interpreter shutdown then lets a
        # deferred agent teardown finish (the parent reaps or kills us).
        for stream in (sys.stdout, sys.stderr):
            try:
                stream.flush()
            except Exception:
                pass
