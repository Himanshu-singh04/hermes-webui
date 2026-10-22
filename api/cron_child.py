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
  death: the child kills its process tree when the parent goes away, so a
  WebUI crash or restart ends the job body exactly as an in-process run would.

Every forced stop (cancel grace, timeout backstop, parent death, a child still
running long after its result) kills the whole process TREE, not just the
child's group: the Agent starts cron scripts with ``start_new_session=True``, so
they live outside the child's process group.  Descendants are found through
``/proc`` (or psutil where installed), frozen, then killed while the child is
still alive to anchor them.

What crosses the boundary (and what deliberately does not):

* The child environment is the parent's ``os.environ`` with ``HERMES_HOME``
  pinned and the per-turn markers WebUI mirrors process-wide during a chat turn
  (human-presence flags such as ``HERMES_EXEC_ASK`` and ``HERMES_SESSION_*``
  identity) removed -- the same unattended-worker hygiene the Agent applies to
  its own restart-safe cron worker.  A child spawned while a chat turn streams
  therefore never inherits that turn's identity or approval mode.
* The Agent's per-fire ContextVars (execution identity, secret scope, terminal
  policy, multiplex semantics) are re-installed in the child for its pinned home
  with the Agent's own helpers -- only the ones active in the parent.  The
  child rebuilds each scope from the profile's own sources.  If the parent has
  an active scope the installed Agent cannot rebuild in the child, the call
  stays in-process (``ChildBoundaryUnavailable``) rather than run a body whose
  unscoped secret reads would fall back to the launch process environment.
* The Agent's frozen launch environment (``tui_gateway.launch_profile_policy``,
  the only env it trusts as the launch profile's once multi-profile hosting is
  active) is process-local, so the parent's snapshot travels over the private
  stdin pipe and is installed in the child before anything can capture its own
  (live, possibly WebUI-mirrored) environment.  An unrecognised snapshot layout
  keeps the call in-process.
* A gateway cron owner execution bound in this process (``gateway.session_cron``)
  is not transferable, so such a call stays in-process too; otherwise the child
  would resubmit the fire to the gateway instead of running it locally.  WebUI
  never binds one itself.

Timeouts: Agent jobs normally declare no wall-clock timeout, and the parent adds
none for them (matching the in-process path).  The effective limits remain the
Agent's own -- the agent turn's inactivity watchdog (``HERMES_CRON_TIMEOUT``,
idle-based, default 600s) wherever the turn runs, the script timeout for
``no_agent`` jobs, and the parent-side fire-claim heartbeat / stale-claim sweep
that ``run_one_job`` keeps owning.  Parent cancellation (claim lost, transport
cancel) is forwarded and enforced with a kill after
``CHILD_CANCEL_GRACE_SECONDS``.  ``job_timeout_seconds`` is only a backstop for
jobs that DO declare a timeout.
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


# Agent APIs the child uses to rebuild each per-fire scope, keyed by scope.
# Resolved by name so WebUI keeps working against older Agents, and pinned by a
# contract test against the installed Agent so a rename cannot silently drop a
# scope (see ``missing_fire_context_apis``).
FIRE_CONTEXT_APIS = {
    "identity": (("cron.execution_identity", "_CURRENT"),),
    "secret_scope": (
        ("cron.scheduler", "_install_fire_secret_scope"),
        ("cron.scheduler", "_reset_fire_secret_scope"),
    ),
    "terminal_scope": (
        ("tools.terminal_scope", "install_profile_terminal_scope"),
        ("tools.terminal_scope", "reset_terminal_scope"),
    ),
    "multiplex_context": (
        ("agent.secret_scope", "set_multiplex_context"),
        ("agent.secret_scope", "reset_multiplex_context"),
        ("agent.secret_scope", "is_multiplex_active"),
    ),
    "multiplex_process": (("agent.secret_scope", "set_multiplex_active"),),
    "routing_home": (("hermes_constants", "pin_process_hermes_home"),),
    "progress": (("cron.executions", "touch_execution_progress"),),
}


def missing_fire_context_apis(scopes=None, *, import_missing: bool = True) -> list[str]:
    """``module.name`` entries of FIRE_CONTEXT_APIS absent from the installed Agent."""
    missing = []
    for scope in (FIRE_CONTEXT_APIS if scopes is None else scopes):
        for module_name, attr in FIRE_CONTEXT_APIS[scope]:
            module = sys.modules.get(module_name)
            if module is None and import_missing:
                try:
                    module = importlib.import_module(module_name)
                except Exception:
                    module = None
            if module is None or getattr(module, attr, None) is None:
                missing.append(f"{module_name}.{attr}")
    return missing


def _loaded_attr(module_name: str, attr: str):
    module = sys.modules.get(module_name)
    return getattr(module, attr, None) if module is not None else None


def _call_quietly(fn, default=None):
    if not callable(fn):
        return default
    try:
        return fn()
    except Exception:
        return default


def _snapshot_fire_context() -> dict:
    """Which of the Agent's per-fire scopes are active here, for the child to rebuild.

    ``run_one_job`` binds the fire's identity, secret scope, terminal policy and
    (for a routed fire) multiplex semantics as ContextVars before it calls
    ``run_job``.  ContextVars do not cross a process boundary, so the child
    re-installs each scope that is active here, for its pinned home, with the
    Agent's own helpers.  Only flags and the frozen identity record cross: the
    child rebuilds secret and terminal scopes from the profile's own sources.

    Raises ChildBoundaryUnavailable when an active scope could not be rebuilt
    by the installed Agent, so the caller keeps the in-process path instead of
    running the body without it.
    """
    identity = _current_cron_execution()
    secret_scope = _call_quietly(_loaded_attr("agent.secret_scope", "current_secret_scope"))
    terminal_scope = _call_quietly(_loaded_attr("tools.terminal_scope", "get_terminal_scope"))
    multiplex_var = _loaded_attr("agent.secret_scope", "_MULTIPLEX_CONTEXT")
    snapshot = {
        "owned_fire": identity is not None,
        "secret_scope": secret_scope is not None,
        "terminal_scope": terminal_scope is not None,
        "multiplex_context": bool(_call_quietly(getattr(multiplex_var, "get", None), False)),
        "multiplex_process": bool(_loaded_attr("agent.secret_scope", "_MULTIPLEX_ACTIVE")),
    }
    if identity is not None:
        try:
            pickle.loads(pickle.dumps(identity))
        except Exception as exc:
            raise ChildBoundaryUnavailable(f"cron execution identity is not transferable: {exc}") from exc
        snapshot["identity"] = identity
    needed = [scope for scope in ("secret_scope", "terminal_scope", "multiplex_context",
                                  "multiplex_process") if snapshot[scope]]
    missing = missing_fire_context_apis(needed)
    if missing:
        raise ChildBoundaryUnavailable(
            "the installed Agent cannot rebuild this fire's active scopes in a child "
            f"(missing {', '.join(missing)})")
    return snapshot


def _gateway_owner_bound(job, home) -> bool:
    """True when this process is the gateway cron owner for the fire or its home.

    ``cron.scheduler.run_job`` runs a bound owner execution locally and submits
    everything else to the profile's gateway; the binding is process-local, so
    it cannot follow the job into a child.
    """
    current = _call_quietly(_loaded_attr("gateway.session_cron", "current_execution"))
    if current is not None:
        return True
    owner_for_home = _loaded_attr("gateway.session_cron", "owner_for_home")
    if callable(owner_for_home):
        try:
            return owner_for_home(home) is not None
        except Exception:
            return False
    return False


# The Agent's frozen launch environment: once multi-profile hosting is active,
# the launch profile's env-only credentials resolve from this snapshot, never
# from the live os.environ (which WebUI's profile mirror can overwrite).
_LAUNCH_ENV_MODULE = "tui_gateway.launch_profile_policy"
_LAUNCH_ENV_ATTRS = ("_snapshot", "_lock")


def _launch_env_snapshot():
    """The parent's frozen launch env for the child, or None when none was captured.

    A process that never imported the policy module never captured one, so the
    child's own capture (from the same live env) decides exactly as the parent
    would.  A loaded module whose layout is unknown means the snapshot cannot be
    carried faithfully: keep the call in-process.
    """
    module = sys.modules.get(_LAUNCH_ENV_MODULE)
    if module is None:
        return None
    if not all(hasattr(module, attr) for attr in _LAUNCH_ENV_ATTRS):
        raise ChildBoundaryUnavailable(
            "the installed Agent's launch-environment snapshot cannot be carried to a child")
    with module._lock:
        snapshot = module._snapshot
        return None if snapshot is None else dict(snapshot)


def _install_launch_env_snapshot(snapshot) -> None:
    """Child side: adopt the parent's frozen launch env before anything captures one."""
    if snapshot is None:
        return
    module = importlib.import_module(_LAUNCH_ENV_MODULE)
    if not all(hasattr(module, attr) for attr in _LAUNCH_ENV_ATTRS):
        raise RuntimeError("cron child cannot install the parent's launch-environment snapshot")
    with module._lock:
        module._snapshot = dict(snapshot)


# Per-turn markers WebUI mirrors into the process environment while a chat turn
# streams.  An unattended job body must not inherit them (the Agent strips the
# same presence markers from its own restart-safe cron worker).
_CHILD_STRIPPED_ENV = (
    "HERMES_INTERACTIVE",
    "HERMES_GATEWAY_SESSION",
    "HERMES_EXEC_ASK",
    "HERMES_SINGLE_QUERY_SESSION",
    "HERMES_UI_SESSION_ID",
    "HERMES_CRON_SESSION",
    "HERMES_CRON_AUTO_DELIVER_PLATFORM",
    "HERMES_CRON_AUTO_DELIVER_CHAT_ID",
    "HERMES_CRON_AUTO_DELIVER_THREAD_ID",
)
_CHILD_STRIPPED_ENV_PREFIXES = ("HERMES_SESSION_",)


def _child_env(home: Path) -> dict:
    for _attempt in range(5):
        try:
            items = list(os.environ.items())
            break
        except RuntimeError:  # another thread resized os.environ mid-copy
            continue
    else:
        items = list(os.environ.copy().items())
    env = {
        key: value for key, value in items
        if key not in _CHILD_STRIPPED_ENV and not key.startswith(_CHILD_STRIPPED_ENV_PREFIXES)
    }
    env["HERMES_HOME"] = str(home)
    return env


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


# ── process-tree kill ───────────────────────────────────────────────────────
_TREE_SCAN_ROUNDS = 8


def _process_table():
    """``{pid: (ppid, start_token)}`` for visible processes, or None if unavailable.

    Linux reads ``/proc`` (no dependency); elsewhere psutil is used when it is
    installed.  Without either only process-group signalling remains.
    """
    if os.path.exists("/proc/self/stat"):
        table = {}
        try:
            names = os.listdir("/proc")
        except OSError:
            return None
        for name in names:
            if not name.isdigit():
                continue
            try:
                with open(f"/proc/{name}/stat", "rb") as fh:
                    fields = fh.read().rsplit(b")", 1)[1].split()
                # fields[1] = ppid (stat field 4); fields[19] = starttime (field 22)
                table[int(name)] = (int(fields[1]), fields[19])
            except (OSError, IndexError, ValueError):
                continue
        return table
    try:
        import psutil  # optional
    except ImportError:
        return None
    table = {}
    try:
        for proc in psutil.process_iter(["pid", "ppid", "create_time"]):
            info = proc.info
            table[info["pid"]] = (info["ppid"], info["create_time"])
    except Exception:
        return None
    return table


def _descendants(root: int, table: dict) -> list:
    children = {}
    for pid, (ppid, _start) in table.items():
        children.setdefault(ppid, []).append(pid)
    found, stack = [], [root]
    while stack:
        for pid in children.get(stack.pop(), ()):
            if pid != root and pid not in found:
                found.append(pid)
                stack.append(pid)
    return found


def kill_process_tree(root: int) -> int:
    """SIGKILL every descendant of *root* (not *root* itself), across sessions.

    Descendants are frozen with SIGSTOP and the tree is rescanned until no new
    pid appears, so a concurrent fork cannot escape a stale snapshot; each is
    then killed if it is still the same process (start time).  Call it while
    *root* is alive: once *root* exits its children are re-parented and can no
    longer be told apart from unrelated processes.  Returns the number killed.
    """
    frozen = {}
    for _round in range(_TREE_SCAN_ROUNDS):
        table = _process_table()
        if table is None:
            break
        fresh = [pid for pid in _descendants(root, table)
                 if pid not in frozen and pid != os.getpid()]
        if not fresh:
            break
        for pid in fresh:
            frozen[pid] = table[pid][1]
            try:
                os.kill(pid, signal.SIGSTOP)
            except (ProcessLookupError, PermissionError):
                pass
    if not frozen:
        return 0
    current = _process_table()
    killed = 0
    for pid, start in frozen.items():
        if current is not None and current.get(pid, (None, None))[1] != start:
            continue  # exited (or the pid was recycled) since it was frozen
        try:
            os.kill(pid, signal.SIGKILL)
            killed += 1
        except (ProcessLookupError, PermissionError):
            pass
    return killed


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
        if _gateway_owner_bound(job, self.home):
            raise ChildBoundaryUnavailable(
                "this process holds the gateway cron owner execution for the fire")
        payload = {
            "target": _callable_target(fn),
            "home": str(self.home),
            "job": job,
            "args": tuple(args),
            "kwargs": kwargs,
            "wants_cancel": self.parent_cancel is not None,
            "fire_context": _snapshot_fire_context(),
            "launch_env": _launch_env_snapshot(),
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
            env = _child_env(self.home)
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
                exited = self._wait_leader_exit(CHILD_TERMINATE_GRACE_SECONDS)
                self._sweep_group_and_reap()
                code = getattr(self.process, "returncode", None) if exited else "unknown (still running)"
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

    # Process-group signalling is only safe while the group id is pinned: the
    # leader (pid == pgid) is alive or an unreaped zombie.  Once it is reaped
    # and its group empties, the id can be recycled by an unrelated process, so
    # the leader is never reaped before the final group sweep.
    def _leader_reaped(self) -> bool:
        return self.process is None or self.process.returncode is not None

    def _wait_leader_exit(self, timeout: float) -> bool:
        """Wait up to *timeout* for the leader to exit WITHOUT reaping it."""
        process = self.process
        deadline = time.monotonic() + max(0.0, float(timeout))
        while True:
            if process is None or process.returncode is not None:
                return True
            try:
                exited = os.waitid(os.P_PID, process.pid,
                                   os.WEXITED | os.WNOWAIT | os.WNOHANG) is not None
            except ChildProcessError:
                # Reaped behind our back; Popen learns the status lazily.
                process.poll()
                return True
            except (AttributeError, OSError):
                exited = process.poll() is not None  # no waitid: best effort
            if exited:
                return True
            if time.monotonic() >= deadline:
                return False
            time.sleep(0.05)

    def _signal_group(self, sig) -> None:
        if self.process is None or self._leader_reaped():
            return
        try:
            os.killpg(self.process.pid, sig)
        except (ProcessLookupError, PermissionError):
            pass
        except OSError:
            logger.debug("cron child killpg(%s) failed", sig, exc_info=True)

    def _signal_leader(self, sig) -> None:
        if self.process is None or self._leader_reaped():
            return
        try:
            os.kill(self.process.pid, sig)
        except (ProcessLookupError, PermissionError):
            pass

    def _kill_descendant_tree(self) -> None:
        """Kill descendants in OTHER sessions/groups (Agent cron scripts) too.

        Only meaningful while the leader is still running: its exit re-parents
        them out of reach.
        """
        if self.process is None or self._leader_reaped() or self._wait_leader_exit(0):
            return
        try:
            kill_process_tree(self.process.pid)
        except Exception:
            logger.debug("cron child tree kill failed", exc_info=True)

    def _sweep_group_and_reap(self) -> None:
        """SIGKILL the descendant tree and whatever is left in the group, then reap."""
        process = self.process
        if process is None:
            return
        self._kill_descendant_tree()
        self._signal_group(signal.SIGKILL)
        try:
            process.wait(timeout=CHILD_TERMINATE_GRACE_SECONDS)
        except subprocess.TimeoutExpired:
            logger.warning("cron job child pid=%s did not exit after SIGKILL", process.pid)

    def _kill(self) -> None:
        if self.process is None:
            return
        if not self._wait_leader_exit(0):
            # Freeze the leader so it cannot spawn past the tree snapshot, kill
            # its whole descendant tree while it still anchors it, then let it
            # handle SIGTERM.
            self._signal_leader(signal.SIGSTOP)
            try:
                self._kill_descendant_tree()
                self._signal_group(signal.SIGTERM)
            finally:
                self._signal_leader(signal.SIGCONT)
            self._wait_leader_exit(CHILD_TERMINATE_GRACE_SECONDS)
        # Always sweep: descendants may outlive the leader.
        self._sweep_group_and_reap()

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
                if self._wait_leader_exit(CHILD_EXIT_GRACE_SECONDS):
                    # The leader exited; sweep any descendants it left in its group.
                    self._sweep_group_and_reap()
                else:
                    logger.warning(
                        "cron job child pid=%s still running %.0fs after returning its result; killing it",
                        self.process.pid, CHILD_EXIT_GRACE_SECONDS)
                    self._kill()
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


def _kill_own_tree_and_exit(code: int = 1) -> None:  # pragma: no cover - exits
    try:
        try:
            kill_process_tree(os.getpid())
        finally:
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
            _kill_own_tree_and_exit(1)
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
    """Re-install, for the pinned home, each per-fire scope active in the parent.

    The parent already refused the boundary when an active scope had no Agent
    helper (``_snapshot_fire_context``), so a helper missing HERE means the
    Agent changed on disk mid-launch: fail closed instead of running the body
    with unscoped secret reads.
    """
    resets = []

    def _require(scope):
        missing = missing_fire_context_apis([scope])
        if missing:
            raise RuntimeError(
                f"cron child cannot rebuild the fire's {scope} (missing {', '.join(missing)})")

    if fire_context.get("multiplex_process"):
        _require("multiplex_process")
        importlib.import_module("agent.secret_scope").set_multiplex_active(True)
    identity = fire_context.get("identity")
    if identity is not None:
        _require("identity")
        var = importlib.import_module("cron.execution_identity")._CURRENT
        identity_token = var.set(identity)
        resets.append(lambda: var.reset(identity_token))
    if fire_context.get("secret_scope") or fire_context.get("owned_fire"):
        if fire_context.get("secret_scope"):
            _require("secret_scope")
        scheduler = importlib.import_module("cron.scheduler")
        install_scope = getattr(scheduler, "_install_fire_secret_scope", None)
        reset_scope = getattr(scheduler, "_reset_fire_secret_scope", None)
        if callable(install_scope) and callable(reset_scope):
            scope_tokens = install_scope()
            resets.append(lambda: reset_scope(scope_tokens))
    if fire_context.get("multiplex_context"):
        # The fire scope re-derives routed-fire multiplexing from the pinned
        # launch home; mirror the parent's flag too so a fail-closed parent
        # never becomes a permissive child.
        _require("multiplex_context")
        secret_scope = importlib.import_module("agent.secret_scope")
        if not secret_scope.is_multiplex_active():
            multiplex_token = secret_scope.set_multiplex_context(True)
            resets.append(lambda: secret_scope.reset_multiplex_context(multiplex_token))
    if fire_context.get("terminal_scope") or fire_context.get("owned_fire"):
        if fire_context.get("terminal_scope"):
            _require("terminal_scope")
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
        _install_launch_env_snapshot(payload.get("launch_env"))
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
