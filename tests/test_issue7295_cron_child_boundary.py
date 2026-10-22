"""Regression tests for #7295 / #6753: scheduled cron job bodies run behind a
profile-pinned child-process boundary.

Root cause: ``install_cron_scheduler_profile_isolation()`` wrapped every
scheduled ``cron.scheduler.run_job()`` in ``cron_profile_context_for_home()``,
which held the process-global ``_cron_env_lock`` until the whole synchronous
job returned (#7295). The in-chat ``cronjob(action='run')`` path entered the
same context in the streaming tool wrapper and called the job inside it
(#6753). Either way every ``/api/crons*`` reader that needed the lock froze for
the full length of the agent run.

These tests drive the real wrapper with a real spawned child running a slow
fake job body, and prove that:

* the parent does not hold ``_cron_env_lock`` while the job body runs, for the
  scheduled (non-nested) path and the in-chat (nested) path;
* the body runs in a different process pinned to the job's profile home;
* the result tuple, job-flag mutations and exception types survive the boundary;
* cancellation is forwarded to the child, and an unresponsive or timed-out
  child (and its descendants) is killed;
* the child dies with the parent, as an in-process run would;
* overlapping jobs for two profiles run concurrently, each in its own home.
"""
from __future__ import annotations

import importlib
import json
import os
import signal
import sys
import textwrap
import threading
import time
import types
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))
from conftest import requires_fork  # noqa: E402

FAKE_MODULE = "wt7295_fake_sched"

FAKE_SOURCE = textwrap.dedent(
    """
    import json
    import os
    import pathlib
    import subprocess
    import sys
    import time


    def _publish(ctl, name, payload):
        tmp = ctl / (name + ".tmp")
        tmp.write_text(json.dumps(payload), encoding="utf-8")
        os.replace(tmp, ctl / name)


    def run_job(job, *, extra_prompt=None, execution_id=None, cancel_event=None,
                defer_agent_teardown=None):
        ctl = pathlib.Path(job["ctl"])
        info = {
            "pid": os.getpid(),
            "home": os.environ.get("HERMES_HOME"),
            "extra_prompt": extra_prompt,
            "execution_id": execution_id,
            "has_cancel": cancel_event is not None,
            "has_defer": defer_agent_teardown is not None,
            "env": {k: os.environ.get(k) for k in job.get("env_probe", ())},
        }
        mode = job.get("mode", "wait")
        if mode == "ignore_cancel" or job.get("spawn_grandchild"):
            grandchild = subprocess.Popen(
                [sys.executable, "-c", "import time; time.sleep(120)"])
            info["grandchild_pid"] = grandchild.pid
        if mode == "progress":
            try:
                import cron.executions as executions
            except ImportError:
                info["progress_returned"] = "unavailable"
            else:
                info["progress_returned"] = executions.touch_execution_progress(execution_id)
        _publish(ctl, "started", info)
        deadline = time.monotonic() + float(job.get("max_wait", 30))
        while time.monotonic() < deadline:
            if (ctl / "release").exists():
                break
            if (cancel_event is not None and cancel_event.is_set()
                    and mode != "ignore_cancel"):
                job["_cancel_seen"] = True
                return False, "cancelled-output", "", "Interrupted by cancel"
            time.sleep(0.02)
        if mode == "raise":
            raise LookupError("boom from cron child")
        if mode == "crash":
            if os.getpid() == job["parent_pid"]:
                raise AssertionError("crash mode reached the parent process")
            os._exit(7)
        job["_model_unreachable"] = True
        return (True, "output home=" + str(os.environ.get("HERMES_HOME")),
                "final from child", None)
    """
)


def _wait_for(path: Path, timeout: float = 30.0) -> dict:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
        time.sleep(0.02)
    raise AssertionError(f"timed out waiting for {path}")


def _pid_gone(pid: int, timeout: float = 10.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            stat = Path(f"/proc/{pid}/stat").read_text()
        except (FileNotFoundError, ProcessLookupError):
            return True
        # field 3 is the state; a zombie has already exited
        if stat.rsplit(")", 1)[1].split()[0] in ("Z", "X"):
            return True
        time.sleep(0.05)
    return False


class _Harness:
    def __init__(self, tmp_path, monkeypatch):
        from api import profiles as p

        self.p = p
        self.tmp_path = tmp_path
        mods = tmp_path / "mods"
        mods.mkdir()
        (mods / f"{FAKE_MODULE}.py").write_text(FAKE_SOURCE, encoding="utf-8")
        monkeypatch.syspath_prepend(str(mods))
        sys.modules.pop(FAKE_MODULE, None)
        self.fake = importlib.import_module(FAKE_MODULE)
        monkeypatch.setitem(sys.modules, FAKE_MODULE, self.fake)

        self.default_home = tmp_path / "home"
        self.research_home = self.default_home / "profiles" / "research"
        self.ops_home = self.default_home / "profiles" / "ops"
        for home in (self.default_home, self.research_home, self.ops_home):
            (home / "cron").mkdir(parents=True, exist_ok=True)

        self.progress_calls = []
        self.events = []
        self.ctls = []
        self.threads = []

        cron_pkg = types.ModuleType("cron")
        cron_pkg.__path__ = []
        cron_jobs = types.ModuleType("cron.jobs")
        cron_jobs.HERMES_DIR = self.default_home
        cron_jobs.CRON_DIR = self.default_home / "cron"
        cron_jobs.JOBS_FILE = cron_jobs.CRON_DIR / "jobs.json"
        cron_jobs.OUTPUT_DIR = cron_jobs.CRON_DIR / "output"
        cron_scheduler = types.ModuleType("cron.scheduler")
        cron_scheduler._hermes_home = None
        cron_scheduler._LOCK_DIR = self.default_home / "cron"
        cron_scheduler._LOCK_FILE = cron_scheduler._LOCK_DIR / ".tick.lock"
        cron_scheduler.run_job = self.fake.run_job
        cron_executions = types.ModuleType("cron.executions")

        def _record_progress(execution_id):
            self.progress_calls.append((execution_id, os.getpid(), threading.get_ident()))
            return True

        cron_executions.touch_execution_progress = _record_progress
        for name, mod in (
            ("cron", cron_pkg),
            ("cron.jobs", cron_jobs),
            ("cron.scheduler", cron_scheduler),
            ("cron.executions", cron_executions),
        ):
            monkeypatch.setitem(sys.modules, name, mod)
        self.cron_jobs = cron_jobs
        self.cron_scheduler = cron_scheduler

        monkeypatch.setenv("HERMES_HOME", str(self.default_home))
        monkeypatch.setattr(p, "_DEFAULT_HERMES_HOME", self.default_home)
        monkeypatch.setattr(
            p,
            "publish_session_list_changed",
            lambda reason, profile=None: self.events.append(("publish", reason, profile)),
        )
        p.install_cron_scheduler_profile_isolation()
        assert getattr(cron_scheduler.run_job, "_webui_profile_isolated", False)
        assert not p._cron_env_lock.locked(), "cron env lock leaked from an earlier test"

    def job(self, name, **extra):
        ctl = self.tmp_path / f"ctl-{name}"
        ctl.mkdir()
        self.ctls.append(ctl)
        job = {"id": f"job-{name}", "name": name, "ctl": str(ctl), "parent_pid": os.getpid()}
        job.update(extra)
        return job, ctl

    def cleanup(self):
        """Release every fake job and wait for its runner so failures never cascade."""
        for ctl in self.ctls:
            try:
                (ctl / "release").write_text("go")
            except OSError:
                pass
        for thread in self.threads:
            thread.join(60)
        deadline = time.monotonic() + 60
        while self.p._cron_env_lock.locked() and time.monotonic() < deadline:
            time.sleep(0.05)

    def run_in_thread(self, job, *args, **kwargs):
        box = {}

        def _target():
            try:
                box["result"] = self.cron_scheduler.run_job(job, *args, **kwargs)
            except BaseException as exc:  # surfaced through box
                box["error"] = exc

        thread = threading.Thread(target=_target, daemon=True)
        thread.start()
        self.threads.append(thread)
        return thread, box

    def assert_reader_not_blocked(self, home, timeout=2.0):
        """A /api/crons-style reader must acquire the cron context promptly."""
        entered = threading.Event()

        def _reader():
            with self.p.cron_profile_context_for_home(home):
                entered.set()

        reader = threading.Thread(target=_reader, daemon=True)
        reader.start()
        ok = entered.wait(timeout)
        reader.join(timeout)
        assert ok, (
            "cron_profile_context_for_home stayed blocked while the scheduled job "
            "body was running; the parent still holds _cron_env_lock for the job"
        )


@pytest.fixture
def harness(tmp_path, monkeypatch):
    h = _Harness(tmp_path, monkeypatch)
    try:
        yield h
    finally:
        h.cleanup()
    assert not h.p._cron_env_lock.locked(), "cron env lock leaked after test"


def test_scheduled_run_job_does_not_hold_cron_lock_while_body_runs(harness):
    """#7295: the scheduled/fire-claim path must not freeze /api/crons*."""
    job, ctl = harness.job("scheduled", profile="research")
    thread, box = harness.run_in_thread(job)

    info = _wait_for(ctl / "started")
    # TLS-profile reader (the /api/crons GET shape) and explicit-home reader.
    harness.assert_reader_not_blocked(harness.ops_home)
    harness.p.set_request_profile("ops")
    try:
        entered = threading.Event()

        def _tls_reader():
            harness.p.set_request_profile("ops")
            with harness.p.cron_profile_context():
                entered.set()

        t = threading.Thread(target=_tls_reader, daemon=True)
        t.start()
        assert entered.wait(2.0), "cron_profile_context() blocked behind a running job"
        t.join(2)
    finally:
        harness.p.clear_request_profile()

    (ctl / "release").write_text("go")
    thread.join(30)
    assert not thread.is_alive()
    assert "error" not in box, box.get("error")

    assert info["pid"] != os.getpid(), "job body ran inside the WebUI parent process"
    assert info["home"] == str(harness.research_home), "child was not pinned to the job profile home"
    assert box["result"] == (True, f"output home={harness.research_home}", "final from child", None)
    # run_job's job-flag mutations must reach the parent's job dict.
    assert job["_model_unreachable"] is True
    assert ("publish", "cron_complete", "research") in harness.events
    assert os.environ["HERMES_HOME"] == str(harness.default_home)


def test_in_chat_nested_run_releases_lock_and_restores_outer_context(harness):
    """#6753: the streaming cronjob wrapper's context must not stay locked for the run."""
    from api import streaming as st

    turn_home = harness.ops_home
    job, ctl = harness.job("inchat")
    observed = {}

    def original_handler(args, **kwargs):
        # Shape of tools.cronjob_tools: claim -> run_one_job -> run_job -> bookkeeping.
        observed["before"] = (os.environ.get("HERMES_HOME"), harness.cron_jobs.JOBS_FILE)
        result = harness.cron_scheduler.run_job(job, extra_prompt="ctx", execution_id="exec-1")
        observed["after"] = (
            os.environ.get("HERMES_HOME"),
            harness.cron_jobs.JOBS_FILE,
            harness.p._cron_env_lock.locked(),
            harness.p._cron_profile_context_depth(),
        )
        return json.dumps({"success": True, "result": list(result)})

    handler = st._streaming_cronjob_profile_handler(original_handler)
    token = st._STREAMING_CRON_PROFILE_HOME.set(str(turn_home))
    box = {}

    def _turn():
        ctx_token = st._STREAMING_CRON_PROFILE_HOME.set(str(turn_home))
        try:
            box["result"] = handler({"action": "run", "job_id": job["id"]})
        except BaseException as exc:
            box["error"] = exc
        finally:
            st._STREAMING_CRON_PROFILE_HOME.reset(ctx_token)

    try:
        thread = threading.Thread(target=_turn, daemon=True)
        thread.start()
        harness.threads.append(thread)
        info = _wait_for(ctl / "started")
        harness.assert_reader_not_blocked(harness.research_home)
        (ctl / "release").write_text("go")
        thread.join(30)
    finally:
        st._STREAMING_CRON_PROFILE_HOME.reset(token)

    assert not thread.is_alive()
    assert "error" not in box, box.get("error")
    assert info["pid"] != os.getpid()
    assert info["home"] == str(turn_home)
    assert info["extra_prompt"] == "ctx"
    assert info["execution_id"] == "exec-1"
    payload = json.loads(box["result"])
    assert payload["result"] == [True, f"output home={turn_home}", "final from child", None]
    # The outer tool context is re-entered before the caller's bookkeeping resumes.
    assert observed["before"] == (str(turn_home), turn_home / "cron" / "jobs.json")
    assert observed["after"] == (str(turn_home), turn_home / "cron" / "jobs.json", True, 1)
    # ...and fully restored after the tool call returns.
    assert os.environ["HERMES_HOME"] == str(harness.default_home)
    assert harness.cron_jobs.JOBS_FILE == harness.default_home / "cron" / "jobs.json"


def test_child_exception_type_and_message_survive_boundary(harness):
    job, ctl = harness.job("raises", mode="raise", profile="research")
    (ctl / "release").write_text("go")
    with pytest.raises(LookupError, match="boom from cron child"):
        harness.cron_scheduler.run_job(job)
    info = _wait_for(ctl / "started")
    assert info["pid"] != os.getpid()


def test_non_transferable_kwargs_stay_in_parent(harness):
    """defer_agent_teardown holds live agents; the child tears down inline."""
    job, ctl = harness.job("kwargs", profile="research")
    (ctl / "release").write_text("go")
    deferred = []
    result = harness.cron_scheduler.run_job(
        job, defer_agent_teardown=deferred, extra_prompt="x", execution_id="e-9")
    info = _wait_for(ctl / "started")
    assert info["pid"] != os.getpid()
    assert info["has_defer"] is False
    assert info["has_cancel"] is False
    assert (info["extra_prompt"], info["execution_id"]) == ("x", "e-9")
    assert deferred == []
    assert result[0] is True


def test_child_crash_raises_without_leaking_lock(harness):
    job, ctl = harness.job("crash", mode="crash", profile="research")
    (ctl / "release").write_text("go")
    with pytest.raises(RuntimeError, match="exit"):
        harness.cron_scheduler.run_job(job)
    info = _wait_for(ctl / "started")
    assert _pid_gone(info["pid"])


def test_cancel_event_is_forwarded_to_child(harness):
    job, ctl = harness.job("cancel", profile="research")
    cancel = threading.Event()
    thread, box = harness.run_in_thread(job, cancel_event=cancel)
    info = _wait_for(ctl / "started")
    assert info["has_cancel"] is True
    cancel.set()
    thread.join(30)
    assert not thread.is_alive()
    assert box.get("result") == (False, "cancelled-output", "", "Interrupted by cancel")
    assert job["_cancel_seen"] is True
    assert info["pid"] != os.getpid()


def test_unresponsive_child_and_descendants_are_killed_after_cancel_grace(harness, monkeypatch):
    from api import cron_child

    monkeypatch.setattr(cron_child, "CHILD_CANCEL_GRACE_SECONDS", 0.5)
    monkeypatch.setattr(cron_child, "CHILD_TERMINATE_GRACE_SECONDS", 1.0)
    job, ctl = harness.job("stubborn", mode="ignore_cancel", profile="research", max_wait=60)
    cancel = threading.Event()
    thread, box = harness.run_in_thread(job, cancel_event=cancel)
    info = _wait_for(ctl / "started")
    cancel.set()
    thread.join(30)
    assert not thread.is_alive()
    assert isinstance(box.get("error"), RuntimeError), box
    assert "cancel" in str(box["error"]).lower()
    assert _pid_gone(info["pid"]), "cron child survived cancellation"
    assert _pid_gone(info["grandchild_pid"]), "cron child's descendants survived cancellation"


def test_job_timeout_kills_child(harness, monkeypatch):
    from api import cron_child

    monkeypatch.setattr(cron_child, "CHILD_TIMEOUT_GRACE_SECONDS", 0.0)
    monkeypatch.setattr(cron_child, "CHILD_TIMEOUT_FLOOR_SECONDS", 0.0)
    monkeypatch.setattr(cron_child, "CHILD_TERMINATE_GRACE_SECONDS", 1.0)
    # The backstop counts from spawn, and a child imports the real Agent
    # cron.scheduler (~2s on a loaded host) before the body starts: leave
    # enough room for startup so the kill lands mid-body, not mid-import.
    job, ctl = harness.job(
        "timeout", mode="ignore_cancel", profile="research", max_wait=60, timeout_seconds=12)
    thread, box = harness.run_in_thread(job)
    info = _wait_for(ctl / "started")
    assert thread.is_alive(), "backstop fired before the job body was observed running"
    thread.join(45)
    assert not thread.is_alive()
    assert isinstance(box.get("error"), RuntimeError), box
    assert "timed out" in str(box["error"]).lower() or "timeout" in str(box["error"]).lower()
    assert _pid_gone(info["pid"])
    assert _pid_gone(info["grandchild_pid"])


def test_overlapping_jobs_for_two_profiles_run_concurrently(harness):
    job_a, ctl_a = harness.job("overlap-a", profile="research")
    job_b, ctl_b = harness.job("overlap-b", profile="ops")
    t_a, box_a = harness.run_in_thread(job_a)
    t_b, box_b = harness.run_in_thread(job_b)
    info_a = _wait_for(ctl_a / "started")
    info_b = _wait_for(ctl_b / "started")
    # Both bodies are running at the same time: neither serialized on the lock.
    assert not (ctl_a / "release").exists() and not (ctl_b / "release").exists()
    harness.assert_reader_not_blocked(harness.default_home)
    (ctl_a / "release").write_text("go")
    (ctl_b / "release").write_text("go")
    t_a.join(30)
    t_b.join(30)
    assert info_a["home"] == str(harness.research_home)
    assert info_b["home"] == str(harness.ops_home)
    assert len({info_a["pid"], info_b["pid"], os.getpid()}) == 3
    assert box_a["result"][1] == f"output home={harness.research_home}"
    assert box_b["result"][1] == f"output home={harness.ops_home}"


def test_child_progress_stamps_are_relayed_to_owning_parent(harness):
    """Agent ledger progress writes are fenced to the owning process; relay them."""
    job, ctl = harness.job("progress", mode="progress", profile="research")
    (ctl / "release").write_text("go")
    result = harness.cron_scheduler.run_job(job, execution_id="exec-progress")
    info = _wait_for(ctl / "started")
    assert result[0] is True
    assert info["pid"] != os.getpid()
    if info["progress_returned"] == "unavailable":
        pytest.skip("Agent cron.executions is not importable in the child (agent-free CI)")
    assert info["progress_returned"] is True
    assert [call[:2] for call in harness.progress_calls] == [("exec-progress", os.getpid())]


@requires_fork
def test_cron_child_dies_with_parent(tmp_path):
    """A WebUI crash/restart must not leave an orphaned job body running."""
    import multiprocessing

    ctl = tmp_path / "ctl-orphan"
    ctl.mkdir()
    ctx = multiprocessing.get_context("fork")
    harness_proc = ctx.Process(target=_orphan_harness_main, args=(str(tmp_path), str(ctl)))
    harness_proc.start()
    try:
        info = _wait_for(ctl / "started", timeout=60)
        assert info["pid"] != harness_proc.pid
        os.kill(harness_proc.pid, signal.SIGKILL)
        harness_proc.join(10)
        assert _pid_gone(info["pid"], timeout=15), "cron child outlived its parent"
        assert _pid_gone(info["grandchild_pid"], timeout=15)
    finally:
        if harness_proc.is_alive():
            harness_proc.kill()
            harness_proc.join(5)


def _orphan_harness_main(tmp_root, ctl):  # pragma: no cover - runs in forked process
    import pytest as _pytest

    mp = _pytest.MonkeyPatch()
    try:
        root = Path(tmp_root) / "orphan"
        root.mkdir()
        h = _Harness(root, mp)
        job = {
            "id": "job-orphan", "ctl": ctl, "mode": "ignore_cancel", "profile": "research",
            "max_wait": 60, "parent_pid": os.getpid(),
        }
        h.cron_scheduler.run_job(job)
    finally:
        os._exit(0)


def test_child_env_drops_per_turn_chat_markers(harness, monkeypatch):
    """A child spawned while a chat turn streams must not inherit that turn's markers."""
    for key, value in (
        ("HERMES_EXEC_ASK", "1"),
        ("HERMES_INTERACTIVE", "1"),
        ("HERMES_GATEWAY_SESSION", "1"),
        ("HERMES_SESSION_KEY", "chat-key"),
        ("HERMES_SESSION_PLATFORM", "webui"),
        ("WT7295_KEEP", "kept"),
    ):
        monkeypatch.setenv(key, value)
    probe = ("HERMES_EXEC_ASK", "HERMES_INTERACTIVE", "HERMES_GATEWAY_SESSION",
             "HERMES_SESSION_KEY", "HERMES_SESSION_PLATFORM", "WT7295_KEEP")
    job, ctl = harness.job("env", profile="research", env_probe=probe)
    (ctl / "release").write_text("go")
    harness.cron_scheduler.run_job(job)
    info = _wait_for(ctl / "started")
    assert info["pid"] != os.getpid()
    assert info["env"] == {
        "HERMES_EXEC_ASK": None, "HERMES_INTERACTIVE": None, "HERMES_GATEWAY_SESSION": None,
        "HERMES_SESSION_KEY": None, "HERMES_SESSION_PLATFORM": None, "WT7295_KEEP": "kept",
    }
    # The parent's own environment is untouched.
    assert os.environ["HERMES_EXEC_ASK"] == "1"


def test_bound_gateway_owner_execution_stays_in_process(harness, monkeypatch):
    """A process-local gateway owner binding cannot follow the fire into a child."""
    job, ctl = harness.job("owner", profile="research")
    session_cron = types.ModuleType("gateway.session_cron")
    session_cron.current_execution = lambda: ("authority", "sid", job["id"], "exec-o")
    session_cron.owner_for_home = lambda home: None
    monkeypatch.setitem(sys.modules, "gateway.session_cron", session_cron)
    (ctl / "release").write_text("go")
    result = harness.cron_scheduler.run_job(job, execution_id="exec-o")
    info = _wait_for(ctl / "started")
    assert info["pid"] == os.getpid(), "owner-bound fire was moved into a child"
    assert info["home"] == str(harness.research_home)
    assert result[0] is True


def test_active_scope_the_agent_cannot_rebuild_stays_in_process(harness, monkeypatch):
    """Never run a child without a secret scope the parent had active."""
    secret_scope = types.ModuleType("agent.secret_scope")
    secret_scope.current_secret_scope = lambda: {"API_KEY": "scoped"}
    monkeypatch.setitem(sys.modules, "agent.secret_scope", secret_scope)
    # The harness's fake cron.scheduler has no _install_fire_secret_scope.
    job, ctl = harness.job("scope", profile="research")
    (ctl / "release").write_text("go")
    result = harness.cron_scheduler.run_job(job)
    info = _wait_for(ctl / "started")
    assert info["pid"] == os.getpid(), "child ran without the parent's active secret scope"
    assert result[0] is True


def test_child_fails_closed_when_scope_helper_vanishes(monkeypatch):
    from api import cron_child

    monkeypatch.setitem(sys.modules, "cron.scheduler", types.ModuleType("cron.scheduler"))
    with pytest.raises(RuntimeError, match="secret_scope"):
        cron_child._install_fire_context({"secret_scope": True}, Path("/nonexistent-home"))


def test_descendants_left_after_a_normal_return_are_swept(harness):
    job, ctl = harness.job("leftover", profile="research", spawn_grandchild=True)
    (ctl / "release").write_text("go")
    result = harness.cron_scheduler.run_job(job)
    info = _wait_for(ctl / "started")
    assert result[0] is True
    assert _pid_gone(info["pid"], timeout=15)
    assert _pid_gone(info["grandchild_pid"], timeout=15), "child's leftover descendant survived"


def test_fire_context_contract_matches_installed_agent(tmp_path):
    """Pin every Agent API the child uses to rebuild per-fire scopes (version skew guard).

    Skips only when no Agent is importable; with an Agent present a renamed or
    removed helper fails here instead of silently dropping a scope in the child.
    """
    import dataclasses
    import inspect

    scheduler = pytest.importorskip("cron.scheduler")
    from api import cron_child
    from api import profiles as p

    assert cron_child.missing_fire_context_apis() == []
    execution_identity = importlib.import_module("cron.execution_identity")
    terminal_scope = importlib.import_module("tools.terminal_scope")
    secret_scope = importlib.import_module("agent.secret_scope")
    hermes_constants = importlib.import_module("hermes_constants")
    inspect.signature(scheduler._install_fire_secret_scope).bind()
    inspect.signature(scheduler._reset_fire_secret_scope).bind(object())
    inspect.signature(terminal_scope.install_profile_terminal_scope).bind(tmp_path)
    inspect.signature(terminal_scope.reset_terminal_scope).bind(object())
    inspect.signature(hermes_constants.pin_process_hermes_home).bind(str(tmp_path))
    inspect.signature(hermes_constants.get_routing_process_hermes_home).bind()
    for name in ("current_secret_scope", "current_secret_scope_home"):
        inspect.signature(getattr(secret_scope, name)).bind()
    inspect.signature(terminal_scope.get_terminal_scope).bind()
    session_cron = importlib.import_module("gateway.session_cron")
    inspect.signature(session_cron.current_execution).bind()
    inspect.signature(session_cron.owner_for_home).bind(tmp_path)

    identity_cls = execution_identity.CronExecution
    identity = identity_cls(**{f.name: "x" for f in dataclasses.fields(identity_cls)})
    assert pickle_roundtrip(identity) == identity

    home = tmp_path / "home"
    (home / "cron").mkdir(parents=True)
    fire = {"owned_fire": True, "identity": identity, "secret_scope": True,
            "terminal_scope": True, "multiplex_context": True}
    with p.cron_profile_context_for_home(home):
        resets = cron_child._install_fire_context(fire, home)
        try:
            assert execution_identity._CURRENT.get() == identity
            assert secret_scope.current_secret_scope() is not None
            assert Path(secret_scope.current_secret_scope_home()).resolve() == home.resolve()
            assert secret_scope.is_multiplex_active()
            assert terminal_scope.get_terminal_scope() is not None
        finally:
            for reset in reversed(resets):
                reset()
    assert execution_identity._CURRENT.get() is None
    assert secret_scope.current_secret_scope() is None
    assert terminal_scope.get_terminal_scope() is None


def pickle_roundtrip(obj):
    import pickle

    return pickle.loads(pickle.dumps(obj))


_E2E_PROBE = textwrap.dedent(
    """
    import json, os, pathlib, sys, threading, time

    root = pathlib.Path(sys.argv[1])
    sys.path.insert(0, sys.argv[2])
    sys.path.insert(1, sys.argv[3])
    home = pathlib.Path(os.environ["HERMES_HOME"])
    marker = root / "script-ran.json"
    (home / "scripts" / "probe.py").write_text(
        "import json, os, time, pathlib\\n"
        f"pathlib.Path({str(marker)!r}).write_text(json.dumps({{'ppid': os.getppid(), "
        "'home': os.environ.get('HERMES_HOME')}))\\n"
        "time.sleep(3)\\nprint('probe-output-ok')\\n", encoding="utf-8")

    from api import profiles as p
    p.install_cron_scheduler_profile_isolation()
    from cron.jobs import create_job, get_job
    from cron.executions import list_executions
    from tools.cronjob_tools import _claim_for_manual_run, _run_claimed_job

    job = create_job(prompt=None, schedule="every 1h", name="e2e-7295", script="probe.py", no_agent=True)
    claimed, err = _claim_for_manual_run(job["id"], "e2e")
    assert err is None, err
    box = {}
    # The Agent's background cronjob(action='run') body, on a delegation-pool-named thread.
    runner = threading.Thread(
        target=lambda: box.update(res=_run_claimed_job(claimed)), name="async-delegate_0")
    runner.start()
    deadline = time.monotonic() + 60
    while not marker.exists() and time.monotonic() < deadline:
        time.sleep(0.05)
    started = json.loads(marker.read_text()) if marker.exists() else None
    entered = threading.Event()

    def _reader():
        with p.cron_profile_context_for_home(home):
            entered.set()

    threading.Thread(target=_reader, daemon=True).start()
    reader_ok = entered.wait(2.0)
    runner.join(90)
    rows = list_executions(job_id=job["id"])
    print("E2E-RESULT " + json.dumps({
        "started": started, "reader_ok": reader_ok, "parent_pid": os.getpid(),
        "alive": runner.is_alive(), "res": box.get("res"),
        "last_status": (get_job(job["id"]) or {}).get("last_status"),
        "rows": [{"status": r.get("status"), "pid": r.get("pid")} for r in rows],
        "lock_after": p._cron_env_lock.locked(),
    }, default=str))
    """
)


def test_background_delegation_fire_runs_body_in_child_end_to_end(tmp_path):
    """#7295 end to end against the real Agent: the async-delegate ``_run_claimed_job`` body
    (claim -> run_one_job -> run_job) runs its script in the profile-pinned child, the parent
    stays lock-free, and the parent-owned lifecycle (ledger row, status) completes."""
    import subprocess as sp

    pytest.importorskip("cron.scheduler")
    pytest.importorskip("tools.cronjob_tools")
    pytest.importorskip("cron.executions")
    from conftest import _discover_agent_dir

    agent_dir = _discover_agent_dir()
    repo_root = Path(__file__).resolve().parent.parent
    home = tmp_path / "home"
    (home / "cron").mkdir(parents=True)
    (home / "scripts").mkdir()
    probe = tmp_path / "probe_main.py"
    probe.write_text(_E2E_PROBE, encoding="utf-8")
    env = {k: v for k, v in os.environ.items() if not k.startswith("HERMES_")}
    env.update(HERMES_HOME=str(home), HERMES_WEBUI_STATE_DIR=str(tmp_path / "state"),
               HERMES_CRON_SCRIPT_TIMEOUT="60", TMPDIR=str(tmp_path))
    proc = sp.run([sys.executable, str(probe), str(tmp_path), str(repo_root), str(agent_dir)],
                  env=env, capture_output=True, text=True, timeout=180)
    line = next((ln for ln in proc.stdout.splitlines() if ln.startswith("E2E-RESULT ")), None)
    assert line, f"probe failed rc={proc.returncode}\n{proc.stdout[-3000:]}\n{proc.stderr[-3000:]}"
    out = json.loads(line[len("E2E-RESULT "):])
    assert out["started"], "script never ran"
    assert out["started"]["home"] == str(home)
    assert out["started"]["ppid"] != out["parent_pid"], "job body ran in the WebUI parent"
    assert out["reader_ok"], "cron reader blocked while the background fire's body ran"
    assert not out["alive"]
    assert out["res"] == {"claimed": True, "success": True, "error": None}
    assert out["last_status"] == "ok"
    assert [r["status"] for r in out["rows"]] == ["completed"]
    assert out["rows"][0]["pid"] == out["parent_pid"], "ledger row not owned by the parent"
    assert out["lock_after"] is False
