#!/usr/bin/env python3
"""Run a command and report it to an Uptime Kuma Push monitor.

usage: run_with_heartbeat.py <push_url | @url_file> -- <command> [args...]

- Sends status=up as soon as the command starts, then every HEARTBEAT_INTERVAL seconds
  (default 60) while it runs.
- When the command exits, sends status=down&msg=exit_<code> and exits with the same code.
  Death by signal is reported shell-style (SIGINT -> exit_130, SIGTERM -> exit_143).
- "@url_file" reads the push URL from that file, keeping the token out of argv and the repo.
- Ctrl-C / SIGHUP from the terminal already reach the command (same process group), so the
  wrapper only notes them and keeps waiting to send the final "down". SIGTERM sent to the
  wrapper alone is forwarded to the command.
- An unreachable Kuma never stops the command; failed pings are logged to stderr.
"""
import os
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

INTERVAL = float(os.environ.get("HEARTBEAT_INTERVAL", "60"))
PUSH_TIMEOUT = 10


def log(msg):
    try:
        print(f"[heartbeat {time.strftime('%H:%M:%S')}] {msg}", file=sys.stderr, flush=True)
    except OSError:  # stderr gone (e.g. tee was interrupted); keep going
        pass


def push_base(url):
    """Strip Kuma's default query (?status=up&msg=OK&ping=) so we can set our own."""
    parts = urllib.parse.urlsplit(url.strip())
    if parts.scheme not in ("http", "https") or "/api/push/" not in parts.path:
        sys.exit("run_with_heartbeat: push URL must look like http(s)://host:port/api/push/<token>")
    return urllib.parse.urlunsplit((parts.scheme, parts.netloc, parts.path, "", ""))


def push(base, status, msg):
    url = base + "?" + urllib.parse.urlencode({"status": status, "msg": msg})
    try:
        with urllib.request.urlopen(url, timeout=PUSH_TIMEOUT) as resp:
            resp.read()
    except urllib.error.HTTPError as e:
        log(f"push {status} failed: HTTP {e.code}")
    except Exception as e:  # never let monitoring break the run; never log the URL (token)
        log(f"push {status} failed: {type(e).__name__} {getattr(e, 'reason', '')}")


def main(argv):
    if len(argv) < 3 or argv[1] != "--":
        sys.exit(__doc__.strip().splitlines()[2])
    target, cmd = argv[0], argv[2:]
    if target.startswith("@"):
        with open(target[1:]) as f:
            target = f.read()
    base = push_base(target)

    proc = None

    def on_terminal_signal(signum, _frame):
        log(f"got {signal.Signals(signum).name}; waiting for the command to exit")

    def on_term(_signum, _frame):
        log("got SIGTERM; forwarding to the command")
        if proc is not None and proc.poll() is None:
            proc.send_signal(signal.SIGTERM)

    # Python-level handlers are reset to the default in the child on exec, so the
    # command still dies normally on Ctrl-C.
    signal.signal(signal.SIGINT, on_terminal_signal)
    signal.signal(signal.SIGHUP, on_terminal_signal)
    signal.signal(signal.SIGTERM, on_term)

    try:
        proc = subprocess.Popen(cmd)
    except OSError as e:
        log(f"could not start {cmd[0]!r}: {e.strerror}")
        push(base, "down", "spawn_failed")
        return 127

    start = time.monotonic()
    log(f"started pid {proc.pid}; heartbeat every {INTERVAL:g}s")
    while proc.poll() is None:
        push(base, "up", f"running_{int(time.monotonic() - start) // 60}m")
        try:
            proc.wait(timeout=INTERVAL)
        except subprocess.TimeoutExpired:
            pass

    code = 128 - proc.returncode if proc.returncode < 0 else proc.returncode
    log(f"command exited with {code}")
    push(base, "down", f"exit_{code}")
    return code


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
