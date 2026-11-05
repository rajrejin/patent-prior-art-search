"""Silent credential checks for EPO OPS. Never prints or logs secret values."""

from __future__ import annotations

import base64
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path

import httpx

from patentrag.config import ROOT, load_secrets

AUTH_URL = "https://ops.epo.org/3.2/auth/accesstoken"
PLACEHOLDER = re.compile(r"paste|your[_-]?(key|secret)|changeme|xxxx|<.*>", re.IGNORECASE)


@dataclass
class CheckResult:
    name: str
    ok: bool
    reason: str = ""


def basic_auth_header(key: str, secret: str) -> str:
    return "Basic " + base64.b64encode(f"{key}:{secret}".encode()).decode()


def check_env_shape(root: Path = ROOT) -> list[CheckResult]:
    results: list[CheckResult] = []
    env = root / ".env"
    results.append(CheckResult("env file present", env.exists(), "" if env.exists() else "missing"))
    try:
        ign = subprocess.run(
            ["git", "check-ignore", "-q", ".env"], cwd=root, capture_output=True
        ).returncode
    except OSError:
        ign = 1
    results.append(CheckResult("env gitignored", ign == 0, "" if ign == 0 else "not ignored"))
    s = load_secrets()
    for label, val in (("OPS_CONSUMER_KEY", s.ops_key), ("OPS_CONSUMER_SECRET", s.ops_secret)):
        if not val:
            results.append(CheckResult(label, False, "missing or empty"))
        elif PLACEHOLDER.search(val):
            results.append(CheckResult(label, False, "placeholder value"))
        elif val != val.strip() or val[0] in "\"'" or val[-1] in "\"'":
            results.append(CheckResult(label, False, "stray quotes or whitespace"))
        elif len(val) < 16:
            results.append(CheckResult(label, False, "implausibly short"))
        else:
            results.append(CheckResult(label, True))
    return results


def check_live() -> list[CheckResult]:
    """One real token request plus one tiny published-data request."""
    s = load_secrets()
    out: list[CheckResult] = []
    try:
        r = httpx.post(
            AUTH_URL,
            headers={"Authorization": basic_auth_header(s.ops_key, s.ops_secret)},
            data={"grant_type": "client_credentials"},
            timeout=30,
        )
    except httpx.HTTPError as exc:
        return [CheckResult("oauth token", False, f"network error ({type(exc).__name__})")]
    if r.status_code != 200:
        reason = {400: "400 bad request (check key/secret)", 401: "401 unauthorised"}.get(
            r.status_code, f"HTTP {r.status_code}"
        )
        return [CheckResult("oauth token", False, reason)]
    token = r.json().get("access_token", "")
    out.append(CheckResult("oauth token", bool(token), "" if token else "no token in response"))
    if not token:
        return out
    try:
        d = httpx.get(
            "https://ops.epo.org/3.2/rest-services/published-data/search",
            params={"q": "pn=EP1000000"},
            headers={"Authorization": f"Bearer {token}", "Accept": "application/json"},
            timeout=30,
        )
        out.append(CheckResult("data request", d.status_code == 200, f"HTTP {d.status_code}"))
    except httpx.HTTPError as exc:
        out.append(CheckResult("data request", False, f"network error ({type(exc).__name__})"))
    return out
