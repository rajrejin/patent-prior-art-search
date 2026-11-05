import json
import logging

from typer.testing import CliRunner

from patentrag.cli import app
from patentrag.config import Secrets, load_config
from patentrag.logging_utils import JsonFormatter
from patentrag.models import Patent, PubId


def test_config_loads_and_hash_is_stable():
    cfg = load_config()
    assert cfg.seed == 0
    assert "H01M10/" in cfg.data.cpc_prefixes
    assert cfg.config_hash() == load_config().config_hash()


def test_secrets_never_printed():
    s = Secrets(ops_key="abc123", ops_secret="zzz999")
    assert "abc123" not in repr(s)
    assert "zzz999" not in str(s)


def test_pubid_keys():
    p = PubId(country="EP", number="3456789", kind="A1")
    assert p.doc_key == "EP3456789"
    assert str(p) == "EP3456789A1"


def test_patent_query_text():
    p = Patent(pub_id=PubId(country="EP", number="1"), title="T", abstract="A b.")
    assert p.text_for_query() == "T. A b."


def test_json_formatter_drops_sensitive_fields():
    rec = logging.LogRecord("x", logging.INFO, "f", 1, "hello", None, None)
    rec.fields = {"password": "p", "ok": 1}
    out = json.loads(JsonFormatter().format(rec))
    assert "password" not in out
    assert out["ok"] == 1


def test_cli_help_lists_all_commands():
    res = CliRunner().invoke(app, ["--help"])
    assert res.exit_code == 0
    for cmd in [
        "fetch",
        "build-corpus",
        "build-graph",
        "build-index",
        "evaluate",
        "faithfulness",
        "faith-audit",
        "tune",
        "tune-agent",
        "ablate",
        "robustness",
        "serve-mcp",
        "serve-api",
        "report",
    ]:
        assert cmd in res.output
