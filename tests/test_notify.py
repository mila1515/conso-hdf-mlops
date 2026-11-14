import dataclasses

import pytest
import requests

import conso_hdf.notify as notify_mod
import conso_hdf.pipeline as pipeline
from conso_hdf.notify import build_payload, notify, webhook_targets

DISCORD = "https://discord.com/api/webhooks/123/abc"
SLACK = "https://hooks.slack.com/services/T000/B000/xyz"


class _Resp:
    def __init__(self, ok=True):
        self.ok = ok

    def raise_for_status(self):
        if not self.ok:
            raise requests.HTTPError("400")


@pytest.fixture
def posts(monkeypatch):
    calls = []

    def fake_post(url, json, timeout):
        calls.append((url, json))
        return _Resp(ok="fail" not in url)

    monkeypatch.setattr(notify_mod.requests, "post", fake_post)
    return calls


def test_payload_formats():
    d = build_payload("discord", "Titre", "Message", "critique", {"MAPE": "6.1 %"})
    embed = d["embeds"][0]
    assert embed["color"] == 0xEF4444 and embed["fields"][0] == {"name": "MAPE", "value": "6.1 %", "inline": True}
    s = build_payload("slack", "Titre", "Message", "critique", {"MAPE": "6.1 %"})
    assert set(s) == {"text"} and "*Titre*" in s["text"] and "MAPE" in s["text"]


def test_targets_and_legacy_url_detection(cfg):
    c = dataclasses.replace(cfg, discord_webhook_url=DISCORD, slack_webhook_url=SLACK)
    assert webhook_targets(c) == [("discord", DISCORD), ("slack", SLACK)]
    legacy = dataclasses.replace(cfg, alert_webhook_url=SLACK)
    assert webhook_targets(legacy) == [("slack", SLACK)]
    assert webhook_targets(cfg) == []


def test_notify_sends_to_all_and_never_raises(cfg, posts):
    c = dataclasses.replace(cfg, discord_webhook_url=DISCORD, slack_webhook_url=SLACK + "/fail")
    assert notify(c, "T", "M", "avertissement") == 1  # Slack échoue, Discord passe, aucune exception
    assert [u for u, _ in posts] == [DISCORD, SLACK + "/fail"]
    assert "embeds" in posts[0][1] and "text" in posts[1][1]


def test_run_url_in_github_actions(monkeypatch):
    monkeypatch.setenv("GITHUB_RUN_ID", "42")
    monkeypatch.setenv("GITHUB_REPOSITORY", "me/repo")
    p = build_payload("discord", "T", "M", "info", None)
    assert p["embeds"][0]["url"] == "https://github.com/me/repo/actions/runs/42"


def test_daily_sends_summary(cfg, posts, monkeypatch):
    c = dataclasses.replace(cfg, discord_webhook_url=DISCORD, notify_daily_summary=True)
    monkeypatch.setattr(pipeline, "settings", c)
    monkeypatch.setattr(pipeline, "ingest_all", lambda _cfg: None)
    pipeline.daily()
    titles = [p["embeds"][0]["title"] for _, p in posts]
    assert any("exécution quotidienne" in t for t in titles)
    summary = next(p for _, p in posts if "quotidienne" in p["embeds"][0]["title"])
    names = {f["name"] for f in summary["embeds"][0]["fields"]}
    assert {"Modèle", "Pic prévu", "Heure du pic"} <= names
