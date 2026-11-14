"""Notifications Discord / Slack (alertes et résumé quotidien).

Configuration (.env ou secrets GitHub Actions) :
- `DISCORD_WEBHOOK_URL` : webhook de salon Discord (Paramètres du salon > Intégrations > Webhooks)
- `SLACK_WEBHOOK_URL`   : Incoming Webhook Slack (api.slack.com/apps > Incoming Webhooks)
- `ALERT_WEBHOOK_URL`   : ancien réglage, toujours accepté (type détecté d'après l'URL)

Chaque service reçoit un message dans son format natif (embed coloré pour Discord, mrkdwn pour Slack).
Une notification ne fait jamais planter le pipeline : les erreurs d'envoi sont seulement journalisées.
"""

from __future__ import annotations

import logging
import os
from datetime import datetime, timezone

import requests

from conso_hdf.config import Settings

log = logging.getLogger(__name__)

SEVERITIES = {
    # sévérité : (emoji, couleur Discord)
    "info": ("🔵", 0x3B82F6),
    "succès": ("🟢", 0x22C55E),
    "avertissement": ("🟠", 0xF59E0B),
    "critique": ("🔴", 0xEF4444),
}


def _kind(url: str) -> str:
    if "hooks.slack.com" in url:
        return "slack"
    if "discord.com/api/webhooks" in url or "discordapp.com/api/webhooks" in url:
        return "discord"
    return "generic"


def webhook_targets(cfg: Settings) -> list[tuple[str, str]]:
    targets: list[tuple[str, str]] = []
    if cfg.discord_webhook_url:
        targets.append(("discord", cfg.discord_webhook_url))
    if cfg.slack_webhook_url:
        targets.append(("slack", cfg.slack_webhook_url))
    if cfg.alert_webhook_url and cfg.alert_webhook_url not in (cfg.discord_webhook_url, cfg.slack_webhook_url):
        targets.append((_kind(cfg.alert_webhook_url), cfg.alert_webhook_url))
    return targets


def run_url() -> str | None:
    """Lien vers l'exécution GitHub Actions en cours (si le pipeline tourne dans Actions)."""
    if os.getenv("GITHUB_RUN_ID") and os.getenv("GITHUB_REPOSITORY"):
        server = os.getenv("GITHUB_SERVER_URL", "https://github.com")
        return f"{server}/{os.environ['GITHUB_REPOSITORY']}/actions/runs/{os.environ['GITHUB_RUN_ID']}"
    return None


def build_payload(kind: str, title: str, message: str, severity: str, fields: dict | None) -> dict:
    emoji, color = SEVERITIES.get(severity, SEVERITIES["info"])
    fields = {k: str(v) for k, v in (fields or {}).items()}
    link = run_url()
    if kind == "discord":
        embed = {
            "title": f"{emoji} {title}"[:256],
            "description": message[:4000],
            "color": color,
            "fields": [{"name": k[:256], "value": v[:1024], "inline": True} for k, v in fields.items()][:25],
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "footer": {"text": "conso-hdf-mlops"},
        }
        if link:
            embed["url"] = link
        return {"username": "conso-hdf", "embeds": [embed]}

    lines = [f"{emoji} *{title}*", message] + [f"• *{k}* : {v}" for k, v in fields.items()]
    if link:
        lines.append(f"<{link}|Voir l'exécution GitHub Actions>" if kind == "slack" else link)
    text = "\n".join(lines)
    if kind == "slack":
        return {"text": text}
    return {"content": text[:2000], "text": text}  # webhook générique (Teams, Mattermost...)


def notify(cfg: Settings, title: str, message: str, severity: str = "info", fields: dict | None = None) -> int:
    """Envoie la notification à tous les webhooks configurés. Renvoie le nombre d'envois réussis."""
    level = logging.WARNING if severity in ("avertissement", "critique") else logging.INFO
    log.log(level, "[%s] %s : %s", severity, title, message)
    sent = 0
    for kind, url in webhook_targets(cfg):
        try:
            r = requests.post(url, json=build_payload(kind, title, message, severity, fields), timeout=10)
            r.raise_for_status()
            sent += 1
        except requests.RequestException as e:  # une notification ne doit jamais faire planter le pipeline
            log.error("Échec de l'envoi %s : %s", kind, e)
    return sent


def send_alert(cfg: Settings, message: str, severity: str = "critique", title: str | None = None,
               fields: dict | None = None) -> int:
    """Alerte (raccourci de `notify` avec une sévérité avertissement/critique)."""
    return notify(cfg, title or f"conso-hdf : alerte {severity}", message, severity, fields)
