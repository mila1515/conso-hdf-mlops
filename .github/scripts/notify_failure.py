"""Alerte Discord / Slack quand un workflow GitHub Actions échoue.

Bibliothèque standard uniquement : fonctionne même si l'installation des dépendances a échoué.
Variables : DISCORD_WEBHOOK_URL, SLACK_WEBHOOK_URL, ALERT_WEBHOOK_URL (optionnelles), TITLE, RUN_URL.
"""

import json
import os
import urllib.request

title = os.getenv("TITLE", "conso-hdf : échec d'un workflow")
run_url = os.getenv("RUN_URL", "")
message = f"Le workflow a échoué. Détails : {run_url}"

payloads = {
    "discord": {"username": "conso-hdf",
                "embeds": [{"title": f"🔴 {title}", "description": message, "url": run_url, "color": 0xEF4444}]},
    "slack": {"text": f"🔴 *{title}*\n<{run_url}|Voir l'exécution GitHub Actions>"},
}

targets = []
if os.getenv("DISCORD_WEBHOOK_URL"):
    targets.append(("discord", os.environ["DISCORD_WEBHOOK_URL"]))
if os.getenv("SLACK_WEBHOOK_URL"):
    targets.append(("slack", os.environ["SLACK_WEBHOOK_URL"]))
legacy = os.getenv("ALERT_WEBHOOK_URL")
if legacy and legacy not in {u for _, u in targets}:
    targets.append(("slack" if "hooks.slack.com" in legacy else "discord", legacy))

if not targets:
    print("Aucun webhook configuré : pas d'alerte envoyée.")
for kind, url in targets:
    req = urllib.request.Request(url, data=json.dumps(payloads[kind]).encode(),
                                 headers={"Content-Type": "application/json", "User-Agent": "conso-hdf-mlops"})
    try:
        urllib.request.urlopen(req, timeout=10)
        print(f"Alerte {kind} envoyée")
    except Exception as e:  # noqa: BLE001
        print(f"Échec de l'alerte {kind} : {e}")
