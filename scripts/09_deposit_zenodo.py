#!/usr/bin/env python
"""Deposit the two Zenodo records from ``zenodo/METADATA.md``.

``METADATA.md`` stays the single source of truth: this script parses the
paste-ready blocks out of it rather than duplicating them. If the parse stops
matching, it fails loudly instead of depositing something half-filled.

    python scripts/09_deposit_zenodo.py --dry-run     # no token needed
    python scripts/09_deposit_zenodo.py --create      # drafts + files + metadata
    python scripts/09_deposit_zenodo.py --publish     # mint both DOIs, cross-link

Token: read from ``~/.zenodo_token`` or ``$ZENODO_TOKEN``. Needs the
``deposit:write`` scope, plus ``deposit:actions`` for ``--publish``.

Nothing here is reversible after ``--publish``: Zenodo mints a DOI and a
published record cannot be deleted, only superseded. ``--create`` leaves drafts,
which can be deleted from the web UI.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

import requests

PROJECT_ROOT = Path(__file__).resolve().parent.parent
METADATA = PROJECT_ROOT / "zenodo" / "METADATA.md"
STATE = PROJECT_ROOT / "zenodo" / "deposits.json"
API = os.environ.get("ZENODO_API", "https://zenodo.org/api")

# Prose in METADATA.md -> Zenodo license vocabulary id.
# Verified against https://zenodo.org/api/vocabularies/licenses
LICENSE_IDS = {
    "MIT License": "mit",
    "Creative Commons Attribution 4.0 International": "cc-by-4.0",
    "Creative Commons Attribution Share Alike 4.0 International": "cc-by-sa-4.0",
}
UPLOAD_TYPES = {"Software": "software", "Publication": "publication"}
# Zenodo requires a subtype when upload_type is 'publication'.
PUBLICATION_TYPES = {"Report": "report", "Preprint": "preprint", "Article": "article"}
ACCESS_RIGHTS = {"Open Access": "open", "Restricted": "restricted", "Closed": "closed"}
LANGUAGES = {"English": "eng", "Thai": "tha"}

RECORD_FILES = {
    1: "EngramQwenASR-software-v1.0.0.zip",
    2: "EngramQwenASR-technical-reports-v1.0.0.zip",
}


def token() -> str:
    if os.environ.get("ZENODO_TOKEN"):
        return os.environ["ZENODO_TOKEN"].strip()
    path = Path.home() / ".zenodo_token"
    if path.is_file():
        return path.read_text().strip()
    raise SystemExit(
        "No token. Set $ZENODO_TOKEN or write one to ~/.zenodo_token\n"
        "Zenodo -> Settings -> Personal access tokens -> New token, scopes "
        "deposit:write + deposit:actions."
    )


def _field(section: str, label: str) -> str:
    m = re.search(rf"^\*\*{re.escape(label)}:\*\*\s*(.+?)\s*$", section, re.M)
    if not m:
        raise ValueError(f"METADATA.md: no '**{label}:**' line")
    return m.group(1)


def _block(section: str, label: str) -> str:
    """The first fenced block after a '**Label**' heading (trailing prose allowed)."""
    m = re.search(rf"^\*\*{re.escape(label)}\*\*[^\n]*\n+```[a-z]*\n(.*?)\n```\s*$",
                  section, re.S | re.M)
    if not m:
        raise ValueError(f"METADATA.md: no fenced block under '**{label}**'")
    return m.group(1)


def parse_records() -> dict[int, dict]:
    text = METADATA.read_text()
    parts = re.split(r"^## Record (\d+) — ", text, flags=re.M)
    if len(parts) < 5:
        raise ValueError("METADATA.md: expected two '## Record N — ' sections")
    records = {}
    for num, body in zip(parts[1::2], parts[2::2]):
        num = int(num)
        licence = _field(body, "Licence")
        upload = _field(body, "Upload type")
        upload, _, subtype = (part.strip() for part in upload.partition("→"))
        access = _field(body, "Access right")
        language = _field(body, "Language").split("—")[0].strip()

        creators = []
        for line in _block(body, "Creators").splitlines():
            line = line.strip()
            if not line:
                continue
            if line.upper().startswith("ORCID:"):
                creators[-1]["orcid"] = line.split(":", 1)[1].strip()
            else:
                creators.append({"name": line})
        if not creators:
            raise ValueError(f"record {num}: no creators parsed")
        if "orcid" not in creators[-1]:
            raise ValueError(f"record {num}: creator {creators[-1]['name']!r} has no ORCID")

        try:
            meta = {
                "title": _block(body, "Title"),
                "upload_type": UPLOAD_TYPES[upload],
                "description": _block(body, "Description"),
                "creators": creators,
                "access_right": ACCESS_RIGHTS[access],
                "license": LICENSE_IDS[licence],
                "version": _field(body, "Version"),
                "language": LANGUAGES[language],
                "keywords": [k.strip() for k in _block(body, "Keywords").splitlines() if k.strip()],
            }
        except KeyError as exc:
            raise ValueError(f"record {num}: unmapped value {exc}") from exc

        if meta["upload_type"] == "publication":
            subtype = subtype.split("(")[0].strip()
            try:
                meta["publication_type"] = PUBLICATION_TYPES[subtype]
            except KeyError:
                raise ValueError(f"record {num}: unmapped publication subtype {subtype!r}") from None

        if num == 1:
            meta["publication_date"] = meta.get("publication_date") or "2026-09-24"
        records[num] = meta
    return records


def check_files(records: dict[int, dict]) -> None:
    for num, name in RECORD_FILES.items():
        path = PROJECT_ROOT / "zenodo" / name
        if not path.is_file():
            raise FileNotFoundError(f"{path} — run ./zenodo/build_archives.sh first")
        records[num]["_file"] = path


def _headers(tok: str) -> dict:
    return {"Authorization": f"Bearer {tok}"}


def create(tok: str, records: dict[int, dict]) -> dict:
    state = json.loads(STATE.read_text()) if STATE.is_file() else {}
    for num, meta in records.items():
        key = str(num)
        if key in state:
            print(f"record {num}: draft {state[key]['id']} already exists — skipping create")
            continue
        meta = {k: v for k, v in meta.items() if not k.startswith("_")}
        r = requests.post(f"{API}/deposit/depositions", json={}, headers=_headers(tok), timeout=120)
        r.raise_for_status()
        dep = r.json()
        dep_id = dep["id"]

        path = records[num]["_file"]
        with path.open("rb") as fh:
            up = requests.put(f"{dep['links']['bucket']}/{path.name}", data=fh,
                              headers=_headers(tok), timeout=3600)
        up.raise_for_status()
        print(f"record {num}: uploaded {path.name} ({path.stat().st_size/2**20:.1f} MiB)")

        r = requests.put(f"{API}/deposit/depositions/{dep_id}", json={"metadata": meta},
                         headers=_headers(tok), timeout=120)
        if r.status_code >= 400:
            raise SystemExit(f"record {num}: metadata rejected ({r.status_code})\n{r.text}")
        state[key] = {"id": dep_id, "file": path.name, "html": dep["links"]["html"]}
        print(f"record {num}: metadata set — review at {dep['links']['html']}")
    STATE.write_text(json.dumps(state, indent=2) + "\n")
    print(f"\ndrafts recorded in {STATE.relative_to(PROJECT_ROOT)}")
    print("Review both in the web UI, then re-run with --publish.")
    return state


def publish(tok: str, records: dict[int, dict]) -> None:
    if not STATE.is_file():
        raise SystemExit("no drafts — run --create first")
    state = json.loads(STATE.read_text())

    for num in sorted(records):
        key = str(num)
        if state[key].get("doi"):
            print(f"record {num}: already published as {state[key]['doi']}")
            continue

        # Cross-link to the other record before publishing this one, if its DOI exists.
        other = "2" if key == "1" else "1"
        other_doi = state.get(other, {}).get("doi")
        if other_doi:
            meta = {k: v for k, v in records[num].items() if not k.startswith("_")}
            meta["related_identifiers"] = [{
                "relation": "is supplemented by" if num == 1 else "is supplement to",
                "identifier": other_doi,
                "resource_type": "publication" if other == "2" else "software",
            }]
            r = requests.put(f"{API}/deposit/depositions/{state[key]['id']}",
                             json={"metadata": meta}, headers=_headers(tok), timeout=120)
            if r.status_code >= 400:
                raise SystemExit(f"record {num}: cross-link rejected\n{r.text}")
            print(f"record {num}: linked to {other_doi}")

        r = requests.post(f"{API}/deposit/depositions/{state[key]['id']}/actions/publish",
                          headers=_headers(tok), timeout=600)
        if r.status_code >= 400:
            raise SystemExit(f"record {num}: publish failed\n{r.text}")
        doi = r.json()["doi"]
        state[key]["doi"] = doi
        state[key]["record_url"] = r.json()["links"]["record_html"]
        STATE.write_text(json.dumps(state, indent=2) + "\n")
        print(f"record {num}: PUBLISHED — {doi}")

    print("\nNow add both DOIs to CITATION.cff and zenodo/METADATA.md, then commit.")


def dry_run(records: dict[int, dict]) -> None:
    check_files(records)
    for num, meta in records.items():
        clean = {k: v for k, v in meta.items() if not k.startswith("_")}
        print(f"=== Record {num} — {RECORD_FILES[num]} ===")
        print(json.dumps(clean, indent=2, ensure_ascii=False))
        print()
    print("dry run OK: both records parsed, both archives present")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    g = p.add_mutually_exclusive_group()
    g.add_argument("--dry-run", action="store_true", help="parse METADATA.md and print (no token)")
    g.add_argument("--create", action="store_true", help="create drafts, upload, set metadata")
    g.add_argument("--publish", action="store_true", help="mint both DOIs and cross-link")
    g.add_argument("--status", action="store_true", help="show recorded drafts and DOIs")
    args = p.parse_args()

    if args.status:
        print(STATE.read_text() if STATE.is_file() else "no deposits.json yet")
        return
    records = parse_records()
    if args.create:
        check_files(records)
        create(token(), records)
    elif args.publish:
        publish(token(), records)
    else:
        dry_run(records)


if __name__ == "__main__":
    main()
