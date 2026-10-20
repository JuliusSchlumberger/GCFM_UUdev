#!/usr/bin/env python3
"""Download and unzip FathomDEM (or any Zenodo record) into a target folder.

FathomDEM v1-0 Americas (record 14523356) has restricted files, read with the
Zenodo share-link token from the access email
(https://zenodo.org/records/14523356?token=...): pass the token as --token or
the whole link as --share-url. The token goes on the record query and on every
file download; share links can expire.

Resumable per zip: a zip on disk is unpacked (skipping extracted members); a
missing zip with any tile in its degree range is skipped, since zips are deleted
only after full extraction; otherwise it is downloaded and unpacked. The range
comes from the name (tile n00e006.tif inside n00e000-n30e030_FathomDEM_v1-0.zip).

Usage:
    python download_fathomdem.py --share-url "https://zenodo.org/records/14523356?token=XXXX" --out <data_root>/FathomDEM
    python download_fathomdem.py --token XXXX --out <data_root>/FathomDEM --filter S60W090
    python download_fathomdem.py --token XXXX --out <data_root>/FathomDEM --keep-zip --no-unzip

The token can also be supplied via the ZENODO_TOKEN environment variable.
"""

import argparse
import hashlib
import os
import sys
import zipfile
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import requests

RECORD_ID = "14523356"  # FathomDEM v1-0 Americas
API_URL = "https://zenodo.org/api/records/{record_id}"
CHUNK = 1 << 20  # 1 MiB streaming chunks


def get_token(cli_token: str | None, share_url: str | None) -> str:
    """Resolve the share-link token from --share-url, --token, or ZENODO_TOKEN.

    Handles Outlook 'SafeLinks' wrappers (eur*.safelinks.protection.outlook.com),
    where the real Zenodo URL is percent-encoded inside the outer '?url=' param.
    """
    if share_url:
        parsed = urlparse(share_url)
        if "safelinks.protection.outlook.com" in parsed.netloc:
            inner = parse_qs(parsed.query).get("url", [None])[0]
            if not inner:
                sys.exit("SafeLinks URL detected but no inner 'url=' value found.")
            share_url = inner  # already decoded by parse_qs
            parsed = urlparse(share_url)
        url_token = parse_qs(parsed.query).get("token", [None])[0]
        if not url_token:
            sys.exit("Could not find a '?token=' value in the provided --share-url.")
        return url_token
    token = cli_token or os.environ.get("ZENODO_TOKEN")
    if not token:
        sys.exit(
            "No token provided. Pass --share-url (the full email link), --token "
            "(the string after '?token='), or set ZENODO_TOKEN. FathomDEM files "
            "are restricted and require the share-link token from your access email."
        )
    return token


def fetch_file_list(record_id: str, token: str) -> list[dict]:
    """Query the Zenodo REST API with the share-link token and return file entries."""
    resp = requests.get(
        API_URL.format(record_id=record_id),
        params={"token": token},
        timeout=60,
    )
    if resp.status_code == 403:
        sys.exit(
            "403 Forbidden. The share-link token was rejected -- it may have "
            "expired or be incomplete. Re-copy the full token from your email link."
        )
    if resp.status_code in (401, 404):
        sys.exit(
            f"{resp.status_code} from Zenodo. Check that the token is complete and "
            "the record id matches the one in your access email."
        )
    resp.raise_for_status()
    files = resp.json().get("files", [])
    if not files:
        sys.exit("No files returned -- the token may not grant file access yet.")
    return files


def md5_of(path: Path) -> str:
    """Return the hex MD5 digest of a file."""
    h = hashlib.md5()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(CHUNK), b""):
            h.update(block)
    return h.hexdigest()


def parse_tile_corner(name: str) -> tuple[int, int] | None:
    """Return the SW corner (lat, lon) of a token like 'n00e000' or 's10w090'.

    Latitude: n/s + 2 digits. Longitude: e/w + 3 digits. South and West are negative.
    Returns None if the token doesn't match.
    """
    import re

    m = re.search(r"([ns])(\d{2})([ew])(\d{3})", name.lower())
    if not m:
        return None
    ns, lat, ew, lon = m.group(1), int(m.group(2)), m.group(3), int(m.group(4))
    if ns == "s":
        lat = -lat
    if ew == "w":
        lon = -lon
    return lat, lon


def parse_zip_bbox(zip_key: str) -> tuple[int, int, int, int] | None:
    """Return (lat_min, lon_min, lat_max, lon_max) from a zip name.

    In 'n00e000-n30e030_FathomDEM_v1-0.zip' the two tokens are the SW and NE
    corners of the 30-degree block. Returns None without two parseable corners.
    """
    import re

    corners = re.findall(r"[ns]\d{2}[ew]\d{3}", zip_key.lower())
    if len(corners) < 2:
        return None
    sw = parse_tile_corner(corners[0])
    ne = parse_tile_corner(corners[1])
    if not sw or not ne:
        return None
    lat_min, lat_max = sorted((sw[0], ne[0]))
    lon_min, lon_max = sorted((sw[1], ne[1]))
    return lat_min, lon_min, lat_max, lon_max


def tifs_in_bbox(out_dir: Path, bbox: tuple[int, int, int, int]) -> list[Path]:
    """Return existing .tif files in out_dir whose SW corner falls inside bbox.

    bbox is (lat_min, lon_min, lat_max, lon_max); the NE corner is exclusive,
    matching the half-open convention of the 30-degree blocks (block
    n00e000-n30e030 owns tiles with corners in [0,30) x [0,30)).
    """
    lat_min, lon_min, lat_max, lon_max = bbox
    hits = []
    for tif in out_dir.glob("**/*.tif"):
        corner = parse_tile_corner(tif.name)
        if corner is None:
            continue
        lat, lon = corner
        if lat_min <= lat < lat_max and lon_min <= lon < lon_max:
            hits.append(tif)
    return hits


def download_file(entry: dict, out_dir: Path, token: str) -> Path:
    """Stream one file to disk, skipping if a checksum-valid copy already exists."""
    name = entry["key"]
    # Zenodo checksum format: "md5:abcdef..."
    checksum = entry.get("checksum", "")
    expected_md5 = checksum.split(":", 1)[1] if checksum.startswith("md5:") else None
    link = entry["links"]["self"]
    dest = out_dir / name

    if dest.exists() and expected_md5 and md5_of(dest) == expected_md5:
        print(f"  [skip] {name} already present and checksum matches")
        return dest

    print(f"  [get ] {name} ({entry.get('size', 0) / 1e9:.2f} GB)")
    with requests.get(link, params={"token": token}, stream=True, timeout=300) as r:
        r.raise_for_status()
        tmp = dest.with_suffix(dest.suffix + ".part")
        with open(tmp, "wb") as f:
            for chunk in r.iter_content(chunk_size=CHUNK):
                f.write(chunk)
        tmp.replace(dest)

    if expected_md5:
        actual = md5_of(dest)
        if actual != expected_md5:
            sys.exit(f"  Checksum mismatch for {name}: {actual} != {expected_md5}")
        print(f"  [ok  ] checksum verified for {name}")
    return dest


def unzip_file(zip_path: Path, out_dir: Path) -> list[str]:
    """Extract a zip, skipping members already on disk; return all member names."""
    if not zipfile.is_zipfile(zip_path):
        print(f"  [warn] {zip_path.name} is not a zip; leaving as-is")
        return []
    with zipfile.ZipFile(zip_path) as z:
        members = [n for n in z.namelist() if not n.endswith("/")]
        missing = [n for n in members if not (out_dir / n).exists()]
        if not missing:
            print(f"  [unzip] {zip_path.name}: all members already extracted")
        else:
            print(
                f"  [unzip] {zip_path.name}: extracting {len(missing)}/{len(members)} member(s)"
            )
            z.extractall(out_dir, members=missing)
    return members


def main() -> None:
    """Download and unpack the selected files of one Zenodo record."""
    p = argparse.ArgumentParser(description="Download + unzip FathomDEM from Zenodo.")
    p.add_argument(
        "--token", help="Zenodo share-link token (the string after '?token=')"
    )
    p.add_argument(
        "--share-url",
        help="Full share link from your access email; token is extracted from it",
    )
    p.add_argument(
        "--out", required=True, help="Target folder for downloads + extraction"
    )
    p.add_argument(
        "--record",
        default=RECORD_ID,
        help="Zenodo record id (default: FathomDEM Americas)",
    )
    p.add_argument(
        "--filter",
        default=None,
        help="Only download files whose name contains this substring (e.g. a tile range)",
    )
    p.add_argument(
        "--no-unzip", action="store_true", help="Download only, do not extract"
    )
    p.add_argument(
        "--keep-zip", action="store_true", help="Keep zip archives after extraction"
    )
    args = p.parse_args()

    token = get_token(args.token, args.share_url)
    out_dir = Path(args.out).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Querying Zenodo record {args.record} ...")
    files = fetch_file_list(args.record, token)
    if args.filter:
        files = [f for f in files if args.filter in f["key"]]
        if not files:
            sys.exit(f"No files matched filter '{args.filter}'.")
    print(f"{len(files)} file(s) to process. Output folder: {out_dir}\n")

    for entry in files:
        name = entry["key"]
        is_zip = name.lower().endswith(".zip")
        dest = out_dir / name

        if not is_zip or args.no_unzip:
            download_file(entry, out_dir, token)
            continue

        # Completes an interrupted unpack.
        if dest.exists():
            print(f"  [have] {name} present on disk")
            unzip_file(dest, out_dir)
            if not args.keep_zip:
                dest.unlink()
                print(f"  [rm  ] removed {name}")
            continue

        # Zips are deleted only after full extraction, so any tile in the
        # zip's range proves it was already unpacked.
        bbox = parse_zip_bbox(name)
        if bbox is not None and tifs_in_bbox(out_dir, bbox):
            print(f"  [skip] {name}: tiles in its range already extracted")
            continue

        path = download_file(entry, out_dir, token)
        unzip_file(path, out_dir)
        if not args.keep_zip:
            path.unlink()
            print(f"  [rm  ] removed {path.name}")

    print("\nDone.")


if __name__ == "__main__":
    main()
