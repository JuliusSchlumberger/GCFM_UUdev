#!/usr/bin/env python3
"""
Download and unzip FathomDEM (or any Zenodo record) into a target folder.

FathomDEM v1-0 Americas (record 14523356) has RESTRICTED files. This version uses
a Zenodo SHARE LINK token -- the kind you receive by email after requesting access.
The email link looks like:

    https://zenodo.org/records/14523356?token=eyJhbGciOiJ...

Copy the long string after "?token=" and pass it as --token, OR paste the whole
URL as --share-url and the script will extract the token for you.

A share-link token is carried on BOTH the record query and every file download.
These links can expire, so download in a reasonable timeframe after receiving it.

RESUMABILITY
The script avoids redoing work, using three cases per zip:
  1. the zip is still on disk      -> unpack it, skipping members already extracted
                                       (completes an interrupted unpack);
  2. the zip is gone but a TIF in its degree-range exists -> the zip was already
     fully unpacked (zips are deleted only after success), so skip;
  3. neither                       -> download and unpack.
Tile/zip names follow the FathomDEM convention (e.g. tile n00e006.tif inside zip
n00e000-n30e030_FathomDEM_v1-0.zip), so a zip's range is read straight from its name.

Usage:
    python download_fathomdem.py --share-url "https://zenodo.org/records/14523356?token=XXXX" --out D:/data/fathomdem
    python download_fathomdem.py --token XXXX --out D:/data/fathomdem --filter S60W090
    python download_fathomdem.py --token XXXX --out D:/data/fathomdem --keep-zip --no-unzip

The token can also be supplied via the ZENODO_TOKEN environment variable.
"""

import argparse
import hashlib
import os
import sys
import zipfile
from pathlib import Path
from urllib.parse import urlparse, parse_qs

import requests  # pip install requests

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
        # Unwrap Outlook SafeLinks: the genuine URL lives in the 'url' parameter.
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
    h = hashlib.md5()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(CHUNK), b""):
            h.update(block)
    return h.hexdigest()


# --- resumability helpers ---------------------------------------------------


def parse_tile_corner(name: str) -> tuple[int, int] | None:
    """Parse a FathomDEM tile/corner token like 'n00e000' or 's10w090' -> (lat, lon).

    Latitude: n/s + 2 digits. Longitude: e/w + 3 digits. South and West are negative.
    Returns the SW-corner integer degrees, or None if the token doesn't match.
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
    """From 'n00e000-n30e030_FathomDEM_v1-0.zip' derive (lat_min, lon_min, lat_max, lon_max).

    The two corner tokens are the SW and NE corners of the 30-degree block.
    Returns None if the name doesn't contain two parseable corners.
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


# ----------------------------------------------------------------------------


def download_file(entry: dict, out_dir: Path, token: str) -> Path:
    """Stream one file to disk, skipping if a checksum-valid copy already exists."""
    name = entry["key"]
    # Zenodo checksum looks like "md5:abcdef..."
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
    """Extract a zip, skipping members already on disk. Returns all member names."""
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

        # Non-zip files (or --no-unzip): just download with the usual checksum skip.
        if not is_zip or args.no_unzip:
            download_file(entry, out_dir, token)
            continue

        # Case 1: the zip is already on disk -> unpack it (skipping members that
        # already exist, so an interrupted previous unpack is completed).
        if dest.exists():
            print(f"  [have] {name} present on disk")
            unzip_file(dest, out_dir)
            if not args.keep_zip:
                dest.unlink()
                print(f"  [rm  ] removed {name}")
            continue

        # Case 2: zip absent, but an in-range TIF exists. Because the zip is only
        # deleted AFTER a successful full extraction, the presence of any tile in
        # the zip's degree-range proves the whole zip was already unpacked. Skip.
        bbox = parse_zip_bbox(name)
        if bbox is not None and tifs_in_bbox(out_dir, bbox):
            print(f"  [skip] {name}: tiles in its range already extracted")
            continue

        # Case 3: nothing present -> download and unpack.
        path = download_file(entry, out_dir, token)
        unzip_file(path, out_dir)
        if not args.keep_zip:
            path.unlink()
            print(f"  [rm  ] removed {path.name}")

    print("\nDone.")


if __name__ == "__main__":
    main()
