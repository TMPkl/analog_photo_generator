"""Download images listed in a CSV (RISE dataset helper).

The script attempts to detect a column containing HTTP/HTTPS links (by default it
looks for a column named 'URL' or any column containing 'http' in its values).
You can override the column name with `--column`.

Usage examples:
  python RISE_dataset_download.py --csv media/RAISE_4k.csv --out data/RISE --workers 8
  python RISE_dataset_download.py --csv media/RAISE_4k.csv --column File
"""

from pathlib import Path
import argparse
import csv
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from tqdm import tqdm
import pandas as pd


def make_session(retries=3, backoff_factor=0.5, status_forcelist=(429, 500, 502, 503, 504)):
    session = requests.Session()
    retry = Retry(total=retries, read=retries, connect=retries, backoff_factor=backoff_factor,
                  status_forcelist=status_forcelist, allowed_methods=frozenset(['GET', 'HEAD']))
    adapter = HTTPAdapter(max_retries=retry)
    session.mount('http://', adapter)
    session.mount('https://', adapter)
    return session


def detect_url_column(df: pd.DataFrame):
    # Prefer common names first
    for name in ("URL", "Url", "url", "File", "file", "Link", "link"):
        if name in df.columns:
            sample = df[name].dropna().astype(str)
            if sample.str.contains(r'^https?://', regex=True).any():
                return name

    # Fallback: look for any column that contains http(s) in at least one row
    for col in df.columns:
        sample = df[col].dropna().astype(str)
        if sample.str.contains(r'^https?://', regex=True).any():
            return col

    return None


def download_one(session, url, out_dir: Path, timeout=30):
    try:
        if not isinstance(url, str) or not url.startswith('http'):
            return (url, False, 'invalid url')

        filename = url.split('/')[-1].split('?')[0]
        if not filename:
            return (url, False, 'empty filename')

        out_path = out_dir / filename
        if out_path.exists() and out_path.stat().st_size > 0:
            return (url, True, 'exists')

        resp = session.get(url, timeout=timeout)
        resp.raise_for_status()
        with open(out_path, 'wb') as f:
            f.write(resp.content)
        return (url, True, 'downloaded')
    except Exception as e:
        return (url, False, str(e))


def main():
    parser = argparse.ArgumentParser(description='Download images from RISE CSV list')
    parser.add_argument('--csv', '-c', type=Path, default=Path('media/RAISE_4k.csv'))
    parser.add_argument('--out', '-o', type=Path, default=Path('data/RISE'))
    parser.add_argument('--column', type=str, default=None, help='CSV column that contains the URLs')
    parser.add_argument('--workers', type=int, default=4, help='Number of concurrent download workers')
    parser.add_argument('--timeout', type=int, default=30, help='HTTP request timeout in seconds')
    parser.add_argument('--retries', type=int, default=3, help='HTTP retry attempts')
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')

    csv_path: Path = args.csv
    out_dir: Path = args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    if not csv_path.exists():
        logging.error('CSV file not found: %s', csv_path)
        return

    df = pd.read_csv(csv_path)

    col = args.column or detect_url_column(df)
    if col is None:
        logging.error('Could not detect a column containing URLs. Use --column to specify.')
        logging.error('Available columns: %s', ', '.join(df.columns.astype(str)))
        return

    urls = df[col].dropna().astype(str).tolist()
    if not urls:
        logging.error('No URLs found in column: %s', col)
        return

    session = make_session(retries=args.retries)

    results = []
    logging.info('Starting downloads: %d items, %d workers', len(urls), args.workers)

    with ThreadPoolExecutor(max_workers=args.workers) as exe:
        futures = {exe.submit(download_one, session, url, out_dir, args.timeout): url for url in urls}
        for f in tqdm(as_completed(futures), total=len(futures)):
            url = futures[f]
            try:
                res = f.result()
            except Exception as e:
                res = (url, False, str(e))
            results.append(res)

    success = sum(1 for r in results if r[1])
    failed = [r for r in results if not r[1]]

    logging.info('Done. Success: %d  Failed: %d', success, len(failed))
    if failed:
        logging.info('Examples of failures:')
        for url, ok, reason in failed[:10]:
            logging.info('  %s -> %s', url, reason)


if __name__ == '__main__':
    main()