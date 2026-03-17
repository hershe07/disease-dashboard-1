"""layer1/utils/http.py -- shared HTTP fetch with retry + rate-limit backoff."""
import time
import requests
from typing import Any, Optional
from chronic_illness_monitor.settings import get_logger

logger = get_logger(__name__)

DEFAULT_TIMEOUT = 30
MAX_RETRIES     = 3
BACKOFF_BASE    = 2


def raw_get(url: str, timeout: int = DEFAULT_TIMEOUT, retries: int = MAX_RETRIES) -> Any:
    """
    GET a URL exactly as given -- no param encoding.
    Use this for APIs like WHO GHO where OData params ($top, $filter)
    must not have their $ signs encoded to %24.
    """
    attempt = 0
    while attempt <= retries:
        try:
            resp = requests.get(url, timeout=timeout)
            if resp.status_code == 429:
                time.sleep(BACKOFF_BASE ** attempt); attempt += 1; continue
            resp.raise_for_status()
            return resp.json()
        except requests.exceptions.Timeout:
            attempt += 1
        except requests.exceptions.ConnectionError as e:
            raise RuntimeError(f"Cannot reach {url}") from e
    raise RuntimeError(f"Exhausted retries for {url}")


def fetch(url: str, params: Optional[dict] = None,
          headers: Optional[dict] = None, timeout: int = DEFAULT_TIMEOUT,
          retries: int = MAX_RETRIES) -> Any:
    attempt = 0
    while attempt <= retries:
        try:
            resp = requests.get(url, params=params, headers=headers or {}, timeout=timeout)
            if resp.status_code == 429:
                wait = BACKOFF_BASE ** attempt
                logger.warning("Rate-limited %s -- waiting %ss", url, wait)
                time.sleep(wait); attempt += 1; continue
            if resp.status_code >= 500:
                wait = BACKOFF_BASE ** attempt
                logger.warning("Server error %s from %s -- retry in %ss",
                               resp.status_code, url, wait)
                time.sleep(wait); attempt += 1; continue
            resp.raise_for_status()
            return resp.json()
        except requests.exceptions.Timeout:
            logger.warning("Timeout %s (attempt %s/%s)", url, attempt+1, retries)
            attempt += 1; time.sleep(BACKOFF_BASE ** attempt)
        except requests.exceptions.ConnectionError as exc:
            logger.error("Connection error %s: %s", url, exc)
            raise RuntimeError(f"Cannot reach {url}") from exc
    raise RuntimeError(f"Exhausted {retries} retries for {url}")


def fetch_paginated(url: str, params: Optional[dict] = None,
                    headers: Optional[dict] = None,
                    page_param: str = "$offset", limit_param: str = "$limit",
                    page_size: int = 1000, max_records: Optional[int] = None) -> list[dict]:
    all_records: list[dict] = []
    offset = 0
    p = dict(params or {})
    p[limit_param] = page_size
    while True:
        p[page_param] = offset
        batch = fetch(url, params=p, headers=headers)
        if not batch:
            break
        all_records.extend(batch)
        offset += page_size
        if max_records and len(all_records) >= max_records:
            all_records = all_records[:max_records]; break
        if len(batch) < page_size:
            break
    logger.info("Fetched %s records from %s", len(all_records), url)
    return all_records
