"""
Kenyamoja news scraper — resumable, auto-saving, connection-resilient version.

Key changes vs. the original script:
1. Saves every page's results to CSV immediately (append mode) instead of
   waiting until the very end. If the script dies, you never lose data.
2. Tracks progress in a small JSON checkpoint file (progress.json) so a
   restart picks up exactly where it left off — source + page number.
3. Wraps every network-touching step in retry logic with backoff. If Chrome
   itself crashes or the connection drops hard, it rebuilds the driver and
   keeps going instead of dying.
4. Same scraping logic/selectors as your original — nothing about *what*
   gets scraped has changed, only the reliability around it.
"""

import json
import os
import time

import pandas as pd
from selenium import webdriver
from selenium.common.exceptions import (
    TimeoutException,
    WebDriverException,
)
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from tqdm import tqdm

urls = [
    'https://www.kenyamoja.com/news/tv47',
    'https://www.kenyamoja.com/news/citizen',
    'https://www.kenyamoja.com/news/star',
    'https://www.kenyamoja.com/news/tuko',
    'https://www.kenyamoja.com/news/standard',
    'https://www.kenyamoja.com/news/nation',
    'https://www.kenyamoja.com/news/capital-fm',
    'https://www.kenyamoja.com/news/k24',
    'https://www.kenyamoja.com/news/kbc'
]

OUTPUT_FILE = "kenyamoja_news.csv"
PROGRESS_FILE = "progress.json"
MAX_PAGES = 200

# How many times to retry a single page before giving up on it and moving on
MAX_PAGE_RETRIES = 5
# How many times to try rebuilding the driver before aborting the whole run
MAX_DRIVER_RESTARTS = 10
PAGE_LOAD_TIMEOUT = 30
EXTRA_WAIT = 15
BETWEEN_PAGE_WAIT = 15


# ---------------------------------------------------------------------------
# Progress tracking (resume support)
# ---------------------------------------------------------------------------
def load_progress():
    if os.path.exists(PROGRESS_FILE):
        with open(PROGRESS_FILE, "r") as f:
            return json.load(f)
    return {}


def save_progress(progress):
    # Write to a temp file then replace, so a crash mid-write can't corrupt it
    tmp = PROGRESS_FILE + ".tmp"
    with open(tmp, "w") as f:
        json.dump(progress, f, indent=2)
    os.replace(tmp, PROGRESS_FILE)


# ---------------------------------------------------------------------------
# Incremental CSV saving
# ---------------------------------------------------------------------------
def append_articles(articles):
    if not articles:
        return
    df = pd.DataFrame(articles)
    df = df[["name", "date", "article_title", "url"]]
    file_exists = os.path.exists(OUTPUT_FILE)
    df.to_csv(
        OUTPUT_FILE,
        mode="a",
        header=not file_exists,
        index=False,
    )


# ---------------------------------------------------------------------------
# Driver management
# ---------------------------------------------------------------------------
def build_driver():
    options = Options()
    options.add_argument("--start-maximized")
    # A couple of flags that help with long unattended runs
    options.add_argument("--disable-gpu")
    options.add_argument("--disable-dev-shm-usage")
    driver = webdriver.Chrome(options=options)
    driver.set_page_load_timeout(PAGE_LOAD_TIMEOUT + 30)
    return driver


class DriverHolder:
    """Keeps a single live driver instance and knows how to replace itself."""

    def __init__(self):
        self.driver = build_driver()
        self.restarts = 0

    def restart(self):
        self.restarts += 1
        if self.restarts > MAX_DRIVER_RESTARTS:
            raise RuntimeError(
                f"Chrome driver failed to recover after {MAX_DRIVER_RESTARTS} restarts. "
                "Stopping — check your internet/VPN connection."
            )
        print(f"Restarting Chrome driver (attempt {self.restarts})...")
        try:
            self.driver.quit()
        except Exception:
            pass
        time.sleep(5)
        self.driver = build_driver()


# ---------------------------------------------------------------------------
# Scraping a single page, with retries
# ---------------------------------------------------------------------------
def scrape_page(holder, page_url):
    """Returns a list of article dicts, or None if the page genuinely has
    no articles (i.e. we've reached the end of that source)."""
    last_error = None
    for attempt in range(1, MAX_PAGE_RETRIES + 1):
        try:
            driver = holder.driver
            driver.get(page_url)
            WebDriverWait(driver, PAGE_LOAD_TIMEOUT).until(
                lambda d: d.execute_script("return document.readyState") == "complete"
            )
            time.sleep(EXTRA_WAIT)

            list_elements = driver.find_elements(
                By.CSS_SELECTOR, "li.news-item.video-item"
            )

            data = []
            for ls in list_elements:
                try:
                    date = ls.find_element(By.CSS_SELECTOR, "time").text.strip()
                    title_el = ls.find_element(By.CSS_SELECTOR, "a")
                    article_title = title_el.text.strip()
                    article_url = title_el.get_attribute("href")
                    data.append(
                        {
                            "date": date,
                            "article_title": article_title,
                            "url": article_url,
                        }
                    )
                except Exception:
                    # Skip a single malformed list item rather than failing the page
                    continue

            return data

        except (TimeoutException, WebDriverException) as e:
            last_error = e
            print(f"  [warn] attempt {attempt}/{MAX_PAGE_RETRIES} failed for "
                  f"{page_url}: {type(e).__name__}: {str(e)[:200]}")
            # Backoff before retrying
            time.sleep(min(10 * attempt, 60))
            # If the driver/session itself looks dead, rebuild it
            try:
                _ = holder.driver.title
            except Exception:
                holder.restart()

    print(f"  [error] giving up on {page_url} after {MAX_PAGE_RETRIES} attempts "
          f"({last_error}). Will retry this exact page on next run.")
    raise RuntimeError(f"Persistent failure on {page_url}") from last_error


def url_generator(url, pages=MAX_PAGES, start_page=0):
    for i in range(start_page, pages + 1):
        yield i, f"{url}?page={i}"


# ---------------------------------------------------------------------------
# Main run
# ---------------------------------------------------------------------------
def main():
    progress = load_progress()
    holder = DriverHolder()
    total_collected = sum(v.get("count", 0) for v in progress.values())

    try:
        for base_url in tqdm(urls, desc="Sources"):
            name = base_url.split("/news/")[1]
            src_progress = progress.get(name, {"next_page": 0, "done": False, "count": 0})

            if src_progress.get("done"):
                print(f"\n[skip] {name} already marked done, skipping.")
                continue

            print("\n====================================")
            print("SOURCE:", name)
            print("====================================")

            start_page = src_progress.get("next_page", 0)
            for page_num, page_url in url_generator(base_url, MAX_PAGES, start_page):
                print(f"\nScraping: {page_url}")

                try:
                    articles = scrape_page(holder, page_url)
                except RuntimeError:
                    # Couldn't get this page after all retries — save progress
                    # as-is (not advanced past this page) and stop the whole
                    # run so you can fix connectivity and resume later.
                    save_progress(progress)
                    print("\nStopping run. Re-run the script to resume from "
                          f"{name} page {page_num}.")
                    return

                if not articles:
                    print("No more articles. Moving to next source.")
                    src_progress["done"] = True
                    src_progress["next_page"] = page_num
                    progress[name] = src_progress
                    save_progress(progress)
                    break

                for article in articles:
                    article["name"] = name

                append_articles(articles)

                src_progress["count"] = src_progress.get("count", 0) + len(articles)
                src_progress["next_page"] = page_num + 1
                src_progress["done"] = False
                progress[name] = src_progress
                save_progress(progress)

                total_collected += len(articles)
                print("Articles found:", len(articles))
                print("Total collected so far:", total_collected)
                time.sleep(BETWEEN_PAGE_WAIT)

    finally:
        try:
            holder.driver.quit()
        except Exception:
            pass

    print("\nAll sources complete.")
    print("Data was saved incrementally to:", OUTPUT_FILE)


if __name__ == "__main__":
    main()