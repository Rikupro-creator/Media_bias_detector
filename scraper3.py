import time
import random
import threading
import pandas as pd
from bs4 import BeautifulSoup
from urllib.parse import urlparse, urlunparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from selenium import webdriver
from selenium.webdriver.edge.options import Options as EdgeOptions
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from tqdm import tqdm

# ---------------------------------------------------------------
# This script handles ONLY the Selenium-rendered sites (Nation, K24).
# Assumes you already have the requests-based sites scraped separately.
# ---------------------------------------------------------------

USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/140.0.0.0 Safari/537.36 Edg/140.0.0.0"
)

DOMAIN_TO_SITE = {
    "nation.africa": "Nation",
    "k24.digital": "K24",
}

SELENIUM_WAIT_SELECTOR = {
    "Nation": "div.paragraph-wrapper",
    "K24": "p.wp-block-paragraph",
}

_thread_local = threading.local()


def get_driver():
    """One persistent headless Edge driver per worker thread."""
    if not hasattr(_thread_local, "driver"):
        options = EdgeOptions()
        options.add_argument("--headless=new")
        options.add_argument("--disable-gpu")
        options.add_argument("--no-sandbox")
        options.add_argument("--disable-dev-shm-usage")
        options.add_argument(f"user-agent={USER_AGENT}")
        # Don't block on full page load (ads, trackers, images) — just DOM ready
        options.page_load_strategy = "eager"
        prefs = {"profile.managed_default_content_settings.images": 2}
        options.add_experimental_option("prefs", prefs)
        # Selenium 4.6+ auto-resolves msedgedriver via Selenium Manager.
        # If that fails in your environment, pass an explicit path:
        #   from selenium.webdriver.edge.service import Service
        #   driver = webdriver.Edge(service=Service("/path/to/msedgedriver"), options=options)
        _thread_local.driver = webdriver.Edge(options=options)
        _thread_local.driver.set_page_load_timeout(25)
    return _thread_local.driver


def get_site_name(url: str):
    domain = urlparse(url).netloc.lower()
    if domain.startswith("www."):
        domain = domain[4:]
    return DOMAIN_TO_SITE.get(domain)


def clean_url(url: str) -> str:
    p = urlparse(url)
    return urlunparse((p.scheme, p.netloc, p.path, "", "", ""))


def scrape(url, site_name):
    driver = get_driver()
    driver.get(url)
    selector = SELENIUM_WAIT_SELECTOR[site_name]
    try:
        WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, selector))
        )
    except Exception:
        pass  # grab whatever rendered rather than failing outright

    soup = BeautifulSoup(driver.page_source, "html.parser")
    if site_name == "Nation":
        return [p.get_text(strip=True) for p in soup.find_all("div", class_="paragraph-wrapper")]
    if site_name == "K24":
        return [p.get_text(strip=True) for p in soup.find_all("p", class_="wp-block-paragraph")]
    return []


def process_row(raw_url):
    url = clean_url(str(raw_url).strip())
    site_name = get_site_name(url)
    if site_name is None:
        return {"url": url, "website": None, "num_paragraphs": 0,
                "article_text": "", "status": "skip:unknown_site"}
    try:
        time.sleep(random.uniform(0.1, 0.3))  # small politeness jitter
        paragraphs = scrape(url, site_name)
    except Exception as e:
        return {"url": url, "website": site_name, "num_paragraphs": 0,
                "article_text": "", "status": f"error:{e}"}

    status = "ok" if paragraphs else "skip:empty_result"
    return {"url": url, "website": site_name, "num_paragraphs": len(paragraphs),
            "article_text": "\n\n".join(paragraphs), "status": status}


def main(max_workers=6):
    df = pd.read_csv("kenyamoja_news.csv")
    site_col = df["url"].astype(str).str.strip().map(clean_url).map(get_site_name)
    df_selenium = df[site_col.notna()].reset_index(drop=True)
    urls = df_selenium["url"].tolist()
    print(f"Selenium-based rows to scrape: {len(urls)}")

    results = []
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {ex.submit(process_row, u): u for u in urls}
        for f in tqdm(as_completed(futures), total=len(futures)):
            results.append(f.result())

    results_df = pd.DataFrame(results)
    ok = results_df[results_df["status"] == "ok"]
    skipped = results_df[results_df["status"] != "ok"]

    ok.drop(columns=["status"]).to_csv("kenyamoja_selenium_articles.csv", index=False)
    skipped.to_csv("kenyamoja_selenium_skipped.csv", index=False)

    print(f"\nScraped: {len(ok)}   Skipped: {len(skipped)}")
    print("Saved -> kenyamoja_selenium_articles.csv & kenyamoja_selenium_skipped.csv")


if __name__ == "__main__":
    main(max_workers=6)