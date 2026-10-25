import re
import time
import requests
import pandas as pd
from bs4 import BeautifulSoup
from urllib.parse import urlparse, urlunparse
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from tqdm import tqdm

# ---------------------------------------------------------------
# Shared headers
# ---------------------------------------------------------------
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/140.0.0.0 Safari/537.36"
    )
}

# ---------------------------------------------------------------
# Domain -> friendly site name
# ---------------------------------------------------------------
DOMAIN_TO_SITE = {
    "citizen.digital": "Citizen",
    "tv47.digital": "TV47",
    "the-star.co.ke": "Star",
    "tuko.co.ke": "Tuko",
    "standardmedia.co.ke": "Standard",
    "nation.africa": "Nation",
    "capitalfm.africa": "Capital FM",
    "k24.digital": "K24",
    "kbc.co.ke": "KBC",
}


def get_site_name(url: str):
    """Extract the domain from a URL and return the news site name."""
    domain = urlparse(url).netloc.lower()
    if domain.startswith("www."):
        domain = domain[4:]
    return DOMAIN_TO_SITE.get(domain)


# ---------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------
def clean_url(url: str) -> str:
    """Strip query string (utm_source etc.) and fragment."""
    p = urlparse(url)
    return urlunparse((p.scheme, p.netloc, p.path, "", "", ""))


def get_selenium_soup(url, wait=3):
    """Render JS-heavy pages with headless Chrome, return BeautifulSoup."""
    options = Options()
    options.add_argument("--headless=new")
    options.add_argument("--disable-gpu")
    options.add_argument("--no-sandbox")
    options.add_argument(f"user-agent={HEADERS['User-Agent']}")
    driver = webdriver.Chrome(options=options)
    try:
        driver.get(url)
        time.sleep(wait)
        return BeautifulSoup(driver.page_source, "html.parser")
    finally:
        driver.quit()


# ---------------------------------------------------------------
# Site-specific scrapers
# ---------------------------------------------------------------
def scrape_citizen(url):
    r = requests.get(url, headers=HEADERS, timeout=20)
    if r.status_code != 200:
        return []
    soup = BeautifulSoup(r.text, "html.parser")
    return [
        q.get_text(strip=True)
        for q in soup.find_all(
            "div",
            class_="article-content__paragraph_XUrcR js-article-paragraph",
        )
    ]


def scrape_tv47(url):
    r = requests.get(url, headers=HEADERS, timeout=20)
    if r.status_code != 200:
        return []
    soup = BeautifulSoup(r.text, "html.parser")
    return [p.get_text(strip=True) for p in soup.find_all("p", class_="wp-block-paragraph")]


def scrape_star(url):
    r = requests.get(url, headers=HEADERS, timeout=20)
    if r.status_code != 200:
        return []
    soup = BeautifulSoup(r.text, "html.parser")
    body = soup.find("div", class_="story-content overflow-hidden")
    return [p.get_text(strip=True) for p in body.find_all("p")] if body else []


def scrape_tuko(url):
    r = requests.get(url, headers=HEADERS, timeout=20)
    if r.status_code != 200:
        return []
    soup = BeautifulSoup(r.text, "html.parser")
    return [p.get_text(strip=True) for p in soup.find_all("p", class_="align-left")]


def scrape_standard(url):
    r = requests.get(url, headers=HEADERS, timeout=20)
    if r.status_code != 200:
        return []
    soup = BeautifulSoup(r.text, "html.parser")
    return [p.get_text(strip=True) for p in soup.find_all("p")]


def scrape_nation(url):
    soup = get_selenium_soup(url)
    return [p.get_text(strip=True) for p in soup.find_all("div", class_="paragraph-wrapper")]


def scrape_capital_fm(url):
    r = requests.get(url, headers=HEADERS, timeout=20)
    if r.status_code != 200:
        return []
    soup = BeautifulSoup(r.text, "html.parser")
    body = soup.find("div", class_="post-content")
    return [p.get_text(strip=True) for p in body.find_all("p")] if body else []


def scrape_k24(url):
    soup = get_selenium_soup(url)
    return [p.get_text(strip=True) for p in soup.find_all("p", class_="wp-block-paragraph")]


def scrape_kbc(url):
    r = requests.get(url, headers=HEADERS, timeout=20)
    if r.status_code != 200:
        return []
    soup = BeautifulSoup(r.text, "html.parser")
    body = soup.find("div", class_="entry-content rbct clearfix")
    return [p.get_text(strip=True) for p in body.find_all("p")] if body else []


# ---------------------------------------------------------------
# Dispatcher: site name -> scraper function
# ---------------------------------------------------------------
SITE_SCRAPERS = {
    "Citizen": scrape_citizen,
    "TV47": scrape_tv47,
    "Star": scrape_star,
    "Tuko": scrape_tuko,
    "Standard": scrape_standard,
    "Nation": scrape_nation,
    "Capital FM": scrape_capital_fm,
    "K24": scrape_k24,
    "KBC": scrape_kbc,
}


# ---------------------------------------------------------------
# Junk-URL filter (tag/category/author pages, etc.)
# ---------------------------------------------------------------
NON_ARTICLE_PATTERNS = re.compile(
    r"/(tag|category|author|radio|tv|wp-login\.php|wp-admin|search|page)/?",
    re.IGNORECASE,
)


def is_article_url(url: str) -> bool:
    path = urlparse(url).path
    return not NON_ARTICLE_PATTERNS.search(path)


# ---------------------------------------------------------------
# Main
# ---------------------------------------------------------------
def main():
    df = pd.read_csv("kenyamoja_news.csv")
    SELENIUM_SITES = {"Nation", "K24"}
    USE_SELENIUM = False   # False = default (non-Selenium sites), True = Selenium sites only
     # ---- split into two dataframes ----
    site_col = df["url"].astype(str).str.strip().map(clean_url).map(get_site_name)
    df_selenium = df[site_col.isin(SELENIUM_SITES)].reset_index(drop=True)
    df_requests = df[~site_col.isin(SELENIUM_SITES)].reset_index(drop=True)
    print(f"Requests-based: {len(df_requests)}   Selenium-based: {len(df_selenium)}")
    df=df_requests[43216:48618]
    print(df.shape)
    results = []
    skipped = []

    for _, row in tqdm(df.iterrows()):
        raw_url = str(row["url"]).strip()
        url = clean_url(raw_url)

        # 1. Identify the news site by domain
        site_name = get_site_name(url)

        if site_name is None:
            print(f"[SKIP] unknown site -> {url}")
            skipped.append({"url": url, "reason": "unknown site"})
            continue

        # 2. Look up the scraper for that site
        scraper = SITE_SCRAPERS.get(site_name)
        if scraper is None:
            print(f"[SKIP] no scraper for {site_name} -> {url}")
            skipped.append({"url": url, "reason": f"no scraper for {site_name}"})
            continue

        # 3. Filter out non-article pages
        if not is_article_url(url):
            print(f"[SKIP] non-article page -> {url}")
            skipped.append({"url": url, "reason": "non-article path"})
            continue

        # 4. Scrape
        print(f"[{site_name}] scraping -> {url}")
        try:
            paragraphs = scraper(url)
        except Exception as e:
            print(f"   ! error: {e}")
            paragraphs = []

        if not paragraphs:
            skipped.append({"url": url, "reason": "empty result"})

        results.append({
            "url": url,
            "website": site_name,
            "num_paragraphs": len(paragraphs),
            "article_text": "\n\n".join(paragraphs),
        })

        time.sleep(1)  # be polite

    # 5. Save
    pd.DataFrame(results).to_csv("kenyamoja_articles.csv", index=False)
    pd.DataFrame(skipped).to_csv("kenyamoja_skipped.csv", index=False)

    print(f"\nScraped: {len(results)}   Skipped: {len(skipped)}")
    print("Saved -> kenyamoja_articles.csv & kenyamoja_skipped.csv")


if __name__ == "__main__":
    main()