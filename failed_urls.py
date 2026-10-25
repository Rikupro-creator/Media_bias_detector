import pandas as pd
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.edge.options import Options
from selenium.common.exceptions import TimeoutException, WebDriverException
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import threading
import time

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
MAIN_CSV = 'kenyamoja_news_with_articles.csv'
FAILED_CSV = 'failed_links.csv'          # your existing file with the failed urls
STILL_FAILED_CSV = 'still_failed_after_selenium.csv'

MAX_WORKERS = 4
PAGE_LOAD_TIMEOUT = 20

# ---------------------------------------------------------------------------
# STEP 1 — Load the failed urls straight from the failed CSV
# ---------------------------------------------------------------------------
failed_df = pd.read_csv(FAILED_CSV)

# if the file has an 'error' column with __FAILED__ / __SKIPPED__ tags, filter to genuine failures
if 'error' in failed_df.columns:
    retry_links = failed_df.loc[
        failed_df['error'].astype(str).str.startswith('__FAILED__'), 'url'
    ].tolist()
else:
    # if it's just a plain list of urls
    retry_links = failed_df['url'].tolist()

print(f"Retrying {len(retry_links)} failed links with Selenium...")

# ---------------------------------------------------------------------------
# STEP 2 — Selenium setup (one driver per worker thread)
# ---------------------------------------------------------------------------
def make_driver():
    options = Options()
    options.add_argument("--headless=new")
    options.add_argument("--start-maximized")
    options.add_argument("--disable-gpu")
    options.add_argument("--no-sandbox")
    driver = webdriver.Edge(options=options)
    driver.set_page_load_timeout(PAGE_LOAD_TIMEOUT)
    return driver

thread_local_driver = {}

def get_driver():
    thread_id = threading.get_ident()
    if thread_id not in thread_local_driver:
        thread_local_driver[thread_id] = make_driver()
    return thread_local_driver[thread_id]

def close_all_drivers():
    for driver in thread_local_driver.values():
        try:
            driver.quit()
        except Exception:
            pass

def get_article_selenium(link):
    try:
        driver = get_driver()
        driver.get(link)
        time.sleep(2)
        paragraphs = driver.find_elements(By.TAG_NAME, "p")
        paragraph_text = [p.text.strip() for p in paragraphs if p.text.strip()]
        article = "\n\n".join(paragraph_text)
        return link, article if article else None
    except (TimeoutException, WebDriverException, Exception):
        return link, None

# ---------------------------------------------------------------------------
# STEP 3 — Run Selenium only on the failed links
# ---------------------------------------------------------------------------
results = {}
try:
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(get_article_selenium, link): link for link in retry_links}
        for future in tqdm(as_completed(futures), total=len(futures)):
            link, article = future.result()
            results[link] = article
finally:
    close_all_drivers()

# ---------------------------------------------------------------------------
# STEP 4 — Merge recovered articles back into the main CSV
# ---------------------------------------------------------------------------
Kenyamoja = pd.read_csv(MAIN_CSV)

succeeded = {link: art for link, art in results.items() if art}
still_failed = [link for link, art in results.items() if not art]

mask = Kenyamoja['url'].isin(succeeded)
Kenyamoja.loc[mask, 'article'] = Kenyamoja.loc[mask, 'url'].map(succeeded)

Kenyamoja.to_csv(MAIN_CSV, index=False)

print(f"\nRecovered: {len(succeeded)}")
print(f"Still failed: {len(still_failed)}")

if still_failed:
    pd.DataFrame({'url': still_failed}).to_csv(STILL_FAILED_CSV, index=False)
    print(f"Saved remaining failures to {STILL_FAILED_CSV}")
else:
    print("All failed links recovered!")