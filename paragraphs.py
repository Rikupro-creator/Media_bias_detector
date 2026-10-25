import re
import threading
import pandas as pd
import requests

from bs4 import BeautifulSoup
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from urllib.parse import urlparse
from tqdm import tqdm


# ============================================================
# FILES
# ============================================================

INPUT_CSV = 'kenyamoja_news.csv'

OUTPUT_FILE = 'kenyamoja_news_with_articles.parquet'

FAILED_FILE = 'failed_links.csv'
STILL_FAILED_FILE = 'still_failed_links.csv'


# ============================================================
# SETTINGS
# ============================================================

MAX_WORKERS = 20
MAX_PER_DOMAIN = 3
RETRY_WORKERS = 5


# ============================================================
# HEADERS
# ============================================================

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0 Safari/537.36"
    )
}


# ============================================================
# URL FILTER
# ============================================================

NON_ARTICLE_PATTERNS = [
    r'/category/',
    r'/tag/',
    r'/author/',
    r'/page/\d+',
    r'/wp-json/',
]


# ============================================================
# DOMAIN SELECTORS
# ============================================================

DOMAIN_SELECTORS = {
    "www.standardmedia.co.ke": None,
    "www.tuko.co.ke": None,
    "www.the-star.co.ke": None,
    "k24.digital": None,
    "nation.africa": None,
    "www.capitalfm.co.ke": None,
    "citizen.digital": None,
    "www.citizen.digital": None,
    "www.tv47.digital": None,
    "www.kbc.co.ke": None,
}


# ============================================================
# GENERIC ARTICLE SELECTORS
# ============================================================

GENERIC_SELECTORS = [
    "article",
    "div.article-body",
    "div.article-content",
    "div.entry-content",
    "div.post-content",
    "div.story-body",
    "div.content-body",
    "div[itemprop='articleBody']",
    "div.single-post-content",
    "div.td-post-content",
]


# ============================================================
# FIND ARTICLE CONTAINER
# ============================================================

def get_article_container(soup, domain):

    override = DOMAIN_SELECTORS.get(domain)

    if override:
        container = soup.select_one(override)

        if container and len(container.get_text(strip=True)) > 200:
            return container

    for selector in GENERIC_SELECTORS:

        container = soup.select_one(selector)

        if container and len(container.get_text(strip=True)) > 200:
            return container

    return soup.body


# ============================================================
# CHECK WHETHER URL LOOKS LIKE AN ARTICLE
# ============================================================

def looks_like_article(url):

    return not any(
        re.search(pattern, url)
        for pattern in NON_ARTICLE_PATTERNS
    )


# ============================================================
# CREATE REQUEST SESSION
# ============================================================

def make_session():

    session = requests.Session()

    session.headers.update(HEADERS)

    retries = Retry(
        total=3,
        backoff_factor=1,
        status_forcelist=[
            429,
            500,
            502,
            503,
            504
        ],
        connect=3,
    )

    adapter = HTTPAdapter(
        max_retries=retries,
        pool_maxsize=50
    )

    session.mount("https://", adapter)
    session.mount("http://", adapter)

    return session


session = make_session()


# ============================================================
# LIMIT REQUESTS PER DOMAIN
# ============================================================

domain_semaphores = defaultdict(
    lambda: threading.Semaphore(MAX_PER_DOMAIN)
)


# ============================================================
# SCRAPE ONE ARTICLE
# ============================================================

def get_article(link):

    # Skip obvious non-article URLs
    if not looks_like_article(link):

        return (
            link,
            "__SKIPPED__: not an article URL "
            "(category/tag/listing page)"
        )

    domain = urlparse(link).netloc

    sem = domain_semaphores[domain]

    with sem:

        try:

            # ------------------------------------------------
            # Request page
            # ------------------------------------------------

            resp = session.get(
                link,
                timeout=(10, 20)
            )

            resp.raise_for_status()


            # ------------------------------------------------
            # Parse HTML
            # ------------------------------------------------

            soup = BeautifulSoup(
                resp.text,
                "html.parser"
            )


            # ------------------------------------------------
            # Find article container
            # ------------------------------------------------

            container = get_article_container(
                soup,
                domain
            )

            if container is None:

                return (
                    link,
                    "__FAILED__: "
                    "no article container found"
                )


            # ------------------------------------------------
            # Extract paragraphs
            # ------------------------------------------------

            paragraphs = container.find_all("p")

            paragraph_text = [
                p.get_text(strip=True)
                for p in paragraphs
                if p.get_text(strip=True)
            ]


            # ------------------------------------------------
            # Combine paragraphs
            # ------------------------------------------------

            article = "\n".join(
                paragraph_text
            )


            if not article:

                return (
                    link,
                    "__FAILED__: "
                    "no paragraph text found"
                )


            return link, article


        except Exception as e:

            return (
                link,
                f"__FAILED__: {e}"
            )


# ============================================================
# RUN SCRAPING
# ============================================================

def run_pass(links, max_workers):

    results = {}

    with ThreadPoolExecutor(
        max_workers=max_workers
    ) as executor:

        futures = {
            executor.submit(
                get_article,
                link
            ): link

            for link in links
        }


        for future in tqdm(
            as_completed(futures),
            total=len(futures)
        ):

            link, article = future.result()

            results[link] = article


    return results


# ============================================================
# LOAD ORIGINAL DATASET
# ============================================================

Kenyamoja = pd.read_csv(
    INPUT_CSV
)


print(
    f"Scraping {len(Kenyamoja)} links..."
)


# ============================================================
# FIRST SCRAPING PASS
# ============================================================

results = run_pass(
    Kenyamoja['url'].tolist(),
    MAX_WORKERS
)


# ============================================================
# IDENTIFY FAILED / SKIPPED LINKS
# ============================================================

failed_links = [
    (link, msg)

    for link, msg in results.items()

    if isinstance(msg, str)

    and (
        msg.startswith("__FAILED__")
        or
        msg.startswith("__SKIPPED__")
    )
]


# ============================================================
# REPLACE FAILED RESULTS WITH NONE
# ============================================================

clean_results = {

    link: (
        None

        if isinstance(msg, str)

        and (
            msg.startswith("__FAILED__")
            or
            msg.startswith("__SKIPPED__")
        )

        else msg
    )

    for link, msg in results.items()
}


# ============================================================
# ADD ARTICLES TO DATAFRAME
# ============================================================

Kenyamoja['article'] = (
    Kenyamoja['url']
    .map(clean_results)
)


# ============================================================
# SAVE MAIN DATASET AS PARQUET
# ============================================================

Kenyamoja.to_parquet(
    OUTPUT_FILE,
    index=False,
    engine='pyarrow'
)


print(
    f"\nDataset saved to: {OUTPUT_FILE}"
)


# ============================================================
# SAVE FAILED LINKS
# ============================================================

if failed_links:

    pd.DataFrame(
        failed_links,
        columns=[
            'url',
            'error'
        ]
    ).to_csv(
        FAILED_FILE,
        index=False,
        encoding='utf-8'
    )


    print(
        f"\n{len(failed_links)} "
        f"links failed/skipped."
    )

    print(
        f"Saved to: {FAILED_FILE}"
    )

else:

    print(
        "\nAll links scraped successfully."
    )


# ============================================================
# RETRY FAILED LINKS
# ============================================================

if failed_links:

    retry_links = [
        link

        for link, error in failed_links

        if error.startswith("__FAILED__")
    ]


    if retry_links:

        print(
            f"\nRetrying {len(retry_links)} "
            f"failed links..."
        )


        retry_results = run_pass(
            retry_links,
            RETRY_WORKERS
        )


        # -----------------------------------------------
        # Keep successful retries
        # -----------------------------------------------

        retry_clean = {

            link: msg

            for link, msg in retry_results.items()

            if not (
                isinstance(msg, str)
                and msg.startswith("__FAILED__")
            )
        }


        # -----------------------------------------------
        # Update successful retries
        # -----------------------------------------------

        if retry_clean:

            Kenyamoja.loc[
                Kenyamoja['url'].isin(
                    retry_clean.keys()
                ),
                'article'
            ] = (
                Kenyamoja['url']
                .map(retry_clean)
            )


            # -------------------------------------------
            # Save updated Parquet
            # -------------------------------------------

            Kenyamoja.to_parquet(
                OUTPUT_FILE,
                index=False,
                engine='pyarrow'
            )


            print(
                f"\nUpdated Parquet file saved to: "
                f"{OUTPUT_FILE}"
            )


        # -----------------------------------------------
        # Find links that still failed
        # -----------------------------------------------

        still_failed = {

            link: msg

            for link, msg in retry_results.items()

            if isinstance(msg, str)

            and msg.startswith("__FAILED__")
        }


        # -----------------------------------------------
        # Save remaining failures
        # -----------------------------------------------

        if still_failed:

            pd.DataFrame(
                still_failed.items(),
                columns=[
                    'url',
                    'error'
                ]
            ).to_csv(
                STILL_FAILED_FILE,
                index=False,
                encoding='utf-8'
            )


            print(
                f"\n{len(still_failed)} "
                f"links still failed after retry."
            )

            print(
                f"Saved to: {STILL_FAILED_FILE}"
            )

        else:

            print(
                "\nAll retried links succeeded."
            )