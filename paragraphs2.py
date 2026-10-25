# Citizen scraper
import requests
from bs4 import BeautifulSoup


# link
url= 'https://citizen.digital/article/gachagua-claims-ruto-on-mission-to-trade-kenyas-minerals-in-new-york-n390557'
# Find links
response = requests.get(url)
if response.status_code == 200:
    # 3. Parse the HTML content using the built-in parser
    soup = BeautifulSoup(response.text, "html.parser")
    
    # 4. Find all containers holding the target data (e.g., <div class="quote">)
    quotes = soup.find_all("div", class_="article-content__paragraph_XUrcR js-article-paragraph")
