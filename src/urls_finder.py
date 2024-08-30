import logging
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.keys import Keys
from webdriver_manager.chrome import ChromeDriverManager
import time
import re
from urllib.parse import urlparse
import threading
from dotenv import load_dotenv
import os


class URLFinder:
    def __init__(self):
        # Configure logging
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
        # Load environment variables from .env file
        load_dotenv()
        # Configure headless browser
        self.options = Options()
        self.options.add_argument("--headless")
        self.options.add_argument("--disable-gpu")
        self.options.add_argument("--no-sandbox")
        self.options.add_argument("--disable-dev-shm-usage")
        # Disable ChromeDriver logs
        self.options.add_experimental_option('excludeSwitches', ['enable-logging'])
        # Disable Selenium logging
        logging.getLogger('selenium').setLevel(logging.CRITICAL)
        # Disable urllib3 logging
        logging.getLogger('urllib3').setLevel(logging.CRITICAL)
        # Disable WebDriver Manager logs
        logging.getLogger('WDM').setLevel(logging.NOTSET)



        # Retrieve search query and search engines from environment variables
        self.search_query = os.getenv("SEARCH_QUERY", "pakistani women clothing brands").replace(" ", "+")
        self.search_engines = os.getenv("SEARCH_ENGINES").split(',')

        # Construct search URLs
        self.search_urls = [f"{engine}{self.search_query}" for engine in self.search_engines]

        self.seen_urls = set()
        self.url_pattern = re.compile(r'https?://[^\s<>"]+|www\.[^\s<>"]+')

    def read_existing_urls(self, file_path):
        try:
            with open(file_path, "r") as file:
                return set(file.read().splitlines())
        except FileNotFoundError:
            return set()

    def run_periodically(self, interval, func, *args, **kwargs):
        def wrapper():
            while True:
                func(*args, **kwargs)
                time.sleep(interval)
        thread = threading.Thread(target=wrapper)
        thread.daemon = True
        thread.start()

    def search_pakistani_women_clothing_brands(self):
        logging.info("Searching for Pakistani women clothing brands.")
        
        # Use ChromeDriverManager without log_level parameter
        service = Service(ChromeDriverManager().install())
        
        # Get the number of pages to search from .env, default to 5 if not specified
        pages_to_search = int(os.getenv("PAGES_TO_SEARCH", "5"))
        
        with webdriver.Chrome(options=self.options, service=service) as driver:
            for search_url in self.search_urls:
                logging.info(f"Sending request to {search_url}")
                driver.get(search_url)
                for page in range(1, pages_to_search + 1):
                    try:
                        WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.TAG_NAME, "a")))
                        links = driver.find_elements(By.TAG_NAME, "a")
                        for link in links:
                            url = link.get_attribute('href')
                            if url and self.url_pattern.match(url):
                                parsed_url = urlparse(url)
                                main_domain = f"{parsed_url.scheme}://{parsed_url.netloc}"
                                if main_domain not in self.seen_urls and all(excluded not in main_domain for excluded in ["bingplaces.com", "youtube.com", "google.com", "bing.com", "microsoft.com", "facebook.com", "instagram.com", "twitter.com", "yahoo.com", "duckduckgo.com"]):
                                    self.seen_urls.add(main_domain)
                                    logging.info(f"Found URL {main_domain}")
                        
                        if page < pages_to_search:
                            try:
                                next_button = WebDriverWait(driver, 10).until(
                                    EC.element_to_be_clickable((By.XPATH, "//a[@aria-label='Next']"))
                                )
                                next_button.click()
                                time.sleep(2)  # Wait for the page to load
                            except Exception as e:
                                logging.warning(f"Failed to navigate to next page: {e}")
                                break
                        else:
                            logging.info(f"Reached the configured number of pages ({pages_to_search}) for {search_url}")
                    except Exception as e:
                        logging.warning(f"Failed to retrieve content from {search_url} page {page}: {e}")
                        break
        with open("urls.txt", "w") as file:
            for url in self.seen_urls:
                file.write(f"{url}\n")
                logging.info(f"Saved URL {url}")
        logging.info("Searched Pakistani women clothing brands and their URLs have been saved to urls.txt")

    def start_search(self):
        while True:
            self.search_pakistani_women_clothing_brands()
            time.sleep(60)


if __name__ == "__main__":
    url_finder = URLFinder()
    url_finder.start_search()
