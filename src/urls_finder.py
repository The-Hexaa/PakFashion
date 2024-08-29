import logging
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.keys import Keys
# from webdriver_manager.chrome import ChromeDriverManager
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
        # Configure headless browser
        self.options = Options()
        self.options.add_argument("--headless")
        self.options.add_argument("--disable-gpu")
        self.options.add_argument("--no-sandbox")
        self.options.add_argument("--disable-dev-shm-usage")
        # Disable ChromeDriver logs
        self.options.add_experimental_option('excludeSwitches', ['enable-logging'])

        # Load environment variables from .env file
        load_dotenv()

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
        
        with webdriver.Chrome(options=self.options) as driver:
            for search_url in self.search_urls:
                logging.info(f"Sending request to {search_url}")
                driver.get(search_url)
                for page in range(1, 6):  # Fetch first 5 pages
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
                        next_button = driver.find_element(By.XPATH, "//a[@aria-label='Next']")  # Adjust the XPath as needed for different search engines
                        if next_button:
                            next_button.click()
                        else:
                            break
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
