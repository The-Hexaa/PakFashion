import logging
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
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

        # Load environment variables from .env file
        load_dotenv()

        # Retrieve search query from environment variables
        self.search_query = os.getenv("SEARCH_QUERY", "pakistani women clothing brands").replace(" ", "+")
        
        # Load search engines from urls.txt or use defaults
        self.search_engines = self.load_search_engines_from_file("urls.txt")

        # Retrieve excluded domains from environment variables or use defaults
        self.excluded_domains = os.getenv("EXCLUDED_DOMAINS", "bingplaces.com,youtube.com,google.com,bing.com,microsoft.com,facebook.com,instagram.com,twitter.com,yahoo.com,duckduckgo.com").split(',')

        # Construct search URLs
        self.search_urls = [f"{engine}{self.search_query}" for engine in self.search_engines]

        self.seen_urls = set()
        self.url_pattern = re.compile(r'https?://[^\s<>"]+|www\.[^\s<>"]+')
        self.scraped_urls = set()

    def load_search_engines_from_file(self, file_path):
        """Loads search engines from a file."""
        try:
            with open(file_path, "r") as file:
                return [line.strip() for line in file if line.strip()]
        except FileNotFoundError:
            logging.warning(f"{file_path} not found. Using default search engines.")
            return [
                "https://www.google.com/search?q=",
                "https://www.bing.com/search?q=",
                "https://search.yahoo.com/search?p="
            ]

    def read_existing_urls(self, file_path):
        """Reads existing URLs from a file if available"""
        try:
            with open(file_path, "r") as file:
                return set(file.read().splitlines())
        except FileNotFoundError:
            return set()

    def run_periodically(self, interval, func, *args, **kwargs):
        """Runs a function at regular intervals in a separate thread"""
        def wrapper():
            while True:
                func(*args, **kwargs)
                time.sleep(interval)
        thread = threading.Thread(target=wrapper)
        thread.daemon = True
        thread.start()

    def search_pakistani_women_clothing_brands(self):
        """Searches for Pakistani women clothing brands and saves the found URLs"""
        logging.info("Starting search for Pakistani women clothing brands.")
        try:
            service = Service(ChromeDriverManager().install())
            with webdriver.Chrome(options=self.options, service=service) as driver:
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
                                    if main_domain not in self.seen_urls and all(excluded not in main_domain for excluded in self.excluded_domains):
                                        self.seen_urls.add(main_domain)
                                        logging.info(f"Found URL: {main_domain}")
                            next_buttons = driver.find_elements(By.XPATH, "//a[contains(text(), 'Next')] | //a[@aria-label='Next']")
                            if next_buttons:
                                next_buttons[0].click()
                                time.sleep(2)
                            else:
                                logging.info("No next button found or no more pages.")
                                break
                        except Exception as e:
                            logging.warning(f"Failed to retrieve content from {search_url} page {page}: {e}")
                            break
            with open("urls.txt", "w") as file:
                for url in self.seen_urls:
                    file.write(f"{url}\n")
                    logging.info(f"Saved URL: {url}")
        except Exception as e:
            logging.error(f"An error occurred during URL search: {e}")
        logging.info("Search complete. URLs saved to urls.txt.")

    def scrape_webpage(self, url):
        """Scrapes content from a single webpage"""
        try:
            logging.info(f"Scraping content from: {url}")
            service = Service(ChromeDriverManager().install())
            options = Options()
            options.add_argument("--headless")
            options.add_argument("--disable-gpu")
            options.add_argument("--no-sandbox")
            options.add_argument("--disable-dev-shm-usage")
            options.add_experimental_option('excludeSwitches', ['enable-logging'])
            with webdriver.Chrome(options=options, service=service) as driver:
                driver.get(url)
                page_content = driver.find_element(By.TAG_NAME, "body").text
                logging.info(f"Page content from {url}: {page_content[:200]}...")  # Print first 200 characters for brevity
        except Exception as e:
            logging.error(f"Failed to scrape content from {url}: {e}")

    def start_scraping(self):
        """Starts the scraping process"""
        # Load URLs from file if available
        existing_urls = self.read_existing_urls("urls.txt")
        for url in existing_urls:
            if url not in self.scraped_urls:
                self.scrape_webpage(url)
                self.scraped_urls.add(url)
        logging.info("Scraping complete.")

    def start_search(self):
        """Continuously runs the search operation at regular intervals"""
        while True:
            self.search_pakistani_women_clothing_brands()
            self.start_scraping()
            time.sleep(3600)  # Run the search and scrape every hour


if __name__ == "__main__":
    url_finder = URLFinder()
    url_finder.start_search()
