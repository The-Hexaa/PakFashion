import logging
import re
import time
from selenium.common.exceptions import StaleElementReferenceException
import threading
from urllib.parse import urlparse
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager
from dotenv import load_dotenv
import os

class URLFinder:
    def __init__(self):
        # Configure logging with DEBUG level for more details
        logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

        # Configure headless browser (removed headless mode for debugging)
        self.options = Options()
        # Comment out headless mode for easier debugging (optional: re-enable once working)
        # self.options.add_argument("--headless")
        self.options.add_argument("--disable-gpu")
        self.options.add_argument("--no-sandbox")
        self.options.add_argument("--disable-dev-shm-usage")
        self.options.add_experimental_option('excludeSwitches', ['enable-logging'])
        logging.getLogger('selenium').setLevel(logging.CRITICAL)
        logging.getLogger('urllib3').setLevel(logging.CRITICAL)

        # Load environment variables
        load_dotenv()
        self.search_query = os.getenv("SEARCH_QUERY", "pakistani women clothing brands").replace(" ", "+")

        # Load search engines
        self.search_engines = self.load_search_engines_from_file("urls.txt")
        self.excluded_domains = os.getenv("EXCLUDED_DOMAINS", "bingplaces.com,youtube.com,google.com").split(',')
        self.search_urls = [f"{engine}{self.search_query}" for engine in self.search_engines]
        self.seen_urls = set()
        self.scraped_urls = self.read_existing_urls("scraped_urls.txt")
        self.url_pattern = re.compile(r'https?://[^\s<>"]+|www\.[^\s<>"]+')

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
                            logging.debug(f"Found {len(links)} links on page {page}")
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
                                logging.info("Clicking next button")
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
        """Scrapes product details and images from a website"""
        try:
            logging.info(f"Scraping content from: {url}")
            service = Service(ChromeDriverManager().install())
            with webdriver.Chrome(options=self.options, service=service) as driver:
                driver.get(url)

                # Step 1: Collect product page links
                product_links = driver.find_elements(By.TAG_NAME, "a")
                product_urls = set(link.get_attribute('href') for link in product_links if "product" in link.get_attribute('href'))
                logging.debug(f"Found {len(product_urls)} product URLs on {url}")

                for product_url in product_urls:
                    try:
                        # Retry logic in case of StaleElementReferenceException
                        retry_count = 3
                        while retry_count > 0:
                            try:
                                driver.get(product_url)
                                WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.TAG_NAME, "body")))

                                # Step 2: Find product details
                                product_name = self.find_product_detail(driver, ["h1", "h2", "title"], ["product", "name", "title"])
                                product_price = self.find_product_detail(driver, ["span", "div"], ["price", "amount", "money"])
                                product_description = self.find_product_detail(driver, ["p", "div"], ["description", "details"])
                                product_image = self.find_product_image(driver)  # Added image scraping

                                logging.info(f"Product Name: {product_name}")
                                logging.info(f"Product Price: {product_price}")
                                logging.info(f"Product Description: {product_description}")
                                logging.info(f"Product Image URL: {product_image}")
                                break  # Break out of the retry loop if successful
                            except StaleElementReferenceException as e:
                                logging.warning(f"Stale element found, retrying... ({3 - retry_count + 1})")
                                retry_count -= 1
                                time.sleep(2)  # Wait for 2 seconds before retrying
                                if retry_count == 0:
                                    logging.error(f"Failed to scrape product details from {product_url} after retries: {e}")
                    except Exception as e:
                        logging.warning(f"Failed to scrape product details from {product_url}: {e}")
        except Exception as e:
            logging.error(f"Failed to scrape content from {url}: {e}")

    def find_product_image(self, driver):
        """Finds product image URL based on img tag and src attribute, ignoring logos and footer images"""
        images = driver.find_elements(By.TAG_NAME, "img")

        for image in images:
            src = image.get_attribute("src")
            alt_text = image.get_attribute("alt") or ""
            class_attr = image.get_attribute("class") or ""

            # Check if it's a product image by checking its attributes (adjust these based on your inspection)
            if "product" in alt_text.lower() or "product" in class_attr.lower():
                return src

        return "Product image not found"


    def find_product_detail(self, driver, tags, keywords):
        """Finds product details like name, price, and description based on tags and keywords"""
        for tag in tags:
            elements = driver.find_elements(By.TAG_NAME, tag)
            logging.debug(f"Checking {len(elements)} elements with tag '{tag}' for keywords {keywords}")
            for element in elements:
                if element is not None:
                    class_attr = element.get_attribute("class") or ""
                    id_attr = element.get_attribute("id") or ""
                    itemprop_attr = element.get_attribute("itemprop") or ""
                    if any(keyword in attr.lower() for keyword in keywords for attr in (class_attr, id_attr, itemprop_attr)):
                        return element.text
        return "Not Found"

    def start_scraping(self):
        """Starts the scraping process"""
        existing_urls = self.read_existing_urls("urls.txt")
        for url in existing_urls:
            if url not in self.scraped_urls:
                self.scrape_webpage(url)
                self.scraped_urls.add(url)
                with open("scraped_urls.txt", "a") as file:
                    file.write(f"{url}\n")
        logging.info("Scraping complete.")

    def start_search(self):
        """Continuously runs the search operation at regular intervals"""
        # while True:
        self.search_pakistani_women_clothing_brands()
        self.start_scraping()
        logging.info("test----------------------------------------------")
        # time.sleep(3600)  # Run the search and scrape every hour


if __name__ == "__main__":
    url_finder = URLFinder()
    url_finder.start_search()
