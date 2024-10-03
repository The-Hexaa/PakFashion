import logging
import re
import time
from selenium.common.exceptions import StaleElementReferenceException
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
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import chromadb
from sentence_transformers import SentenceTransformer

class URLFinder:
    def __init__(self):
        # Configure logging with DEBUG level for more details
        logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
        self.logger = logging.getLogger(__name__)

        # Configure headless browser (optional: uncomment headless mode)
        self.options = Options()
        self.options.add_argument("--headless")
        self.options.add_argument("--disable-gpu")
        self.options.add_argument("--no-sandbox")
        self.options.add_argument("--disable-dev-shm-usage")
        self.options.add_experimental_option('excludeSwitches', ['enable-logging'])
        logging.getLogger('selenium').setLevel(logging.CRITICAL)
        logging.getLogger('urllib3').setLevel(logging.CRITICAL)

        # Load environment variables
        load_dotenv()
        self.search_query = os.getenv("SEARCH_QUERY", "pakistani women clothing brands").replace(" ", "+")

        # Load search engines from a separate file to avoid confusion
        self.search_engines = self.load_search_engines_from_file("search_engines.txt")
        self.excluded_domains = os.getenv("EXCLUDED_DOMAINS", "bingplaces.com,youtube.com,google.com").split(',')
        self.search_urls = [f"{engine}{self.search_query}" for engine in self.search_engines]
        self.seen_urls = set()
        self.scraped_urls = self.read_existing_urls("scraped_urls.txt")
        self.url_pattern = re.compile(r'https?://[^\s<>"]+|www\.[^\s<>"]+')

        # Initialize ChromaDB client and collection
        self.chroma_client = chromadb.Client()
        self.collection = self.chroma_client.get_or_create_collection("clothing_brands")

        # Initialize SentenceTransformer model for embedding generation
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

    def load_search_engines_from_file(self, file_path):
        """Loads search engines from a file."""
        try:
            with open(file_path, "r") as file:
                return [line.strip() for line in file if line.strip()]
        except FileNotFoundError:
            self.logger.warning(f"{file_path} not found. Using default search engines.")
            return [
                "https://www.google.com/search?q=",
                "https://www.bing.com/search?q=",
                "https://search.yahoo.com/search?p="
            ]

    def read_existing_urls(self, file_path):
        """Reads existing URLs from a file if available."""
        try:
            with open(file_path, "r") as file:
                return set(file.read().splitlines())
        except FileNotFoundError:
            return set()

    def search_pakistani_women_clothing_brands(self):
        """Searches for Pakistani women clothing brands and saves the found URLs."""
        self.logger.info("Starting search for Pakistani women clothing brands.")
        try:
            service = Service(ChromeDriverManager().install())
            with webdriver.Chrome(options=self.options, service=service) as driver:
                for search_url in self.search_urls:
                    self.logger.info(f"Sending request to {search_url}")
                    driver.get(search_url)
                    for page in range(1, 6):  # Fetch first 5 pages
                        try:
                            WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.TAG_NAME, "a")))
                            links = driver.find_elements(By.TAG_NAME, "a")
                            self.logger.debug(f"Found {len(links)} links on page {page}")
                            for link in links:
                                url = link.get_attribute('href')
                                if url and self.url_pattern.match(url):
                                    parsed_url = urlparse(url)
                                    main_domain = f"{parsed_url.scheme}://{parsed_url.netloc}"
                                    if main_domain not in self.seen_urls and all(excluded not in main_domain for excluded in self.excluded_domains):
                                        self.seen_urls.add(main_domain)
                                        self.logger.info(f"Found URL: {main_domain}")
                            # Attempt to click the 'Next' button to navigate to the next page
                            next_buttons = driver.find_elements(By.XPATH, "//a[contains(text(), 'Next')] | //a[@aria-label='Next']")
                            if next_buttons:
                                self.logger.info("Clicking next button")
                                next_buttons[0].click()
                                time.sleep(2)  # Wait for the next page to load
                            else:
                                self.logger.info("No next button found or no more pages.")
                                break
                        except Exception as e:
                            self.logger.warning(f"Failed to retrieve content from {search_url} page {page}: {e}")
                            break
            # Save all found URLs to 'urls.txt'
            with open("urls.txt", "w") as file:
                for url in self.seen_urls:
                    file.write(f"{url}\n")
                    self.logger.info(f"Saved URL: {url}")
        except Exception as e:
            self.logger.error(f"An error occurred during URL search: {e}")
        self.logger.info("Search complete. URLs saved to urls.txt.")

    def scrape_webpage(self, url, start_time):
        """Scrapes product details and images from a website."""
        try:
            self.logger.info(f"Scraping content from: {url}")
            service = Service(ChromeDriverManager().install())
            with webdriver.Chrome(options=self.options, service=service) as driver:
                driver.get(url)

                # Step 1: Collect product page links
                product_links = driver.find_elements(By.TAG_NAME, "a")
                product_urls = set(link.get_attribute('href') for link in product_links if link.get_attribute('href') and "product" in link.get_attribute('href'))
                self.logger.debug(f"Found {len(product_urls)} product URLs on {url}")

                for product_url in product_urls:
                    # Check if 2 minutes have passed for the entire scraping session
                    if time.time() - start_time >= 120:
                        self.logger.info("Global time limit reached. Stopping scraping.")
                        return  # Exit scraping early if time limit is exceeded
                    
                    try:
                        if self.is_product_already_stored(product_url):
                            self.logger.info(f"Product already exists in ChromaDB, skipping: {product_url}")
                            continue  # Skip to the next product if already stored

                        # Retry logic in case of StaleElementReferenceException
                        retry_count = 3
                        while retry_count > 0:
                            try:
                                driver.get(product_url)
                                WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.TAG_NAME, "body")))

                                product_name = self.find_product_detail(driver, ["h1", "h2", "title"], ["product", "name", "title"])
                                product_price = self.find_product_detail(driver, ["span", "div"], ["price", "amount", "money"])
                                product_description = self.find_product_detail(driver, ["p", "div"], ["description", "details"])
                                product_image = self.find_product_image(driver)

                                self.logger.info(f"Product Name: {product_name}")
                                self.logger.info(f"Product Price: {product_price}")
                                self.logger.info(f"Product Description: {product_description}")
                                self.logger.info(f"Product Image URL: {product_image}")

                                if product_description != "Detail not found":
                                    description_embedding = self.embedding_model.encode(product_description).tolist()
                                else:
                                    description_embedding = []

                                if description_embedding:
                                    product_id = product_url.split("/")[-1]

                                    self.collection.add(
                                        ids=[product_id],
                                        embeddings=[description_embedding],
                                        metadatas=[{
                                            "url": product_url,
                                            "name": product_name,
                                            "price": product_price,
                                            "description": product_description,
                                            "image": product_image
                                        }]
                                    )
                                    self.logger.info(f"Successfully added product to ChromaDB: {product_name}")
                                else:
                                    self.logger.warning(f"No embedding generated for {product_url}, skipping storage.")

                                break  # Successful scrape, break retry loop
                            except StaleElementReferenceException as e:
                                self.logger.warning(f"Stale element found, retrying... ({3 - retry_count + 1})")
                                retry_count -= 1
                                time.sleep(1)
                                if retry_count == 0:
                                    self.logger.error(f"Failed to retrieve product details after retries: {product_url}")
                                    continue
                    except Exception as e:
                        self.logger.error(f"Error while scraping product URL {product_url}: {e}")

        except Exception as e:
            self.logger.error(f"An error occurred while scraping the webpage: {e}")

    def is_product_already_stored(self, product_url):
        """Check if the product with the given URL or ID already exists in the ChromaDB collection."""
        try:
            # Use the product URL or a derived product ID as a unique identifier
            product_id = product_url.split("/")[-1]  # Assuming the last part of the URL is unique

            # Query ChromaDB to check if this product ID already exists
            existing_product = self.collection.get(ids=[product_id])
            if existing_product['ids']:
                return True  # Product already exists
        except Exception as e:
            self.logger.warning(f"Failed to check if product already exists: {e}")
        return False

    def find_product_detail(self, driver, tag_names, keywords):
        """Tries to extract a product detail based on tag names and keywords."""
        for tag in tag_names:
            elements = driver.find_elements(By.TAG_NAME, tag)
            for element in elements:
                text = element.text.lower()
                if any(keyword in text for keyword in keywords):
                    return element.text
        return "Detail not found"

    def find_product_image(self, driver):
        """Attempts to find and return a product image URL."""
        img_elements = driver.find_elements(By.TAG_NAME, "img")

        for img in img_elements:
            src = img.get_attribute("src")
            alt_text = img.get_attribute("alt")  # Fetch the alt attribute to check for relevant keywords
            
            # Assuming product image URLs contain "product" or "item" and the alt attribute contains descriptive keywords
            if src and ("product" in src or "item" in src or "clothing" in src or "dress" in src):
                return src
            if alt_text and ("product" in alt_text or "item" in alt_text or "clothing" in alt_text or "dress" in alt_text):
                return src
        
        return "Product image not found"
    
    def fetch_chromadb_documents(self):
        """Fetch and print all stored documents from ChromaDB."""
        try:
            documents = self.collection.get()
            print(documents)
        except Exception as e:
            self.logger.error(f"Error fetching documents from ChromaDB: {e}")


    def run(self):
        self.search_pakistani_women_clothing_brands()

        start_time = time.time()  

        # Read saved URLs and scrape products
        with open("urls.txt", "r") as file:
            for url in file.readlines():
                # Stop scraping if the time limit is reached
                if time.time() - start_time >= 120:
                    self.logger.info("Global time limit reached. Stopping scraping.")
                    break
                
                self.scrape_webpage(url.strip(), start_time)

        self.fetch_chromadb_documents()


if __name__ == "__main__":
    finder = URLFinder()
    finder.run()