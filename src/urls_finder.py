import logging
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from urllib.parse import urlparse
import time
import re
import threading
from dotenv import load_dotenv
import os

class URLFinder:
    def __init__(self):
        # Configure logging
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname=s - %(message=s')
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
        logging.getLogger('urllib3').setLevel(logging.CRITICAL)
        logging.getLogger('http.client').setLevel(logging.CRITICAL)
        logging.getLogger('asyncio').setLevel(logging.CRITICAL)

    def find_urls(self, search_query, max_results=10):
        service = Service("/usr/local/bin/chromedriver")
        options = webdriver.ChromeOptions()
        options.add_argument("--headless")  # Run in headless mode

        driver = webdriver.Chrome(service=service, options=options)
        urls = []
        try:
            driver.get("https://www.google.com")
            search_box = driver.find_element(By.NAME, "q")
            search_box.send_keys(search_query)
            search_box.submit()
            time.sleep(3)
            wait = WebDriverWait(driver, 10)
            wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, 'div#search')))
            links = driver.find_elements(By.CSS_SELECTOR, 'div#search a')

            for link in links[:max_results]:
                href = link.get_attribute("href")
                if href and re.match(r'^https?://', href):
                    parsed_url = urlparse(href)
                    if parsed_url.netloc.endswith(".com"):
                        urls.append(href)

        except Exception as e:
            logging.error(f"An error occurred while finding URLs: {e}")
        finally:
            driver.quit()  # Ensure the driver is closed properly