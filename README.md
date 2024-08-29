# PakFashion
This is rag based chatbot which scrap data from different pakistani brands and show it on user requirements.
## URL Finder

This project is a URL Finder that scrapes data from different Pakistani women clothing brands and saves the URLs to a file. It uses Selenium to automate the web browsing and scraping process.

### Features

- Searches for Pakistani women clothing brands using multiple search engines.
- Scrapes URLs from the search results.
- Filters out unwanted URLs (e.g., social media, search engines).
- Saves the found URLs to a file (`urls.txt`).
- Runs periodically to keep the list of URLs updated.

### Requirements

- Python 3.x
- Selenium
- WebDriver Manager
- dotenv

### Installation

1. Clone the repository:
    ```sh
    git clone https://github.com/yourusername/url-finder.git
    cd url-finder
    ```

2. Install the required packages:
    ```sh
    pip install -r requirements.txt
    ```

3. Create a `.env` file in the root directory and add your environment variables:
    ```sh
    cp .env.example .env
    ```

4. Update the `.env` file with your API key and search query:
    ```sh
    GROQ_API_KEY=your_api_key
    SEARCH_QUERY="pakistani women clothing brands"
    SEARCH_ENGINES="https://www.google.com/search?q=,https://www.bing.com/search?q=,https://search.yahoo.com/search?p=,https://duckduckgo.com/?q="
    ```

### Usage

1. Run the URL Finder:
    ```sh
    python urls_finder.py
    ```

2. The script will start searching for Pakistani women clothing brands and save the found URLs to `urls.txt`.

### File Structure

- `urls_finder.py`: Main script that performs the URL finding and scraping.
- `urls.txt`: File where the found URLs are saved.
- `.env`: Environment variables file.
- `.env.example`: Example environment variables file.

### License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

### Acknowledgements

- [Selenium](https://www.selenium.dev/)
- [WebDriver Manager](https://github.com/SergeyPirogov/webdriver_manager)
- [dotenv](https://github.com/theskumar/python-dotenv)
