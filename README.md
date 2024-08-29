# PakFashion
## Overview

The **Fashion Brand Query Bot** is a Streamlit application that allows users to search and explore fashion items from brands like Khaadi, Sapphire, Generation, and Rang Ja. Users can ask the bot questions about various fashion items, and the bot will respond with relevant details.

## Features

- Search for fashion items by color, type, or brand.
- Display detailed information about each item, including price and description.
- Visual display of the item images within the Streamlit app.
- Easy-to-use interface powered by Streamlit.

## Installation

1. Clone the repository:
    ```bash
    git clone git@github.com:The-Hexaa/PakFashion.git 
    ```

2. Navigate to the project directory:
    ```bash
    cd PakFashion
    ```

3. Set up a virtual environment (optional but recommended):
    ```bash
    python3 -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

4. Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

1. Run the Streamlit app:
    ```bash
    streamlit run app.py
    ```

2. The application will open in your web browser. You can now start querying the bot about fashion items.

## How It Works

- The bot fetches data from a catalog of fashion items.
- You can type queries like "Show me blue dresses" or "Find embroidered shirts from Khaadi."
- The bot will display a list of items matching your query, including pictures and detailed descriptions.

## URL Finder Script

The `urls_finder.py` script is designed to periodically search for URLs related to Pakistani women clothing brands using various search engines. It uses Selenium to automate the browser and fetch URLs, which are then saved to a file.

### Features

- **Automated Search**: The script automatically searches for URLs based on a predefined query and search engines.
- **Periodic Execution**: The search is performed periodically at a specified interval.
- **URL Filtering**: Filters out unwanted URLs from specific domains.
- **Logging**: Logs the search process and results for easy monitoring.

### How to Use

1. **Set Up Environment Variables**: Create a `.env` file in the project directory with the following variables:
    ```
    SEARCH_QUERY="your search query"
    SEARCH_ENGINES="comma,separated,list,of,search,engine,urls"
    ```

2. **Run the Script**: Execute the script to start the periodic search.
    ```bash
    python urls_finder.py
    ```

3. **Check Results**: The found URLs will be saved in `urls.txt`.

### Example

To search for Pakistani women clothing brands, set the following in your `.env` file:

