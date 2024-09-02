import os
import requests
import asyncio
import aiohttp
from bs4 import BeautifulSoup
from langchain_ollama import OllamaEmbeddings
from langchain.vectorstores import Chroma
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import ConversationalRetrievalChain
from langchain_groq import ChatGroq
from langchain.memory import ConversationBufferMemory
from langchain.prompts import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from dotenv import load_dotenv
import threading
import time
import logging

# Configure logging
logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    filename='fashion_bot.log',
                    filemode='w')

# Create a console handler
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)

# Add console handler to the root logger
logging.getLogger('').addHandler(console_handler)

logger = logging.getLogger(__name__)

load_dotenv()

# Load API Key
API_KEY = os.getenv("GROQ_API_KEY")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

class FashionBot:
    def __init__(self):
        self.documents = []
        self.vector_store = None
        self.llm = ChatGroq(temperature=0, groq_api_key=API_KEY, model_name="llama3-70b-8192")
        self.retriever = None
        self.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
        self.conversation = None
        self.data_fetching = False
        self.first_fetch = True
        self.fetch_interval = 3600  # 1 hour in seconds
        logger.info("FashionBot initialized")

    async def scrape_data_from_urls(self, urls):
        self.data_fetching = True
        logger.info("Starting data scraping")
        async with aiohttp.ClientSession() as session:
            tasks = [self.fetch_content(session, url) for url in urls]
            await asyncio.gather(*tasks)
        self.prepare_vector_store()
        self.data_fetching = False
        self.first_fetch = False
        logger.info("Data scraping completed")


        

    async def fetch_content(self, session, url):
        try:
            async with session.get(url) as response:
                if response.status == 200:
                    text = await response.text()
                    soup = BeautifulSoup(text, 'html.parser')

                    # Extract product details
                    products = []

                    # Example of how you might extract product details - adjust based on the actual HTML structure
                    product_elements = soup.find_all('div', class_='product')  # Adjust the selector based on the website

                    for product in product_elements:
                        product_name = product.find('h2', class_='product-name').text.strip() if product.find('h2', class_='product-name') else 'No Name'
                        product_image = product.find('img', class_='product-image')['src'] if product.find('img', class_='product-image') else 'No Image URL'
                        product_price = product.find('span', class_='price').text.strip() if product.find('span', class_='price') else 'No Price'
                        product_code = product.find('span', class_='product-code').text.strip() if product.find('span', class_='product-code') else 'No Code'

                        products.append({
                            "name": product_name,
                            "image_url": product_image,
                            "price": product_price,
                            "code": product_code,
                            "source": url
                        })

                    # Add each product as a document
                    for product in products:
                        self.documents.append(Document(
                            page_content=f"Product: {product['name']}\nPrice: {product['price']}\nCode: {product['code']}\nImage URL: {product['image_url']}",
                            metadata={"source": url, "image_url": product['image_url']}
                        ))
                        logger.debug(f"Extracted product details from {url}: {product}")

                else:
                    logger.error(f"Failed to retrieve content from {url}, status code: {response.status}")
        except Exception as e:
            logger.exception(f"An error occurred while fetching {url}: {e}")

    def prepare_vector_store(self):
        logger.info("Preparing vector store")
        # embeddings = HuggingFaceEmbeddings(
        #     model_name="sentence-transformers/all-MiniLM-L6-v2",
        #     model_kwargs={'device': 'cpu'},
        #     encode_kwargs={'normalize_embeddings': False}
        # )
        embeddings = OllamaEmbeddings(
            base_url=os.getenv("OLLAMA_URL"),
            model="mxbai-embed-large"
        )
                        
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=128)
        chunks = text_splitter.split_documents(self.documents)
        
        for i, chunk in enumerate(chunks[:5]):
            logger.debug(f"Chunk {i} from {chunk.metadata['source']}:\n{chunk.page_content[:500]}...")
        
        self.vector_store = Chroma.from_documents(chunks, embeddings)
        self.retriever = self.vector_store.as_retriever(k=20)
        self.setup_conversation_chain()
        self.data_fetching = False
        logger.info("Vector store prepared")

    def setup_conversation_chain(self):
        logger.info("Setting up conversation chain")

        general_system_template = """
        You are a helpful assistant with extensive knowledge about fashion brands and products. Your responses should:
        1. Provide accurate and relevant information based on the context provided.
        2. Mention the specific brand, color, fabric, and size for each item when referring to products.
        3. Ensure clarity and comprehensiveness in your responses.
        4. Filter the results, including images, based on the user's specified criteria such as brand, color, fabric, size, or other attributes mentioned in their query.
        5. Scrape the web to find the exact image URLs that match the filtered criteria and include them in your response in PNG format.

        Here's how to handle user queries:
        - Parse the user's question to identify filtering criteria (e.g., brand, color, fabric, size).
        - Use a scraping function to search for images that match the specified criteria.
        - Retrieve and list the fashion items, along with the exact image URLs, that match the user's query.
        - Ensure that the images provided correspond to the filtered products based on the user's criteria.
        - If no exact matches are found, suggest similar items or inform the user accordingly.
        - Always refer to the specific brand for each product mentioned in your response.

        Here's some information about the context and user queries:
        - Chat history: {chat_history}
        - Context: {context}
        - User's question: {question}

        Use the context to enhance the response and refer to the brands explicitly. Ensure that your answers are relevant, filtered according to the user's query, and useful. Use the correct and verified image URLs that match the user's filtering criteria.
        """

        general_user_template = """
        The user is asking:
        Question: ```{question}```
        """

        messages = [
            SystemMessagePromptTemplate.from_template(general_system_template),
            HumanMessagePromptTemplate.from_template(general_user_template)
        ]
        aqa_prompt = ChatPromptTemplate.from_messages(messages)

        self.conversation = ConversationalRetrievalChain.from_llm(
            self.llm, retriever=self.retriever, memory=self.memory, combine_docs_chain_kwargs={"prompt": aqa_prompt}, verbose=True
        )
        logger.info("Conversation chain set up")

    def get_urls(self):
        logger.info("Reading URLs from file")
        with open('urls.txt', 'r') as file:
            urls = file.read().splitlines()
        return urls

    async def initialize_data(self):
        logger.info("Initializing data")
        self.data_fetching = True
        urls = self.get_urls()
        await self.scrape_data_from_urls(urls)

    def get_response(self, question, num_results=20):
        if self.data_fetching:
            logger.warning("Data fetching in progress, unable to respond")
            return "Currently fetching data. Please try again in a few moments."
        elif not self.vector_store:
            logger.warning("Vector store not available, unable to respond")
            return "Data is not yet available. Please wait a moment and try again."
        self.first_fetch = False  # Reset first_fetch flag after successful data load
        if not self.vector_store:
            logger.warning("Vector store not available, unable to respond")
            return "Data is not yet available. Please try again in a few moments."
        try:
            logger.info(f"Generating response for question: {question}")
            self.retriever.search_kwargs['k'] = num_results
            response = self.conversation.run({'question': question})
            logger.debug(f"Generated response: {response}")
            return response
        except Exception as e:
            logger.exception(f"An error occurred while generating response: {e}")
            return "Sorry, I couldn't generate a response at the moment."

    def start_periodic_scraping(self):
        def run_scraping():
            while True:
                logger.info("Starting periodic scraping")
                asyncio.run(self.initialize_data())
                logger.info(f"Sleeping for {self.fetch_interval} seconds before next scrape")
                time.sleep(self.fetch_interval)

        thread = threading.Thread(target=run_scraping, daemon=True)
        thread.start()
        logger.info("Periodic scraping thread started")

# Initialize the bot and start periodic scraping
fashion_bot = FashionBot()
fashion_bot.start_periodic_scraping()

# Expose the fashion_bot instance
def get_fashion_bot():
    return fashion_bot