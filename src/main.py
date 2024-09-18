import os
import logging
import time
import threading
import requests
import schedule
from dotenv import load_dotenv
from urls_finder import URLFinder  # Import URLFinder from urls_finder.py
from langchain_community.vectorstores import Chroma  # Updated import for Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.memory import ConversationBufferMemory
from langchain.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain.chains import ConversationalRetrievalChain
from langchain_groq import ChatGroq

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', filename='fashion_bot.log', filemode='w')
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
logging.getLogger('').addHandler(console_handler)

logger = logging.getLogger(__name__)

# Load environment variables from .env
load_dotenv()

API_KEY = os.getenv("GROQ_API_KEY")
GROQ_API_URL = "https://api.groq.com/v1/embeddings"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

class FashionBot:
    def __init__(self):
        self.documents = []
        self.vector_store = None
        self.retriever = None
        self.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
        self.conversation = None
        self.url_finder = URLFinder()  # Instantiate URLFinder for scraping
        logger.info("FashionBot initialized")

    def get_groq_embedding(self, text):
        """Gets text embeddings from Groq API."""
        try:
            headers = {
                "Authorization": f"Bearer {API_KEY}",
                "Content-Type": "application/json"
            }
            response = requests.post(
                GROQ_API_URL,
                headers=headers,
                json={"inputs": [text]}
            )
            response.raise_for_status()
            data = response.json()
            embedding = data['data'][0]['embedding']
            return embedding
        except requests.exceptions.RequestException as e:
            logger.error(f"Error fetching embeddings from Groq: {e}")
            return None
        
    def start_scraping(self, timeout=60):
        """Starts scraping using URLFinder and stores data in the vector store."""
        logger.info("Starting URL scraping using URLFinder...")

        documents=""

        try:
            start_time = time.time()
            documents = self.url_finder.start_search()
            if not documents:
                logger.warning("No documents found to scrape.")
            else:
                elapsed_time = time.time() - start_time
                logger.info(f"Scraped {len(documents)} documents in {elapsed_time:.2f} seconds.")
                
                if elapsed_time > timeout:
                    logger.warning(f"Scraping took too long. Timeout after {timeout} seconds.")
        except Exception as e:
            logger.exception("Failed during the URL search process.")

        self.update_vector_store(documents)

    def setup_conversation_chain(self):
        """Set up the conversation chain for querying the LLM."""
        logger.info("Setting up conversation chain")
        try:
            if not self.retriever:
                logger.error("Retriever is not initialized, can't set up conversation chain.")
                return

            # Define the prompts for the system and human messages
            system_message_prompt = SystemMessagePromptTemplate(
                content="You are a helpful assistant for fashion-related queries."
            )
            human_message_prompt = HumanMessagePromptTemplate.from_template("{input}")
            chat_prompt_template = ChatPromptTemplate.from_messages([system_message_prompt, human_message_prompt])

            # Set up the conversational retrieval chain
            self.conversation = ConversationalRetrievalChain(
                retriever=self.retriever,
                memory=self.memory,
                llm=self.get_groq_llm(),
                prompt_template=chat_prompt_template  # Ensure template is correctly formatted
            )

            logger.info("Conversation chain set up successfully.")
        except Exception as e:
            logger.error(f"Failed to set up conversation chain: {e}")

    def update_vector_store(self, documents):
        """Update the vector store with new documents."""
        if documents:
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=128)
            chunks = text_splitter.split_text("\n".join(documents))  # Adjusted for list of documents

            try:
                embeddings = [self.get_groq_embedding(chunk) for chunk in chunks]
                embeddings = [e for e in embeddings if e]  # Filter out failed embeddings

                if embeddings:
                    if self.vector_store:
                        logger.info("Adding documents to the existing vector store.")
                        self.vector_store.add_texts(chunks, embeddings=embeddings)
                    else:
                        logger.info("Creating a new vector store.")
                        self.vector_store = Chroma.from_texts(chunks, embeddings=embeddings)

                    self.retriever = self.vector_store.as_retriever(k=20)
                    logger.info("Vector store updated successfully.")

                    # After the vector store is ready, set up the conversation chain
                    self.setup_conversation_chain()

                    if not self.conversation:
                        logger.error("Conversation chain setup failed.")
                    else:
                        logger.info("Conversation chain is now available.")
            except Exception as e:
                logger.error(f"Failed to update vector store: {e}")
        else:
            logger.warning("No documents available to update the vector store.")

    def get_groq_llm(self):
        """Returns a Groq LLM object for conversation."""
        return ChatGroq(temperature=0, groq_api_key=API_KEY, model_name="llama3-70b-8192")

    def answer_query(self, query):
        """Process the user query using the conversation chain."""
        logger.info(f"User query: {query}")
        
        if not self.conversation:
            logger.error("Conversation chain not set up yet.")
            return "Error: Conversation chain is not available."

        try:
            response = self.conversation.run(query)
            return response
        except Exception as e:
            logger.error(f"Error during conversation chain execution: {e}")
            return "Error: Failed to process the query."
        
    def cronjob_for_scraping(self):
        """Run the scraping process in a separate thread."""
        scraping_thread = threading.Thread(target=self.start_scraping)
        scraping_thread.start()

    def start(self):
        """Start the bot and handle user queries."""
        logger.info("Bot is now accepting queries.")
        while True:
            user_query = input("Enter your fashion query: ")
            response = self.answer_query(user_query)
            print(f"Response: {response}")
            time.sleep(1)

def run_scheduler(bot):
    """Scheduler to run scraping job every hour."""
    schedule.every(1).hour.do(bot.cronjob_for_scraping)
    while True:
        schedule.run_pending()
        time.sleep(1)

if __name__ == "__main__":
    bot = FashionBot()

    # Run the scraping job immediately at start
    bot.cronjob_for_scraping()

    # Start the scheduler in a separate thread
    scheduler_thread = threading.Thread(target=run_scheduler, args=(bot,), daemon=True)
    scheduler_thread.start()

    # Run the bot's start method in the main thread (for user input)
    try:
        bot.start()
    except KeyboardInterrupt:
        print("Interrupted by user. Exiting...")
    finally:
        print("Shutting down...")
