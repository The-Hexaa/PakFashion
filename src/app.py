import streamlit as st
from main import get_fashion_bot
import threading
from urls_finder import URLFinder
import logging

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    filename='fashion_bot_app.log',
                    filemode='a')

# Create a console handler
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)

# Add console handler to the root logger
logging.getLogger('').addHandler(console_handler)

logger = logging.getLogger(__name__)

# Get the FashionBot instance
fashion_bot = get_fashion_bot()
logger.info("FashionBot instance created")

def start_url_finder():
    """Start the URLFinder in a separate thread."""
    url_finder = URLFinder()
    url_finder.run()  # Change this to call the run method that handles the search and scraping
    logger.info("URL Finder started")

def initialize_url_finder():
    """Initialize the URL Finder using threading."""
    url_finder_thread = threading.Thread(target=start_url_finder)
    url_finder_thread.daemon = True  # Daemon thread will exit when the program does
    url_finder_thread.start()
    logger.info("URL Finder thread initialized")

# Initialize the URLFinder in a separate thread
initialize_url_finder()

# Streamlit app configuration
st.set_page_config(
    page_title="Fashion Brand Query Bot",
    page_icon="🛍️",
    layout="wide"
)
logger.info("Streamlit page configured")

# Streamlit App Header
st.title("Fashion Brand Query Bot")
st.write("Ask me about different fashion items from brands")

# User input section
user_input = st.chat_input("Type your question here...")

# Display conversation history
if "conversation" not in st.session_state:
    st.session_state["conversation"] = []
    logger.debug("Conversation history initialized")

# Handle user input and generate response
if user_input:
    logger.info(f"Received user input: {user_input}")
    st.session_state.conversation.append({"role": "user", "content": user_input})
    with st.spinner("Generating response..."):
        response = fashion_bot.rag_pipeline(user_input)  # Updated to use RAG pipeline
    st.session_state.conversation.append({"role": "bot", "content": response})
    logger.info("Response generated and added to conversation")

# Display the conversation
for message in st.session_state.conversation:
    if message["role"] == "user":
        st.chat_message("user").write(message["content"])
    else:
        st.chat_message("assistant").write(message["content"])

# Clear chat history button
if st.button("Clear Chat History"):
    st.session_state["conversation"] = []
    logger.info("Chat history cleared")
