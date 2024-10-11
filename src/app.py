import streamlit as st
import threading
import logging
import os
from sqlalchemy import create_engine, Column, Integer, String
from sqlalchemy.orm import declarative_base, sessionmaker
import bcrypt
from main import get_fashion_bot
from urls_finder import URLFinder

# =======================
# Database Configuration
# =======================

Base = declarative_base()

class User(Base):
    __tablename__ = 'users'
    
    id = Column(Integer, primary_key=True)
    email = Column(String(120), unique=True, nullable=False)
    username = Column(String(150), unique=True, nullable=False)
    password = Column(String(60), nullable=False)
    
# Create SQLite engine and session
engine = create_engine('sqlite:///users.db', connect_args={"check_same_thread": False})
Base.metadata.create_all(engine)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
db_session = SessionLocal()

# =======================
# Logging Configuration
# =======================

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    filename='fashion_bot_app.log',
                    filemode='a')

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)

logging.getLogger('').addHandler(console_handler)
logger = logging.getLogger(__name__)

# =======================
# Initialize FashionBot
# =======================

fashion_bot = get_fashion_bot()
logger.info("FashionBot instance created")

def start_url_finder():
    url_finder = URLFinder()
    search_query = "Pakistani women fashion brands"  # or any appropriate search term
    url_finder.find_urls(search_query)
    logger.info("URL Finder started")

def initialize_url_finder():
    url_finder_thread = threading.Thread(target=start_url_finder)
    url_finder_thread.daemon = True
    url_finder_thread.start()
    logger.info("URL Finder thread initialized")

# Initialize the URLFinder
initialize_url_finder()

# =======================
# Streamlit Configuration
# =======================

st.set_page_config(
    page_title="Fashion Brand Query Bot",
    page_icon="🛍️",
    layout="wide"
)
logger.info("Streamlit page configured")

# =======================
# Authentication Functions
# =======================

def hash_password(password: str) -> bytes:
    return bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())

def verify_password(password: str, hashed: bytes) -> bool:
    return bcrypt.checkpw(password.encode('utf-8'), hashed)

def register_user(email: str, username: str, password: str) -> bool:
    existing_user = db_session.query(User).filter((User.email == email) | (User.username == username)).first()
    if existing_user:
        return False
    hashed_pw = hash_password(password)
    new_user = User(email=email, username=username, password=hashed_pw)
    db_session.add(new_user)
    db_session.commit()
    return True

def authenticate_user(email: str, password: str) -> bool:
    user = db_session.query(User).filter(User.email == email).first()
    if user and verify_password(password, user.password):
        return True
    return False

# =======================
# Streamlit App Layout
# =======================

def main():
    # App Header
    st.title("Fashion Brand Query Bot")
    st.write("Welcome to the Fashion Brand Query Bot. Please log in or register to continue.")
    
    # Initialize session state
    if 'authenticated' not in st.session_state:
        st.session_state['authenticated'] = False
    if 'user_email' not in st.session_state:
        st.session_state['user_email'] = ''
    if 'conversation' not in st.session_state:
        st.session_state['conversation'] = []
    
    # Display Login or Registration Forms
    if not st.session_state['authenticated']:
        auth_option = st.selectbox("Login or Register", ["Login", "Register"])
        
        if auth_option == "Register":
            st.subheader("Create an Account")
            with st.form("register_form"):
                reg_email = st.text_input("Email")
                reg_username = st.text_input("Username")
                reg_password = st.text_input("Password", type="password")
                reg_confirm_password = st.text_input("Confirm Password", type="password")
                reg_submit = st.form_submit_button("Register")
            
            if reg_submit:
                if not reg_email or not reg_username or not reg_password or not reg_confirm_password:
                    st.error("Please fill out all fields.")
                elif reg_password != reg_confirm_password:
                    st.error("Passwords do not match.")
                else:
                    success = register_user(reg_email, reg_username, reg_password)
                    if success:
                        st.success("Registration successful! You can now log in.")
                        logger.info(f"New user registered: {reg_email}")
                    else:
                        st.error("User with this email or username already exists.")
                        logger.warning(f"Registration failed for email: {reg_email}")
        
        elif auth_option == "Login":
            st.subheader("Login to Your Account")
            with st.form("login_form"):
                login_email = st.text_input("Email")
                login_password = st.text_input("Password", type="password")
                login_submit = st.form_submit_button("Login")
            
            if login_submit:
                if not login_email or not login_password:
                    st.error("Please enter both email and password.")
                else:
                    authenticated = authenticate_user(login_email, login_password)
                    if authenticated:
                        st.session_state['authenticated'] = True
                        st.session_state['user_email'] = login_email
                        st.success("Logged in successfully!")
                        logger.info(f"User logged in: {login_email}")
                    else:
                        st.error("Invalid email or password.")
                        logger.warning(f"Failed login attempt for email: {login_email}")
    
    else:
        # Logged-in User Interface
        st.sidebar.success(f"Logged in as {st.session_state['user_email']}")

        if st.sidebar.button("Logout"):
            # Reset session state for user logout
            st.session_state['authenticated'] = False
            st.session_state['user_email'] = ''
            st.session_state['conversation'] = []
            st.sidebar.success("Logged out successfully!")
            logging.info(f"User logged out.")

            # Use st.experimental_set_query_params to force rerun after logout
            st.experimental_set_query_params(logged_out="true")
        
        st.header("Ask About Fashion Brands")
        user_input = st.chat_input("Type your question here...")
        
        if user_input:
            logger.info(f"Received user input from {st.session_state['user_email']}: {user_input}")
            st.session_state.conversation.append({"role": "user", "content": user_input})
            with st.spinner("Generating response..."):
                response = fashion_bot.get_response(user_input)
            st.session_state.conversation.append({"role": "bot", "content": response})
            logger.info(f"Response generated for user {st.session_state['user_email']}")
        
        # Display the conversation
        for message in st.session_state.conversation:
            if message["role"] == "user":
                st.chat_message("user").write(message["content"])
            else:
                st.chat_message("assistant").write(message["content"])
                
                # Check if the message content is a dictionary
                if isinstance(message["content"], dict):
                    if "images" in message["content"]:
                        # Ensure images is a list before iterating
                        for img_url in message["content"]["images"]:
                            if isinstance(img_url, str):  # Check if img_url is a string
                                st.image(img_url, use_column_width=True)  # Display the image
                                logger.debug(f"Image displayed: {img_url}")
                    else:
                        logger.warning("No 'images' key found in message['content']")
                else:
                    logger.error("message['content'] is not a dictionary: ", message["content"])
        
        # Clear chat history button
        if st.button("Clear Chat History"):
            st.session_state["conversation"] = []
            st.success("Chat history cleared.")
            logger.info(f"Chat history cleared for user {st.session_state['user_email']}")

if __name__ == "__main__":
    main()
