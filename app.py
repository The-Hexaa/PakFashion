import streamlit as st
from main import get_response

# Streamlit app configuration
st.set_page_config(
    page_title="Fashion Brand Query Bot",
    page_icon="🛍️",
    layout="wide"
)

# Streamlit App Header
st.title("Fashion Brand Query Bot")
st.write("Ask me about different fashion items from Khaadi, Sapphire, Generation, and Rangja!")

# User input section
user_input = st.chat_input("Type your question here...")

# Display conversation history
if "conversation" not in st.session_state:
    st.session_state["conversation"] = []

# Handle user input and generate response
if user_input:
    st.session_state.conversation.append({"role": "user", "content": user_input})
    with st.spinner("Generating response..."):
        response = get_response(user_input)
    st.session_state.conversation.append({"role": "bot", "content": response})

# Display the conversation
for message in st.session_state.conversation:
    if message["role"] == "user":
        st.chat_message("user").write(message["content"])
    else:
        st.chat_message("assistant").write(message["content"])
        # Check for images or additional information in the response
        if "images" in message["content"]:
            for img_url in message["content"]["images"]:
                st.image(img_url, use_column_width=True)  # Display the image

# Clear chat history button
if st.button("Clear Chat History"):
    st.session_state["conversation"] = []
