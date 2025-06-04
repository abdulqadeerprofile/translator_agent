import os
import asyncio
import streamlit as st
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from openai import AsyncOpenAI
from agents import Agent, Runner, OpenAIChatCompletionsModel

# Load .env keys
load_dotenv()
gemini_api_key = os.getenv('GEMINI_API_KEY')

# OpenAI (Gemini) client
client = AsyncOpenAI(
    api_key=gemini_api_key,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
)

# Pydantic response model for translations
class TranslationResponse(BaseModel):
    source_language: str = Field(description="The language of the input text")
    target_language: str = Field(description="The language the text was translated to")
    translated_text: str = Field(description="The word-by-word translated text")
    original_text: str = Field(description="The original input text")

# Agent setup for translation
translation_agent = Agent(
    name="Multilanguage Translator",
    instructions="""
    You are a professional word-by-word translator.
    
    Your task is to translate text between languages, focusing on accurate word-by-word translation:
    
    1. Translate the input text from the source language to the target language
    2. Maintain a word-by-word approach rather than contextual/idiomatic translation
    3. Always provide the translation even if the result might sound unnatural
        
    Return the translation in the following fields:
    - source_language: The language of the input text
    - target_language: The language the text was translated to
    - translated_text: The word-by-word translated text
    - original_text: The original input text
    
    If the input is not clear, identify the apparent language and respond professionally.
    """,
    model=OpenAIChatCompletionsModel(model="gemini-2.0-flash", openai_client=client),
    output_type=TranslationResponse
)

# Async runner with improved error handling
async def get_translation(text: str, source_lang: str, target_lang: str):
    try:
        query = f"Translate from {source_lang} to {target_lang}: {text}"
        result = await Runner.run(translation_agent, query)
        return result.final_output
    except Exception as e:
        st.error(f"Error during translation: {str(e)}")
        return TranslationResponse(
            source_language=source_lang,
            target_language=target_lang,
            translated_text="Sorry, an error occurred during translation. Please try again.",
            original_text=text
        )

# --- Streamlit UI ---
st.set_page_config(page_title="Multilanguage Translator", page_icon="🌐", layout="wide")

# Custom CSS for chat-like interface with improved responsiveness
st.markdown("""
    <style>
    .main {
        background-color: #f9f9f9;
    }
    .stApp {
        max-width: 1200px;
        margin: 0 auto;
    }
    h1, h2, h3 {
        color: #1a3a5f;
        font-family: 'Arial', sans-serif;
    }
    .header-container {
        display: flex;
        align-items: center;
        border-bottom: 2px solid #1a3a5f;
        padding-bottom: 10px;
        margin-bottom: 20px;
        flex-wrap: wrap;
    }
    .logo {
        margin-right: 15px;
    }
    .chat-container {
        display: flex;
        flex-direction: column;
        height: 60vh;
        overflow-y: auto;
        padding: 10px;
        background-color: white;
        border-radius: 10px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        margin-bottom: 20px;
    }
    .user-bubble {
        align-self: flex-end;
        background-color: #0084ff;
        color: white;
        border-radius: 18px;
        padding: 10px 15px;
        margin: 5px;
        max-width: 80%;
        box-shadow: 0 1px 2px rgba(0,0,0,0.1);
        word-wrap: break-word;
    }
    .bot-bubble {
        align-self: flex-start;
        background-color: #f0f2f5;
        color: black;
        border-radius: 18px;
        padding: 10px 15px;
        margin: 5px;
        max-width: 80%;
        box-shadow: 0 1px 2px rgba(0,0,0,0.1);
        word-wrap: break-word;
    }
    .input-container {
        display: flex;
        flex-direction: row;
        background-color: white;
        padding: 10px;
        border-radius: 10px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }
    .language-selector {
        display: flex;
        align-items: center;
        justify-content: center;
        padding: 10px;
        background-color: white;
        border-radius: 10px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        margin-bottom: 20px;
        flex-wrap: wrap;
    }
    .swap-button {
        margin: 0 10px;
        height: 38px;
    }
    .footer {
        margin-top: 30px;
        padding-top: 10px;
        border-top: 1px solid #ddd;
        font-size: 0.8em;
        color: #666;
        text-align: center;
    }
    .developer-credit {
        font-weight: bold;
        color: #1a3a5f;
        text-align: center;
        margin: 10px 0;
        padding: 10px;
        background-color: #f0f2f5;
        border-radius: 10px;
    }
    
    /* Responsive adjustments */
    @media (max-width: 768px) {
        .stApp {
            max-width: 100%;
            padding: 5px;
        }
        .chat-container {
            height: 50vh;
        }
        .user-bubble, .bot-bubble {
            max-width: 90%;
            padding: 8px 12px;
        }
        .header-container {
            flex-direction: column;
            text-align: center;
        }
        .logo {
            margin-right: 0;
            margin-bottom: 10px;
        }
        h1 {
            font-size: 1.5rem;
        }
        .input-container {
            flex-direction: column;
        }
        .language-selector {
            flex-direction: column;
        }
        .developer-credit {
            font-size: 0.9em;
        }
    }
    </style>
    """, unsafe_allow_html=True)

# Header
st.markdown("""
    <div class="header-container">
        <div class="logo">
            <img src="https://cdn-icons-png.flaticon.com/512/3898/3898082.png" width="50">
        </div>
        <div>
            <h1>Multilanguage Translator</h1>
            <p style="margin-top:-15px;color:#666;">Word-by-word translation between languages</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

# Initialize chat history if not exists
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Language selection
languages = [
    "English", "Spanish", "French", "German", "Chinese", "Japanese", 
    "Russian", "Arabic", "Hindi", "Urdu", "Portuguese", "Italian", 
    "Dutch", "Korean", "Turkish", "Swedish", "Polish"
]

# Language selection with swap button - improved responsiveness
st.markdown('<div class="language-selector">', unsafe_allow_html=True)

# Use different column ratios based on screen size
col1, col2, col3 = st.columns([2, 1, 2], gap="small")

# Default language selections
if "source_lang" not in st.session_state:
    st.session_state.source_lang = "English"
if "target_lang" not in st.session_state:
    st.session_state.target_lang = "Urdu"

with col1:
    source_lang = st.selectbox("From:", languages, index=languages.index(st.session_state.source_lang))
    st.session_state.source_lang = source_lang

with col2:
    if st.button("🔄 Swap", key="swap_button"):
        st.session_state.source_lang, st.session_state.target_lang = st.session_state.target_lang, st.session_state.source_lang
        st.rerun()

with col3:
    target_lang = st.selectbox("To:", languages, index=languages.index(st.session_state.target_lang))
    st.session_state.target_lang = target_lang

st.markdown('</div>', unsafe_allow_html=True)

# Display chat history
st.markdown('<div class="chat-container" id="chat-container">', unsafe_allow_html=True)
for message in st.session_state.chat_history:
    if message["role"] == "user":
        st.markdown(f'<div class="user-bubble">{message["content"]}</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="bot-bubble">{message["content"]}</div>', unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)

# Input area - improved for responsiveness
st.markdown('<div class="input-container">', unsafe_allow_html=True)

# Adjust column ratios for better mobile experience
if st.session_state.get("is_mobile", False):
    col1, col2 = st.columns([1], gap="small")
    with col1:
        user_input = st.text_area("Enter text to translate:", height=80, key="user_input", label_visibility="collapsed")
    with col1:
        submit_button = st.button("Send", use_container_width=True)
else:
    col1, col2 = st.columns([6, 1], gap="small")
    with col1:
        user_input = st.text_area("Enter text to translate:", height=80, key="user_input", label_visibility="collapsed")
    with col2:
        submit_button = st.button("Send", use_container_width=True)

st.markdown('</div>', unsafe_allow_html=True)

# Process the translation
if submit_button and user_input:
    # Add user message to chat history
    st.session_state.chat_history.append({"role": "user", "content": user_input})
    
    # Get translation
    with st.spinner(f"Translating from {source_lang} to {target_lang}..."):
        response = asyncio.run(get_translation(user_input, source_lang, target_lang))
        
    # Format the response with attribution
    formatted_response = f"As Abdul Qadeer Trained Me The Translation Of This Line Will Be: {response.translated_text}"
    
    # Add translation to chat history
    st.session_state.chat_history.append({"role": "assistant", "content": formatted_response})
    
    # Use a different approach to reset the input field
    st.session_state["reset_input"] = True
    
    # Rerun to update the UI
    st.rerun()

# Reset input field using JavaScript instead of directly modifying session state
if st.session_state.get("reset_input", False):
    st.session_state["reset_input"] = False
    st.markdown("""
    <script>
        const textareas = window.parent.document.querySelectorAll('textarea');
        for(let textarea of textareas) {
            textarea.value = '';
        }
    </script>
    """, unsafe_allow_html=True)

# Auto-scroll to bottom of chat (using JavaScript)
st.markdown("""
<script>
    function scrollToBottom() {
        const chatContainer = document.querySelector('#chat-container');
        if (chatContainer) {
            chatContainer.scrollTop = chatContainer.scrollHeight;
        }
    }
    scrollToBottom();
</script>
""", unsafe_allow_html=True)

# Footer with attribution
st.markdown("""
<div class="footer">
    <div class="developer-credit">DEVELOPED BY ABDUL QADEER</div>
    <p>This tool provides word-by-word translations between multiple languages.</p>
</div>
""", unsafe_allow_html=True)

# Detect mobile based on browser size
st.markdown("""
<script>
    // Check if device is mobile
    function checkMobile() {
        return window.innerWidth <= 768;
    }
    
    // Store the result in sessionStorage
    if (!window.sessionStorage.getItem('ran_once')) {
        window.sessionStorage.setItem('is_mobile', checkMobile());
        window.sessionStorage.setItem('ran_once', 'true');
        // Force a page reload to ensure Streamlit picks up the sessionStorage
        if (window.sessionStorage.getItem('is_mobile') === 'true') {
            const params = new URLSearchParams(window.location.search);
            params.set('is_mobile', 'true');
            window.location.search = params.toString();
        }
    }
</script>
""", unsafe_allow_html=True)

# Check URL parameters for mobile flag
import urllib.parse
if st.query_params.get("is_mobile", ["false"])[0] == "true":
    st.session_state["is_mobile"] = True