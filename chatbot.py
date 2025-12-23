import os
import json
import cv2
import pytesseract
import spacy
import streamlit as st
from PIL import Image
from openai import OpenAI
from dotenv import load_dotenv
from spacy.cli import download
import os

load_dotenv()


# Load NLP
# ------------------------------------
SPACY_MODEL_DIR = os.path.join(os.getcwd(), "spacy_model")

try:
    # Try loading the full English model
    nlp = spacy.load("en_core_web_sm")
except OSError:
    # Fall back to a blank English model (minimal tokenizer)
    nlp = spacy.blank("en")

# Prompt Registries
# ------------------------------------

PERSONALITIES = {
    "Helpful Assistant": "You are a helpful, polite, and concise AI assistant.",
    "Pirate": "You are a salty pirate captain. Speak in pirate slang. Be adventurous but helpful.",
    "Sarcastic Tech Support": "You are a bored, sarcastic tech support agent. You help, with snark.",
    "Shakespearean Poet": "You speak in Shakespearean English.",
    "Motivational Coach": "You are a high-energy motivational coach."
}

MODES = {
    "Deep Reasoning": {
        "description": "Step-by-step logical reasoning.",
        "prompt": "Explain reasoning step by step and state assumptions."
    },
    "Code Assistant": {
        "description": "High-quality code with explanations.",
        "prompt": "Write clean, idiomatic code and explain decisions."
    },
    "Tutor": {
        "description": "Teach through guided questions.",
        "prompt": "Teach using hints and questions, not direct answers."
    },
    "Debate": {
        "description": "Challenge assumptions and argue logically.",
        "prompt": "Challenge assumptions and present counterarguments."
    },
    "Summarizer": {
        "description": "Concise summaries.",
        "prompt": "Summarize clearly using bullet points."
    }
}

MODELS = [
    "llama-3.3-70b-versatile",
    "llama-3.1-8b-instant"
]


# OCR Utilities
# ------------------------------------

def extract_text_from_image(image: Image.Image) -> str:
    img = cv2.cvtColor(
        cv2.imread(image.filename),
        cv2.COLOR_BGR2GRAY
    )
    text = pytesseract.image_to_string(img)
    return text.strip()

def clean_text(text: str) -> str:
    doc = nlp(text)
    return " ".join(sent.text for sent in doc.sents)


# App
# ------------------------------------

def chat_ui():
    st.set_page_config(
        page_title="AI Multi-Mode Chatbot (Image Aware)",
        page_icon="🤖",
        layout="wide"
    )

    if "messages" not in st.session_state:
        st.session_state.messages = []

    messages = st.session_state.messages


    # Sidebar
    # ------------------------------------
    with st.sidebar:
        st.title("Configuration")

        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            st.error("Missing GROQ_API_KEY")
            st.stop()

        selected_model = st.selectbox("Model", MODELS)
        selected_personality = st.selectbox("Personality", PERSONALITIES.keys())
        selected_mode = st.selectbox("Mode", MODES.keys())
        st.caption(MODES[selected_mode]["description"])





        st.divider()

        if st.button("Clear Chat"):
            st.session_state.messages.clear()
            st.session_state.image_context = None
            st.rerun()

    st.title(f"🤖 {selected_mode} — {selected_personality}")


    # Image Upload
    # ------------------------------------
    uploaded_image = st.file_uploader(
        "Upload an image (notes, screenshot, diagram)",
        type=["png", "jpg", "jpeg"]
    )

    image_context = None

    if uploaded_image:
        image = Image.open(uploaded_image)
        st.image(image, caption="Uploaded Image", use_column_width=True)

        raw_text = pytesseract.image_to_string(image)
        image_context = clean_text(raw_text)

        if not image_context.strip():
            st.warning("No readable text detected in image.")

    # ------------------------------------
    # Display Chat
    # ------------------------------------
    for msg in messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # ------------------------------------
    # Model Call
    # ------------------------------------
    def call_model(user_prompt, image_context=None):
        system_prompt = f"""
You are an AI assistant.

{PERSONALITIES[selected_personality]}
{MODES[selected_mode]["prompt"]}

Rules:
- If image context is provided, answer ONLY from it.
- If the answer is not present, say so clearly.
"""

        if image_context:
            user_prompt = f"""
Image Content:
{image_context}

Question:
{user_prompt}
"""

        client = OpenAI(
            api_key=api_key,
            base_url="https://api.groq.com/openai/v1"
        )

        response = client.chat.completions.create(
            model=selected_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
        )

        return response.choices[0].message.content

    # ------------------------------------
    # User Input
    # ------------------------------------
    prompt = st.chat_input("Ask a question (about the image or generally)")

    if prompt:
        messages.append({"role": "user", "content": prompt})

        with st.chat_message("assistant"):
            answer = call_model(prompt, image_context)
            st.markdown(answer)

        messages.append({"role": "assistant", "content": answer})
        st.rerun()


    # Export
    # ------------------------------------
    if messages:
        st.divider()

        md_export = "\n\n".join(
            f"**{m['role'].upper()}**:\n{m['content']}" for m in messages
        )

        st.download_button(
            "Download as Markdown",
            md_export,
            file_name="conversation.md"
        )

        st.download_button(
            "Download as JSON",
            json.dumps(messages, indent=2),
            file_name="conversation.json"
        )

chat_ui()
