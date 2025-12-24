import json
import cv2
import pytesseract
import spacy
import streamlit as st
from PIL import Image
from openai import OpenAI
from dotenv import load_dotenv
import os
import re

load_dotenv()

# ------------------------------------
# NLP LOAD (SAFE FOR CLOUD)
# ------------------------------------
try:
    nlp = spacy.load("en_core_web_sm")
except Exception:
    nlp = spacy.blank("en")

# ------------------------------------
# NLP INTELLIGENCE
# ------------------------------------

INTENT_KEYWORDS = {
    "Summarizer": ["summarize", "summary", "shorten", "tl;dr"],
    "Code Assistant": ["code", "bug", "error", "fix", "function"],
    "Tutor": ["teach", "learn", "explain", "how", "why"],
    "Debate": ["agree", "disagree", "argue", "opinion"],
}

ANGER_WORDS = ["angry", "hate", "worst", "trash", "useless"]
FRUSTRATION_WORDS = ["confusing", "hard", "stuck", "not working"]

#LOCAL RESPONSES
#---------------------------------
LOCAL_RESPONSES = {
    "hello": "Hello! How can I help you today?",
    "hi": "Hi there! What would you like to work on?",
    "help": (
        "You can:\n"
        "- Ask general questions\n"
        "- Upload an image and ask about it\n"
        "- Switch modes and personalities from the sidebar"
    ),
    "who are you": "I’m an intelligent AI assistant with multiple modes and personalities.",
    "who is your owner": "Mister zayan chaus is my owner"
}

def get_local_response(user_text: str):
    text = user_text.lower().strip()
    for trigger, response in LOCAL_RESPONSES.items():
        if trigger in text:
            return response
    return None


def sanitize_ocr_text(text: str) -> str:
    text = text.encode("utf-8", "ignore").decode("utf-8")
    text = re.sub(r"[^\x20-\x7E\n]", " ", text)  # remove non-ASCII
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def detect_intent(text):
    text = text.lower()
    for mode, words in INTENT_KEYWORDS.items():
        if any(w in text for w in words):
            return mode
    return None

def detect_sentiment(text):
    text = text.lower()
    if any(w in text for w in ANGER_WORDS):
        return "angry"
    if any(w in text for w in FRUSTRATION_WORDS):
        return "frustrated"
    return "neutral"

def tone_instruction(sentiment):
    return {
        "angry": "Remain calm, professional, and de-escalate the situation.",
        "frustrated": "Be patient, reassuring, and explain clearly step by step.",
        "neutral": "Maintain a clear, professional, and helpful tone."
    }[sentiment]


# ------------------------------------

PERSONALITIES = {
    "Helpful Assistant": "You are a helpful, polite, and concise AI assistant.",
    "Pirate": "You are a salty pirate captain. Speak in pirate slang. Be adventurous but helpful.",
    "Sarcastic Tech Support":(
    "You are a bored but competent tech support engineer. "
    "Respond in clear English sentences. "
    "Use mild sarcasm, but always provide a helpful and complete answer."
),
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
        "prompt": (
        "Engage in a structured debate. "
        "Clearly state assumptions, then present logical counterarguments. "
        "Write in fluent, complete English sentences. "
        "Avoid meta commentary and internal reasoning tokens."
    )
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

# ------------------------------------
# OCR UTILS (UNCHANGED)
# ------------------------------------

def clean_text(text: str) -> str:
    doc = nlp(text)
    return " ".join(sent.text for sent in doc.sents)

# ------------------------------------
# APP
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
            st.rerun()

    st.title(f"🤖 {selected_mode} — {selected_personality}")

    # Image Upload
    uploaded_image = st.file_uploader(
        "Upload an image (notes, screenshot, diagram)",
        type=["png", "jpg", "jpeg"]
    )

    image_context = None

    if uploaded_image:
        image = Image.open(uploaded_image)
        st.image(image, caption="Uploaded Image", use_column_width=True)
        raw_text = pytesseract.image_to_string(image)
        image_context = sanitize_ocr_text(clean_text(raw_text))


        if not image_context.strip():
            st.warning("No readable text detected in image.")

    # Display Chat
    for msg in messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Model Call
    def call_model(user_prompt, image_context=None):
        detected_intent = detect_intent(user_prompt)
        sentiment = detect_sentiment(user_prompt)

        active_mode = detected_intent if detected_intent in MODES else selected_mode

        system_prompt = f"""
You are an AI assistant.

{PERSONALITIES[selected_personality]}
MODE: {active_mode}
{MODES[active_mode]["prompt"]}

TONE:
{tone_instruction(sentiment)}

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

    # User Input
    prompt = st.chat_input("Ask a question (about the image or generally)")

    if prompt:
        messages.append({"role": "user", "content": prompt})

        local_answer = get_local_response(prompt)

        with st.chat_message("assistant"):
            if local_answer:
                answer = local_answer
            else:
                answer = call_model(prompt, image_context)

            st.markdown(answer)

        messages.append({"role": "assistant", "content": answer})
        st.rerun()

    # Export
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
