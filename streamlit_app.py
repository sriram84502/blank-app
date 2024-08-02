import os
import logging
import warnings
import streamlit as st
import tensorflow as tf
from transformers import MarianTokenizer, TFMarianMTModel, GPT2Tokenizer, TFGPT2LMHeadModel
import psutil
from colorama import init, Fore, Style

init()

def log_memory_usage(stage):
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    logging.debug(f'{stage} - Memory Usage: {memory_info.rss / 1024 ** 2:.2f} MB')

# Suppress TensorFlow logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Suppress TensorFlow and Transformers logs
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("tensorflow").setLevel(logging.ERROR)
logging.basicConfig(level=logging.ERROR)

# Suppress specific warnings
warnings.filterwarnings("ignore", category=UserWarning, module='transformers')
warnings.filterwarnings("ignore", category=DeprecationWarning, module='tensorflow')

# Set up logging to file
logging.basicConfig(filename='app.log', level=logging.DEBUG)

def load_translation_model(language_code):
    model_name = f'Helsinki-NLP/opus-mt-en-{language_code}'
    tokenizer = MarianTokenizer.from_pretrained(model_name)
    model = TFMarianMTModel.from_pretrained(model_name)
    return tokenizer, model

def load_style_transfer_model():
    model_name = 'gpt2-medium'
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    model = TFGPT2LMHeadModel.from_pretrained(model_name)
    return tokenizer, model

try:
    st.title("Bilingual Stylistic Translator")

    language_options = {'French': 'fr', 'German': 'de', 'Spanish': 'es'}
    selected_language = st.selectbox("Select target language:", list(language_options.keys()))

    text = st.text_input("Enter the text to be translated and styled:")

    if text and selected_language:
        log_memory_usage('Before loading translation model')
        translation_tokenizer, translation_model = load_translation_model(language_options[selected_language])
        log_memory_usage('After loading translation model')

        inputs = translation_tokenizer(text, return_tensors="tf", padding=True)
        translated_tokens = translation_model.generate(**inputs)
        translated_text = translation_tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)[0]
        st.write("Translated Text:", translated_text)

        log_memory_usage('Before loading style transfer model')
        style_tokenizer, style_model = load_style_transfer_model()
        log_memory_usage('After loading style transfer model')

        style_prompt = "translate to formal: "
        style_inputs = style_tokenizer.encode(style_prompt + translated_text, return_tensors="tf")
        style_outputs = style_model.generate(style_inputs, max_length=200, num_return_sequences=1)
        styled_text = style_tokenizer.decode(style_outputs[0], skip_special_tokens=True)
        st.write("Styled Text:", styled_text)

except Exception as e:
    logging.exception("Exception occurred")
    st.error("An error occurred. Please check the logs for more details.")
