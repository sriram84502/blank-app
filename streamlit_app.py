import streamlit as st
from transformers import MarianTokenizer, TFMarianMTModel, GPT2Tokenizer, TFGPT2LMHeadModel

translation_models = {
    'French': 'Helsinki-NLP/opus-mt-en-fr',
    'German': 'Helsinki-NLP/opus-mt-en-de',
    'Spanish': 'Helsinki-NLP/opus-mt-en-es'
}
translation_tokenizers = {}
translation_models_tf = {}
for lang, model_name in translation_models.items():
    translation_tokenizers[lang] = MarianTokenizer.from_pretrained(model_name)
    translation_models_tf[lang] = TFMarianMTModel.from_pretrained(model_name)

style_model_name = 'gpt2-medium'
style_tokenizer = GPT2Tokenizer.from_pretrained(style_model_name)
style_model = TFGPT2LMHeadModel.from_pretrained(style_model_name)

def translate(text, model, tokenizer):
    inputs = tokenizer(text, return_tensors="tf", padding=True)
    translated_tokens = model.generate(**inputs)
    translated_text = tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)
    return translated_text[0]

def style_transfer(text, model, tokenizer, style_prompt):
    inputs = tokenizer.encode(style_prompt + text, return_tensors="tf")
    outputs = model.generate(inputs, max_length=200, num_return_sequences=1)
    styled_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return styled_text

st.title("Machine Translation with Style Transfer Chat")

target_language = st.selectbox("Select target language:", list(translation_models.keys()))

if 'messages' not in st.session_state:
    st.session_state.messages = []

def add_message(role, text):
    st.session_state.messages.append({"role": role, "text": text})

for message in st.session_state.messages:
    if message["role"] == "user":
        st.text_area("You", value=message["text"], height=100, disabled=True)
    else:
        st.text_area("Bot", value=message["text"], height=100, disabled=True)

with st.form(key='chat_form', clear_on_submit=True):
    user_input = st.text_input("You:")
    submit_button = st.form_submit_button(label='Send')

if submit_button and user_input:
    add_message("user", user_input)
    
    translation_model = translation_models_tf[target_language]
    translation_tokenizer = translation_tokenizers[target_language]
    translated_text = translate(user_input, translation_model, translation_tokenizer)
    
    style_prompt = "translate to formal: "
    styled_text = style_transfer(translated_text, style_model, style_tokenizer, style_prompt)
    
    add_message("bot", styled_text)
    st.experimental_rerun()
