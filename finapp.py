import streamlit as st
import pickle
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load LSTM model and tokenizer
lstm_model = load_model('v1.5/text_generator_model_v1.keras', compile=False)
with open('v1.5/tokenizer.pickle', 'rb') as f:
    lstm_tokenizer = pickle.load(f)

# Define maximum sequence length for padding
max_sequence_len = 2000

# Text generation function for LSTM model
def generate_text_lstm(seed_text, next_words=10, model=None, tokenizer=None, temperature=1.0, diversity=0.8):
    for _ in range(next_words):
        token_list = tokenizer.texts_to_sequences([seed_text])[0]
        token_list = pad_sequences([token_list], maxlen=max_sequence_len - 1, padding='pre')
        predicted = model.predict(token_list, verbose=0)
        
        # Apply temperature
        predicted = np.asarray(predicted).astype("float64")
        predicted = np.log(predicted + 1e-8) / temperature
        exp_preds = np.exp(predicted)
        predicted = exp_preds / np.sum(exp_preds)

        # Determine predicted_index based on diversity
        if diversity < 1.0:
            predicted_index = np.random.choice(range(len(predicted[0])), p=predicted[0])
        else:
            predicted_index = np.argmax(predicted, axis=1)[0]  # Use argmax for greedy selection if diversity is 1.0

        # Find the word corresponding to the predicted index
        output_word = ""
        for word, index in tokenizer.word_index.items():
            if index == predicted_index:
                output_word = word
                break
        seed_text += " " + output_word

    return seed_text


# Custom CSS for styling
st.markdown("""
    <style>
        body { background-color: #f5f5f5; font-family: Arial, sans-serif; }
        .main-header { font-size: 2.5em; font-weight: bold; color: #0a4f54; text-align: center; margin-bottom: 20px; }
        .subtitle { font-size: 1.2em; font-weight: 500; color: #4d4d4d; text-align: center; margin-bottom: 20px; }
        .stButton>button { background-color: #0a4f54; color: white; font-size: 1.2em; padding: 10px 20px; border-radius: 8px; transition: background-color 0.3s; }
        .stButton>button:hover { background-color: #05403e; }
        .sidebar .sidebar-content { background-color: #eef7f6; padding: 20px; border-radius: 8px; }
        .slider-label { font-weight: bold; color: #0a4f54; }
        .text-input { font-size: 1.1em; padding: 15px; border-radius: 8px; border: 1px solid #ddd; background-color: #fafafa; }
        .output-text { font-size: 1.1em; padding: 15px; border-radius: 8px; border: 1px solid #ddd; background-color: #f4f4f4; }
    </style>
""", unsafe_allow_html=True)

# Sidebar for customization options
st.sidebar.title("Configuration")
text_length = st.sidebar.slider("Text Length", min_value=5, max_value=100, value=20)
temperature = st.sidebar.slider("Temperature", min_value=0.5, max_value=2.0, value=1.0, step=0.1)
diversity = st.sidebar.slider("Diversity", min_value=0.1, max_value=1.0, value=0.8, step=0.1)
sampling_method = st.sidebar.selectbox("Sampling Method", ("Greedy", "Random", "Top-K", "Top-P"))
top_k = st.sidebar.slider("Top-K Sampling", min_value=1, max_value=100, value=50)
top_p = st.sidebar.slider("Top-P Sampling (Nucleus)", min_value=0.5, max_value=1.0, value=0.9)

# Set up the Streamlit UI
st.markdown('<p class="main-header">LSTM Text Generation App</p>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Generate custom text responses using an LSTM model</p>', unsafe_allow_html=True)

# Input text from user
user_input = st.text_input("Enter Seed Text", "", key="input_text", placeholder="Type your question or seed text here...")

# Generate text based on user input
if st.button("Generate Response"):
    if user_input:
        generated_text = generate_text_lstm(user_input, next_words=text_length, model=lstm_model, tokenizer=lstm_tokenizer, temperature=temperature, diversity=diversity)
        
        st.text_area("Generated Response", generated_text, height=200, max_chars=1000, key="output_text", placeholder="Response will appear here...")
        
        # Add download option
        st.download_button("Download Response", generated_text, file_name="generated_text.txt", mime="text/plain")
    else:
        st.warning("Please enter a seed text to generate a response.")

# Optional information about models and settings
with st.expander("About the Models and Settings"):
    st.write("""
        **Settings**:
        - **Temperature**: Controls creativity. Lower values (e.g., 0.7) make the model more focused, higher values (e.g., 1.2) make it more creative.
        - **Diversity**: Adjusts how varied the generated text should be. Lower diversity makes output more predictable.
        - **Sampling Methods**:
            - **Greedy**: Chooses the word with the highest probability each time, resulting in deterministic output.
            - **Random**: Chooses words based on probability distribution, introducing randomness.
            - **Top-K**: Limits the next-word predictions to the top-K most likely words.
            - **Top-P (Nucleus)**: Limits predictions to a cumulative probability (e.g., 0.9) of the most likely words.
    """)

st.markdown("<br><br>", unsafe_allow_html=True)
