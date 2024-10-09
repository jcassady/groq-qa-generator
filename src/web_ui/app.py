import streamlit as st

# Ensure page config is called first
st.set_page_config(page_title="Groq QA Generator", layout="wide")

# Default content for the text input box
default_text = '''
Alice was beginning to get very tired of sitting by her sister on the bank, and of having nothing to do: once or twice she had peeped into the book her sister was reading, but it had no pictures or conversations in it, 'and what is the use of a book,' thought Alice 'without pictures or conversations?'
'''

# Sidebar Configuration (dropdown for LLM model, slider for temperature)
st.sidebar.markdown("### ⚙️ Configuration")
with st.sidebar.expander("Settings (Click to Configure)", expanded=False):
    llm_model = st.selectbox("Choose Model", ["LLaMA 2", "GPT-4"])
    temperature = st.slider("Temperature", min_value=0.1, max_value=0.9, value=0.5, step=0.1, format="%.1f")

# Theme toggle switch
theme_choice = st.sidebar.checkbox("Toggle Light/Dark Theme", value=False)

# Apply theme-based styles
if theme_choice:
    st.markdown('''
        <style>
        body {
            background-color: #f9f9f9;
        }
        .text-box, .chat-container {
            background-color: #ffffff;
            border-radius: 10px;
            padding: 20px;
            box-shadow: 0px 4px 10px rgba(0, 0, 0, 0.1);
        }
        .question-bubble, .answer-bubble {
            padding: 15px;
            border-radius: 10px;
            margin-bottom: 10px;
            font-size: 16px;
            font-family: 'Inter', 'Arial', sans-serif;
        }
        .question-bubble {
            background-color: #007bff;
            color: white;
        }
        .answer-bubble {
            background-color: #cccccc;
            color: black;
        }
        </style>
    ''', unsafe_allow_html=True)
else:
    st.markdown('''
        <style>
        body {
            background-color: #1e1e1e;
        }
        .text-box, .chat-container {
            background-color: #2b2b2b;
            border-radius: 10px;
            padding: 20px;
            box-shadow: 0px 4px 10px rgba(0, 0, 0, 0.5);
        }
        .question-bubble, .answer-bubble {
            padding: 15px;
            border-radius: 10px;
            margin-bottom: 10px;
            font-size: 16px;
            font-family: 'Inter', 'Arial', sans-serif;
        }
        .question-bubble {
            background-color: #4caf50;
            color: white;
        }
        .answer-bubble {
            background-color: #3a3a3a;
            color: white;
        }
        </style>
    ''', unsafe_allow_html=True)

# Layout: Text input on the left and chat dialog on the right
col1, col2 = st.columns([1, 1])

# Left pane: text input box
with col1:
    # Display the text input box for user to enter text
    st.markdown("<div class='text-box'>", unsafe_allow_html=True)
    user_input = st.text_area("Text Input", value=default_text, height=250, label_visibility="collapsed")
    st.markdown("</div>", unsafe_allow_html=True)

# Right pane: chat Q&A dialog
with col2:
    button_pressed = st.button("Generate Questions and Answers")

    if button_pressed:
        st.markdown("""
            <div class="chat-container">
                <div class='question-bubble'>
                    What is the theme of this book?
                </div>
                <div class='answer-bubble'>
                    The main theme revolves around curiosity and wonder.
                </div>
            </div>
        """, unsafe_allow_html=True)

