import streamlit as st
from streamlit_chat import message as st_message
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


@st.cache_resource
def get_models():
    tokenizer = AutoTokenizer.from_pretrained("facebook/blenderbot_small-90M")
    model = AutoModelForSeq2SeqLM.from_pretrained("facebook/blenderbot_small-90M")
    return tokenizer, model


if "history" not in st.session_state:
    st.session_state.history = []


st.title("Blenderbot - Conversational Chatbot")


def generate_answer():
    tokenizer, model = get_models()
    user_message = st.session_state.input_text
    if user_message.strip():
        try:
            inputs = tokenizer(user_message, return_tensors="pt")
            result = model.generate(**inputs)
            bot_response = tokenizer.decode(result[0], skip_special_tokens=True)

            st.session_state.history.append({"message": user_message, "is_user": True})
            st.session_state.history.append({"message": bot_response, "is_user": False})

            st.session_state.input_text = ""
        except Exception as e:
            st.error(f"Error generating response: {e}")


st.text_input(
    "Type your message below:",
    key="input_text",
    on_change=generate_answer,
)

if st.session_state.history:
    for chat in st.session_state.history:
        st_message(**chat)
else:
    st.write("Start the conversation by typing a message!")
