import streamlit as st
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq

# Function to get human-like message and score
def get_human_message(user_message):
    # Initialize Groq LLM
    llm = ChatGroq(   
        model_name="llama3-8b-8192",        # Groq-supported model
        temperature=0.3,
        max_tokens=512
    )

    # Prompt Template
    template = """
    You are an assistant that converts text into a natural human-like message.
    1. Rephrase the given text into a natural human message.
    2. Provide a human-likeness score from 0 to 100 (where 100 = perfectly human).
    
    Text: "{user_message}"

    Respond strictly in JSON format like this:
    {{
      "human_message": "...",
      "score": "..."
    }}
    """

    prompt = PromptTemplate(
        input_variables=["user_message"],
        template=template
    )

    # Generate response
    response = llm.predict(prompt.format(user_message=user_message))

    # Parse JSON safely
    import json
    try:
        result = json.loads(response)
    except:
        result = {"human_message": response, "score": "N/A"}
    
    return result


# Streamlit UI
st.set_page_config(
    page_title="AI → Human Message Chatbot",
    page_icon="🤖",
    layout="centered"
)

st.header("AI → Human Message Chatbot 🤖➡️👨")

user_input = st.text_area("Enter your message:")

if st.button("Convert"):
    if not user_input.strip():
        st.warning("⚠️ Please enter a message first.")
    else:
        result = get_human_message(user_input)
        st.subheader("📝 Human-like Message")
        st.write(result["human_message"])
        st.subheader("📊 Human-likeness Score")
        st.metric(label="Score", value=f"{result['score']} %")
