import os
import asyncio
import nest_asyncio
import streamlit as st
from dotenv import load_dotenv, find_dotenv

from utils.pdf_processor import extract_pdf_text, split_text

from langchain_groq import ChatGroq
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings

# Load Environment
load_dotenv(".env", override=True)
nest_asyncio.apply()

# Sidebar
st.sidebar.markdown(
    "<h2 style='color:white;'>📌 Description</h2>",
    unsafe_allow_html=True
)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
image_path = os.path.join(BASE_DIR, "utils", "ph2.png")
st.sidebar.image(image_path, use_container_width=True)

st.sidebar.markdown(
    "<p class='sidebar-text'>The LLM Medical Chatbot is an AI-powered assistant designed to provide instant, accurate and reliable healthcare insights.</p>",
    unsafe_allow_html=True
)

# Async loop

try:
    asyncio.get_running_loop()
except RuntimeError:
    asyncio.set_event_loop(asyncio.new_event_loop())

# Constants

DB_FAISS_PATH = "vectorstore/db_faiss"

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    st.error("❌ GROQ_API_KEY not found.")
    st.stop()

# Embedding Model
@st.cache_resource
def load_embedding_model():

    embedding_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    return embedding_model

embedding_model = load_embedding_model()

# Load Medical Knowledge Base

@st.cache_resource
def load_vectorstore():

    try:

        db = FAISS.load_local(
            DB_FAISS_PATH,
            embedding_model,
            allow_dangerous_deserialization=True
        )

        return db

    except Exception as e:

        st.error(f"Error loading FAISS database: {e}")

        return None


vectorstore = load_vectorstore()

# Prompt
def get_prompt_template():

    return PromptTemplate(

        template="""
Use the provided context to answer the user's question.

If you don't know the answer, simply say "I don't know".

Context:
{context}

Question:
{question}

Give a concise medical answer.
""",

        input_variables=[
            "context",
            "question"
        ]
    )

# Load Groq
def load_llm():
    return ChatGroq(
        model="llama-3.3-70b-versatile",
        groq_api_key=GROQ_API_KEY,
        temperature=0.5
    )

# Source Formatter
def format_sources(source_documents):

    if not source_documents:

        return "**Sources:** No sources found."

    text = "\n\n**Sources:**"

    for i, doc in enumerate(source_documents, start=1):

        text += f"\n🔹 Source {i}: {doc.metadata.get('source','Unknown')}"

    return text

def generate_report_summary(report_text):
    llm = load_llm()

    prompt = f"""
You are an experienced medical assistant.

Below is a patient's medical report.

Your task is to generate a concise and easy-to-understand summary.

Include:
1. Patient Details (if available)
2. Tests performed
3. Important Results
4. Abnormal Values (if any)
5. Overall Summary

Do not invent information.
If a value is not available, ignore it.

Medical Report:

{report_text}
"""
    response = llm.invoke(prompt)
    return response.content

# Main
def main():

    st.title("💬 Medibot - AI Health Assistant")

    st.markdown("""
Ask any medical question.

🤖 Powered by Groq + FAISS + Gemini Vision
""")

    uploaded_file = st.file_uploader(
        "📤 Upload Medical Report (PDF)",
        type=["pdf"]
    )

    if uploaded_file:
        st.success("✅ File uploaded successfully!")
        st.info(f"📄 Uploaded PDF: {uploaded_file.name}")
        with st.spinner("📄 Processing file..."):
            report_context = extract_pdf_text(uploaded_file)

            chunks = split_text(report_context)

            report_vectorstore = FAISS.from_texts(
                chunks,
                embedding_model
            )    
            st.session_state.report_vectorstore = report_vectorstore
            st.session_state.report_context = report_context
            if report_context.startswith("Error"):
                st.error(report_context)

            else:
                 st.success("✅ Report processed successfully!")

        st.subheader("📄 AI Report Summary")     
        with st.spinner("Generating summary..."):
            summary = generate_report_summary(report_context)

        st.success(summary)    

        #st.write(report_context[:1000])

    with st.sidebar:

        st.markdown("""
### 🔍 About Medibot

- 📄 Upload PDF reports
- 🤖 AI Report Summarization 
- 💬 Medical Chatbot
- 🧠 Powered by Groq + FAISS
""")

    if "messages" not in st.session_state:

        st.session_state.messages = []

    if "report_context" not in st.session_state:

        st.session_state.report_context = ""


    for message in st.session_state.messages:

        st.chat_message(
            message["role"]
        ).markdown(
            message["content"]
        )

    user_query = st.chat_input(
        "Type your medical question..."
    )

    if user_query:

        # Display user message
        st.chat_message("user").markdown(user_query)

        st.session_state.messages.append(
            {
                "role": "user",
                "content": user_query
            }
        )

        with st.spinner("🤖 Medibot is thinking..."):

            try:

                if vectorstore is None:
                    st.error("❌ Vector database not found.")
                    return

                # Choose Retriever

                if "report_vectorstore" in st.session_state:

                    retriever = st.session_state.report_vectorstore.as_retriever(
                        search_kwargs={"k": 5}
                    )

                else:

                    retriever = vectorstore.as_retriever(
                        search_kwargs={"k": 5}
                    )

                # Build QA Chain

                qa_chain = RetrievalQA.from_chain_type(
                    llm=load_llm(),
                    chain_type="stuff",
                    retriever=retriever,
                    return_source_documents=True,
                    chain_type_kwargs={
                        "prompt": get_prompt_template()
                    }
                )

                # Query

                if st.session_state.report_context:

                    final_query = f"""
You are a helpful medical assistant.

Uploaded Medical Report:

{st.session_state.report_context}

User Question:

{user_query}

Instructions:
- First answer using the uploaded report.
- If the report does not contain enough information,
  answer using your medical knowledge.
- Explain medical terms simply.
- Do not provide a final diagnosis.
- Recommend consulting a doctor when appropriate.
"""

                else:

                    final_query = user_query

                # Get Response

                response = qa_chain.invoke(
                    {
                        "query": final_query
                    }
                )

                result = response.get(
                    "result",
                    "No answer generated."
                )

                sources = response.get(
                    "source_documents",
                    []
                )

                formatted_response = (
                    f"**Medibot:**\n\n{result}"
                )

                st.chat_message("assistant").markdown(
                    formatted_response
                )

                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": formatted_response
                    }
                )

            except Exception as e:

                st.error(f"⚠️ {e}")


if __name__ == "__main__":
    main()


