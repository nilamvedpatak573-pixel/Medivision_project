import streamlit as st
import os

# page setup

st.set_page_config(
    page_title="MediVision AI",
    page_icon="🩺",
    layout="wide"
)

# Base Directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# sidebar design

sidebar_image = os.path.join(BASE_DIR, "utils", "ph3.png")

# Title
st.sidebar.markdown(
"""
<div style="text-align:center; font-size:24px; font-weight:700;">
🩺 MediVision AI
</div>
""",
unsafe_allow_html=True
)

# Navigation
st.sidebar.markdown(
"""
<div style="font-size:18px; font-weight:700; margin-top:10px;">
📌 Navigation
</div>

<div style="font-size:15px; line-height:1.8;">

🏠 Home

🧬 Disease Prediction

❤️ Heart Risk Assessment

💊 Drug Recommendation

🤖 Medical Assistant

</div>
""",
unsafe_allow_html=True
)

# Sidebar Image
if os.path.exists(sidebar_image):
    st.sidebar.image(
        sidebar_image,
        use_container_width=True
    )
else:
    st.sidebar.warning("Sidebar image not found.")

# Short Description
st.sidebar.markdown(
"""
<div style="margin-top:8px;">

<h4 style="margin-bottom:6px;">
AI-Powered Healthcare Intelligence
</h4>

<p style="
margin-top:0px;
line-height:1.5;
text-align:justify;
font-size:15px;
">

MediVision AI is an intelligent healthcare platform
that combines Machine Learning, Generative AI, and
RAG technology to provide smart disease prediction,
heart risk assessment, drug recommendations, and
AI-powered medical assistance.

</p>

</div>
""",
unsafe_allow_html=True
)
# main title page

st.markdown(
"""
<div style='text-align:center;'>

<div style='font-size:32px;font-weight:700;'>
🩺 MediVision AI Healthcare System
</div>

<div style='font-size:18px;margin-top:8px;'>
AI-Powered Healthcare Intelligence Platform
</div>

</div>
""",
unsafe_allow_html=True
)

# =========================
# MAIN BANNER IMAGE
# =========================

image_path = os.path.join(
    BASE_DIR,
    "utils",
    "ph1.png"
)


if os.path.exists(image_path):

    st.image(
        image_path,
        use_container_width=True
    )

else:
    st.warning("Home image not found.")

# Module cards

st.markdown(
"""
<div style='text-align:center;font-size:32px;font-weight:700;'>
Main Features
</div>
""",
unsafe_allow_html=True
)

st.markdown(
"""
<style>

.module-card {

    padding:18px;
    border-radius:15px;
    border:1px solid rgba(128,128,128,0.4);
    background-color:rgba(128,128,128,0.08);
    height:180px;

}

.module-card h3 {

    font-size:20px;

}

.module-card p {

    font-size:15px;
    line-height:1.4;

}

</style>
""",
unsafe_allow_html=True
)

# First Row

col1, col2 = st.columns(2)

with col1:

    st.markdown(
    """
    <div class="module-card">

    <h3>🧬 Disease Prediction</h3>

    <p>
    Predict possible diseases based on symptoms
    using Machine Learning algorithms for early
    healthcare assistance.
    </p>

    </div>
    """,
    unsafe_allow_html=True
    )

with col2:

    st.markdown(
    """
    <div class="module-card">

    <h3>❤️ Heart Risk Assessment</h3>

    <p>
    Analyze patient health parameters and estimate
    heart disease risk using AI-based prediction models.
    </p>

    </div>
    """,
    unsafe_allow_html=True
    )


# Second Row

col3, col4 = st.columns(2)

with col3:

    st.markdown(
    """
    <div class="module-card">

    <h3>💊 Drug Recommendation</h3>

    <p>
    Provides medicine information and healthcare
    recommendations using intelligent matching
    techniques.
    </p>

    </div>
    """,
    unsafe_allow_html=True
    )

with col4:

    st.markdown(
    """
    <div class="module-card">

    <h3>🤖 Medical Assistant</h3>

    <p>
    AI-powered medical chatbot using LLM and RAG
    technology for medical report analysis and queries.
    </p>

    </div>
    """,
    unsafe_allow_html=True
    )


# technologies used

st.markdown(
"""
<div style='text-align:center; font-size:30px; font-weight:700; margin-top:25px; margin-bottom:20px;'>
Technologies Used
</div>
""",
unsafe_allow_html=True
)

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.button("🐍 Python", use_container_width=True)

with col2:
    st.button("🎨 Streamlit", use_container_width=True)

with col3:
    st.button("🤖 Machine Learning", use_container_width=True)

with col4:
    st.button("🧠 LLM + RAG", use_container_width=True)

col5, col6, col7, col8 = st.columns(4)

with col5:
    st.button("📚 LangChain", use_container_width=True)

with col6:
    st.button("🗂️ FAISS", use_container_width=True)

with col7:
    st.button("🤗 Hugging Face", use_container_width=True)

with col8:
    st.button("⚡ Groq API", use_container_width=True)    