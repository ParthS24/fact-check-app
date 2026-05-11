import streamlit as st
import pdfplumber
import google.generativeai as genai
from duckduckgo_search import DDGS

# -----------------------------
# GEMINI API CONFIGURATION
# -----------------------------

API_KEY = st.secrets("GEMINI_API_KEY")

genai.configure(api_key=API_KEY)

model = genai.GenerativeModel("gemini-1.5-flash")

# -----------------------------
# STREAMLIT UI
# -----------------------------

st.set_page_config(page_title="AI Fact Checker")

st.title("AI Fact-Checking Web App")

st.write(
    "Upload a PDF and verify factual claims using AI + live web search."
)

uploaded_file = st.file_uploader(
    "Upload PDF",
    type=["pdf"]
)

# -----------------------------
# PDF TEXT EXTRACTION
# -----------------------------


def extract_text(pdf_file):

    text = ""

    with pdfplumber.open(pdf_file) as pdf:

        for page in pdf.pages:

            page_text = page.extract_text()

            if page_text:
                text += page_text + "\n"

    return text


# -----------------------------
# CLAIM EXTRACTION
# -----------------------------


def extract_claims(text):

    prompt = f"""
    Extract factual claims, statistics, dates,
    technical claims, and financial figures.

    Return ONLY bullet points.

    TEXT:
    {text[:12000]}
    """

    response = model.generate_content(prompt)

    claims = response.text.split("\n")

    final_claims = []

    for claim in claims:

        claim = claim.strip()

        if len(claim) > 10:
            final_claims.append(claim)

    return final_claims


# -----------------------------
# WEB SEARCH
# -----------------------------


def search_web(claim):

    snippets = []

    with DDGS() as ddgs:

        results = list(
            ddgs.text(claim, max_results=3)
        )

    for r in results:

        body = r.get("body", "")

        if body:
            snippets.append(body)

    return snippets


# -----------------------------
# FACT VERIFICATION
# -----------------------------


def verify_claim(claim, web_data):

    prompt = f"""
    You are a fact-checking AI.

    CLAIM:
    {claim}

    WEB SEARCH DATA:
    {web_data}

    Determine whether the claim is:
    - Verified
    - Inaccurate
    - False

    Also provide:
    1. Short explanation
    2. Correct fact if needed

    Keep answer concise.
    """

    response = model.generate_content(prompt)

    return response.text


# -----------------------------
# MAIN WORKFLOW
# -----------------------------

if uploaded_file:

    st.success("PDF Uploaded Successfully")

    with st.spinner("Extracting text from PDF..."):

        text = extract_text(uploaded_file)

    with st.spinner("Extracting claims..."):

        claims = extract_claims(text)

    st.subheader("Detected Claims")

    for i, claim in enumerate(claims):

        st.markdown("---")

        st.write(f"## Claim {i+1}")

        st.write(claim)

        with st.spinner("Searching live web data..."):

            web_data = search_web(claim)

        with st.spinner("Verifying claim..."):

            verdict = verify_claim(claim, web_data)

        st.write(verdict)