import os
import streamlit as st
from pymongo import MongoClient
from dotenv import load_dotenv

load_dotenv()

MONGO_URI = os.getenv("MONGO_URI")

client = MongoClient(MONGO_URI)
db = client["nlp"]

st.set_page_config(page_title="Search UI", layout="centered")

colA, colB = st.columns([4, 1])
with colB:
    search_mode = st.selectbox(
        "Search method",
        options=["MongoDB", "Custom"],
        index=1,  # Default to "Custom"
        key="search_mode",
        label_visibility="collapsed",
    )


# --- Custom CSS ---
st.markdown(
    """
    <style>
    .search-container {
        display: flex;
        align-items: center;
        justify-content: center;
        margin-top: 100px;
    }
    .search-input input {
        width: 400px !important;
        height: 40px;
        border-radius: 20px;
        border: 1px solid #ccc;
        padding: 0 15px;
        font-size: 16px;
    }
    .search-button button {
        border-radius: 50%;
        width: 40px;
        height: 40px;
        font-size: 18px;
        margin-left: 5px;
    }
    </style>
""",
    unsafe_allow_html=True,
)

# --- UI Layout with columns ---
st.markdown('<div class="search-container">', unsafe_allow_html=True)
col1, col2 = st.columns([5, 1])

with col1:
    query = st.text_input(
        "Search box",  # give it a label for accessibility
        placeholder="Search...",
        key="search",
        label_visibility="collapsed",  # hides it visually
    )


with col2:
    search_clicked = st.button("🔍")

st.markdown("</div>", unsafe_allow_html=True)

# --- Handle click event ---
if search_clicked:
    if query.strip():
        st.success(f"🔎 You searched for: **{query}**")

        if search_mode == "MongoDB":
            # MongoDB full-text search
            results = (
                db.article.find(
                    {"$text": {"$search": query}}, {"score": {"$meta": "textScore"}}
                )
                .sort([("score", {"$meta": "textScore"})])
                .limit(10)
            )
        elif search_mode == "Custom":
            # Simulate a custom search (replace with your logic)
            results = db.article.find(
                {"title": {"$regex": query, "$options": "i"}}
            ).limit(10)

        results = list(results)

        # Display results
        st.markdown("### Search Results")
        if len(results) == 0:
            st.markdown("No results found.")
        else:
            for r in results:
                st.markdown(f"- **{r['title']}**: {r['content'][:200]}...")
    else:
        st.warning("Please enter something!")
