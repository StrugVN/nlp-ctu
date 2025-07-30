import streamlit as st

st.set_page_config(page_title="Search UI", layout="centered")

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
        print("awdawdw")
        st.success(f"🔎 You searched for: **{query}**")
    else:
        st.warning("Please enter something!")
