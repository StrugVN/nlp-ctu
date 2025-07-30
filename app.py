import streamlit as st

st.set_page_config(page_title="Search UI", layout="centered")


# # --- Input box ---
# query = st.text_input(
#     "Search", placeholder="Type something...", label_visibility="collapsed"
# )

# # --- Search icon as a button ---
# search_clicked = st.button("🔍")

# # --- Handle events ---
# if search_clicked:
#     if query.strip():
#         st.write(f"🔎 You searched for: **{query}**")
#     else:
#         st.warning("Please enter a search term!")

# --- Custom CSS for search bar with icon ---
st.markdown(
    """
    <style>
    .search-container {
        display: flex;
        align-items: center;
        justify-content: center;
        margin-top: 100px;
    }
    .search-input {
        width: 400px;
        height: 40px;
        border-radius: 20px;
        border: 1px solid #ccc;
        padding: 0 40px 0 15px;
        font-size: 16px;
    }
    .search-icon {
        margin-left: -35px;
        cursor: pointer;
    }
    </style>
""",
    unsafe_allow_html=True,
)

# --- UI Layout ---
st.markdown(
    """
<div class="search-container">
    <input type="text" id="search" placeholder="Search..." class="search-input">
    <span class="search-icon">🔍</span>
</div>
""",
    unsafe_allow_html=True,
)


# search_clicked = st.button("🔍")
