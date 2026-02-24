# ui.py
import requests
import streamlit as st
from PIL import Image

# -----------------------------------------------------------------------------
# Configuration & Styles
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="AI Product Search | Multi-Modal Engine",
    page_icon="page",
    layout="wide",
)

st.markdown("""
<style>
    .product-card {
        background-color: #ffffff;
        border-radius: 8px;
        overflow: hidden;
        box-shadow: 0 2px 5px rgba(0,0,0,0.05);
        margin-bottom: 15px;
        border: 1px solid #eee;
        height: 100%;
    }
    .product-card:hover {
        box-shadow: 0 5px 15px rgba(0,0,0,0.1);
        transform: translateY(-2px);
        transition: all 0.2s ease;
    }
    .result-img {
        width: 100%;
        height: 250px;
        object-fit: contain;
        background-color: #f8f9fa;
        border-bottom: 1px solid #eee;
    }
    .subtitle {
        font-size: 1.2rem;
        color: #666;
        margin-bottom: 30px;
    }
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# State Management
# -----------------------------------------------------------------------------
if "search_results" not in st.session_state:
    st.session_state.search_results = []
if "current_query" not in st.session_state:
    st.session_state.current_query = ""

API_URL = "http://api:8000"
IMAGE_BASE_URL = "http://localhost:8000/static/images/"

def run_search(endpoint, payload, files=None):
    """Helper to execute search and update state."""
    try:
        if files:
            response = requests.post(f"{API_URL}{endpoint}", files=files)
        else:
            response = requests.post(f"{API_URL}{endpoint}", json=payload)
            
        if response.status_code == 200:
            data = response.json()
            st.session_state.search_results = data.get("results", [])
            if "query" in payload:
                st.session_state.current_query = payload["query"]
        else:
            st.error(f"Search failed: {response.text}")
    except Exception as e:
        st.error(f"Connection error: {e}")

# -----------------------------------------------------------------------------
# Styles

# Title
st.markdown("<h1>AI Product Search</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>Multi-Modal Search Engine</p>", unsafe_allow_html=True)

# Search Interface (Tabs for stability)
tab_text, tab_image = st.tabs(["Text Search", "Visual Search"])

with tab_text:
    col1, col2 = st.columns([3, 1], gap="small")
    with col1:
        text_query = st.text_input("Describe what you're looking for...", placeholder="e.g. sleeveless red summer dress", key="text_q", label_visibility="collapsed")
    with col2:
        if st.button("Search Text", type="primary", use_container_width=True):
            if text_query:
                # Use Hybrid by default for best results
                run_search("/hybrid_search/", {"query": text_query})
            else:
                st.warning("Please enter a description.")

with tab_image:
    uploaded_file = st.file_uploader("Upload an image to find similar items", type=["jpg", "jpeg", "png"])
    
    if uploaded_file:
        # Layout: Image Preview | Search Button
        p_col1, p_col2 = st.columns([1, 4])
        with p_col1:
            st.image(uploaded_file, width=150, caption="Preview")
        with p_col2:
            st.write("Image selected.")
            if st.button("Search Similar Images", type="primary"):
                files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
                run_search("/image_upload_search/", {}, files=files)

# Results Section
if st.session_state.search_results:
    st.divider()
    
    # Grid Layout
    results = st.session_state.search_results
    num_cols = 5
    cols = st.columns(num_cols)
    
    for i, item in enumerate(results):
        col = cols[i % num_cols]
        with col:
            # Construct URL
            img_id = item["image_id"]
            img_url = img_id if img_id.startswith("http") else f"{IMAGE_BASE_URL}{img_id}"
            
            # Card HTML
            # Format Data
            brand = item.get("brand") or "Brand"
            price = item.get("price") or 0

            st.markdown(f"""
            <div class="product-card">
                <img src="{img_url}" class="result-img" loading="lazy">
                <div style="padding: 10px;">
                    <p style="font-weight: 600; font-size: 0.9rem; margin: 0; white-space: nowrap; overflow: hidden; text-overflow: ellipsis;" title="{item['name']}">{item['name']}</p>
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-top: 4px;">
                        <span style="color: #666; font-size: 0.8rem;">{brand}</span>
                        <span style="font-weight: 600; font-size: 0.85rem; color: #333;">₹{float(price):,.0f}</span>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("<div style='text-align: center; color: #aaa; font-size: 0.8rem;'>Powered by CLIP & Weaviate</div>", unsafe_allow_html=True)
