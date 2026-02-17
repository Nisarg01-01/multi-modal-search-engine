# ui.py
import requests
import streamlit as st
from PIL import Image

# -----------------------------------------------------------------------------
# Configuration & Styles
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="TrendScout | AI Fashion Search",
    page_icon="page",
    layout="wide",
)

# ... (CSS omitted for brevity) ...

# Title
st.markdown("<h1>TrendScout</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>Discover fashion with AI-powered semantic and visual search</p>", unsafe_allow_html=True)

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
            st.markdown(f"""
            <div class="product-card">
                <img src="{img_url}" class="result-img" loading="lazy">
                <div style="padding: 10px 0 0 0;">
                    <p style="font-weight: 600; font-size: 0.9rem; margin: 0; white-space: nowrap; overflow: hidden; text-overflow: ellipsis;">{item['name']}</p>
                    <p style="color: #666; font-size: 0.8rem; margin: 0;">Match: {1 - item['distance']:.0%}</p>
                </div>
            </div>
            """, unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("<div style='text-align: center; color: #aaa; font-size: 0.8rem;'>Powered by CLIP & Weaviate</div>", unsafe_allow_html=True)
