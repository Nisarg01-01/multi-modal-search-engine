# ui.py
import streamlit as st
import requests

# --- Page Configuration ---
st.set_page_config(page_title="Unified Search", page_icon="🔍", layout="wide")

# --- Static Variables ---
API_URL = "http://api:8000"
# This is the base URL needed to construct the full image path
IMAGE_BASE_URL = "https://m.media-amazon.com/images/I/"

# --- UI Components ---
st.title("🔍 Unified Multi-Modal Search")
st.write("Enter a text description OR upload an image to search for products.")

with st.form(key="search_form"):
    col1, col2 = st.columns([4, 1])
    with col1:
        text_query = st.text_input(
            "Search query",
            placeholder="e.g., 'a comfortable chair for my office' or upload an image ->",
        )
    with col2:
        uploaded_file = st.file_uploader(
            "Upload Image", type=["jpg", "jpeg", "png"], label_visibility="collapsed"
        )

    search_button = st.form_submit_button(label="Search")


# --- Logic to Handle Search and Display Results ---
def display_results(results):
    if not results:
        st.warning("No products found that match your query.")
        return

    st.header("Search Results")
    num_columns = 5
    cols = st.columns(num_columns)
    for i, item in enumerate(results):
        with cols[i % num_columns]:
            # --- THIS IS THE FIX ---
            # Construct the full URL from the base and the image ID
            image_url = f"{IMAGE_BASE_URL}{item['image_id']}.jpg"
            st.image(
                image_url,
                use_container_width=True,
                caption=f"Dist: {item['distance']:.2f}",
            )
            st.markdown(f"**{item['name']}**")


if search_button:
    # --- Image Search Logic ---
    if uploaded_file is not None:
        with st.spinner("Searching for visually similar products..."):
            files = {
                "file": (
                    uploaded_file.name,
                    uploaded_file.getvalue(),
                    uploaded_file.type,
                )
            }
            response = requests.post(f"{API_URL}/image_upload_search/", files=files)
            if response.status_code == 200:
                display_results(response.json().get("results", []))
            else:
                st.error(f"API Error: {response.status_code} - {response.text}")

    # --- Text Search Logic ---
    elif text_query:
        with st.spinner("Searching for products based on your text..."):
            response = requests.post(
                f"{API_URL}/text_search/", json={"query": text_query}
            )
            if response.status_code == 200:
                display_results(response.json().get("results", []))
            else:
                st.error(f"API Error: {response.status_code} - {response.text}")
    else:
        st.warning("Please enter a text query or upload an image.")
