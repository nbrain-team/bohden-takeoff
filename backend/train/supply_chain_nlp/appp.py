import streamlit as st
import requests

st.title("🔍 Construction Supplier Copilot")

query = st.text_input("Enter Material Type or Supplier Need (e.g., 'Steel Supplier in Mumbai')")

if st.button("Get Recommendations"):
    response = requests.get(f"http://127.0.0.1:8000/recommend/?query={query}")
    results = response.json()

    for supplier in results["Recommendations"]:
        st.write(f"**{supplier['Supplier Name']}** - {supplier['Material Type']} ({supplier['Location']})")
        st.write(f"🔹 Distance Score: {supplier['Distance Score']}")
        st.write("---")
