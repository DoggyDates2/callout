import streamlit as st
import pandas as pd

st.title("🐕 Debug: Testing Each Google Sheet URL")

# Test URLs with ALL CORRECT spreadsheet IDs and gids
urls_to_test = {
    "Distance Matrix (CORRECT)": "https://docs.google.com/spreadsheets/d/1421xCS86YH6hx0RcuZCyXkyBK_xl-VDSlXyDNvw09Pg/export?format=csv&gid=2146002137",
    
    "Map Data (CORRECT)": "https://docs.google.com/spreadsheets/d/1mg8d5CLxSR54KhNUL8SpL5jzrGN-bghTsC9vxSK8lR0/export?format=csv&gid=267803750",
    
    "Driver Counts (known working)": "https://docs.google.com/spreadsheets/d/1mg8d5CLxSR54KhNUL8SpL5jzrGN-bghTsC9vxSK8lR0/export?format=csv&gid=1359695250"
}

st.write("**Testing URLs based on the Matrix link you provided:**")
st.code("https://docs.google.com/spreadsheets/d/e/2PACX-1vSh-t6fIfgsli9D79KZeXM_-V5fO3zam6T_Bcp94d-IoRucxWusl6vbtT-WSaqFimHw7ABd76YGcKGV/pubhtml")

for name, url in urls_to_test.items():
    st.subheader(f"Testing: {name}")
    try:
        if "Distance Matrix" in name:
            data = pd.read_csv(url, index_col=0)
        else:
            data = pd.read_csv(url)
        
        st.success(f"✅ {name} loaded successfully!")
        st.write(f"Shape: {data.shape}")
        st.write(f"First 5 columns: {list(data.columns[:5])}")
        
        if "Driver" in name:
            st.write("First few rows:")
            st.dataframe(data.head(3))
            
    except Exception as e:
        st.error(f"❌ Failed to load {name}")
        st.error(f"Error: {e}")
        st.write(f"URL being tested: {url}")

st.write("---")
st.write("**Need to check:**")
st.write("1. Are both spreadsheets public ('Anyone with the link can view')?")
st.write("2. What are the correct tab names and gid numbers?")

st.write("**To find the correct gid:**")
st.write("1. Open your spreadsheet")
st.write("2. Click on the tab you want")
st.write("3. Look at the URL - it will show gid=XXXXXXXX")
st.write("4. Use that number in the CSV URL")
