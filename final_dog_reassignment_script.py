import streamlit as st
import pandas as pd

st.title("🐕 Debug: Testing Each Google Sheet")

# Test each URL individually
urls = {
    "Distance Matrix": "https://docs.google.com/spreadsheets/d/e/2PACX-1vSh-t6fIfgsli9D79KZeXM_-V5fO3zam6T_Bcp94d-IoRucxWusl6vbtT-WSaqFimHw7ABd76YGcKGV/pub?gid=0&single=true&output=csv",
    "Map Data": "https://docs.google.com/spreadsheets/d/e/2PACX-1vT_9o2CJOgNpwtLPaqZkYBYVHtNJo5D-0qfeRqtdW-9yYV9cp5TMOvI5YTR8Xp3GcGhOU25mGBTHEdF/pub?gid=0&single=true&output=csv",
    "Driver Counts": "https://docs.google.com/spreadsheets/d/e/2PACX-1vT_9o2CJOgNpwtLPaqZkYBYVHtNJo5D-0qfeRqtdW-9yYV9cp5TMOvI5YTR8Xp3GcGhOU25mGBTHEdF/pub?gid=1359695250&single=true&output=csv"
}

for name, url in urls.items():
    st.subheader(f"Testing: {name}")
    try:
        if name == "Distance Matrix":
            data = pd.read_csv(url, index_col=0)
        else:
            data = pd.read_csv(url)
        
        st.success(f"✅ {name} loaded successfully!")
        st.write(f"Shape: {data.shape}")
        st.write(f"Columns: {list(data.columns[:5])}...")  # Show first 5 columns
        
        if name == "Driver Counts":
            st.write("First few rows:")
            st.dataframe(data.head())
            
            # Check for Drew
            if 'Driver' in data.columns:
                drew_data = data[data['Driver'] == 'Drew']
                if not drew_data.empty:
                    st.write("✅ Found Drew:", drew_data.iloc[0].to_dict())
                else:
                    st.warning("❌ Drew not found in Driver column")
            else:
                st.warning(f"❌ No 'Driver' column found. Columns are: {list(data.columns)}")
                
    except Exception as e:
        st.error(f"❌ Failed to load {name}")
        st.error(f"Error: {e}")
        st.write(f"URL: {url}")

st.write("---")
st.write("If all three show ✅, then the issue is elsewhere.")
st.write("If any show ❌, that's the problem sheet.")
