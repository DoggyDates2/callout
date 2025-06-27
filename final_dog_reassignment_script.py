import pandas as pd
import streamlit as st

def load_google_sheet_data():
    """Load data from Google Sheets using correct URLs"""
    try:
        # Distance Matrix - CORRECT URL
        matrix_url = "https://docs.google.com/spreadsheets/d/1421xCS86YH6hx0RcuZCyXkyBK_xl-VDSlXyDNvw09Pg/export?format=csv&gid=2146002137"
        distance_matrix = pd.read_csv(matrix_url, index_col=0)
        
        # Map Data - CORRECT URL  
        map_url = "https://docs.google.com/spreadsheets/d/1mg8d5CLxSR54KhNUL8SpL5jzrGN-bghTsC9vxSK8lR0/export?format=csv&gid=267803750"
        map_data = pd.read_csv(map_url)
        
        # Driver Counts - CORRECT URL
        driver_url = "https://docs.google.com/spreadsheets/d/1mg8d5CLxSR54KhNUL8SpL5jzrGN-bghTsC9vxSK8lR0/export?format=csv&gid=1359695250"
        driver_data = pd.read_csv(driver_url)
        
        return distance_matrix, map_data, driver_data
        
    except Exception as e:
        st.error(f"Error loading data from Google Sheets: {e}")
        return None, None, None

def find_callout_drivers(driver_data):
    """Find drivers who called out (have X in their group columns)"""
    callouts = {}
    
    for _, row in driver_data.iterrows():
        driver = row['Driver']
        callout_groups = []
        
        if str(row['Group 1']).upper() == 'X':
            callout_groups.append(1)
        if str(row['Group 2']).upper() == 'X':
            callout_groups.append(2)
        if str(row['Group 3']).upper() == 'X':
            callout_groups.append(3)
        
        if callout_groups:
            callouts[driver] = callout_groups
    
    return callouts

def debug_live_data():
    """Debug what's happening with live data"""
    st.title("🔍 Debug: Live Data Analysis")
    
    # Load data
    distance_matrix, map_data, driver_data = load_google_sheet_data()
    
    if distance_matrix is None or map_data is None or driver_data is None:
        st.error("Could not load data")
        return
    
    # Display data status
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Distance Matrix", f"{len(distance_matrix)} x {len(distance_matrix.columns)}")
    with col2:
        st.metric("Dogs in Map", len(map_data))
    with col3:
        st.metric("Drivers", len(driver_data))
    
    # Check Drew's callout
    auto_callouts = find_callout_drivers(driver_data)
    st.subheader("Detected Callouts:")
    st.write(auto_callouts)
    
    # Debug Drew's dogs
    st.subheader("🔍 Debugging Drew's Dogs")
    
    # Check Today column
    st.write("**Today column info:**")
    st.write(f"Column exists: {'Today' in map_data.columns}")
    
    if 'Today' in map_data.columns:
        # Check for Drew assignments
        drew_filter = map_data['Today'].str.contains('Drew:', na=False)
        drew_dogs = map_data[drew_filter]
        
        st.write(f"Dogs with 'Drew:' in Today column: {len(drew_dogs)}")
        
        if len(drew_dogs) > 0:
            st.write("**Drew's dogs found:**")
            st.dataframe(drew_dogs[['Dog Name', 'Dog ID', 'Today', 'Number of dogs']])
        else:
            st.error("❌ No Drew dogs found!")
            
            # Show sample data
            st.write("**Sample Today values:**")
            sample_today = map_data['Today'].dropna().head(10)
            for i, value in enumerate(sample_today):
                st.write(f"{i}: '{value}' (type: {type(value)})")
            
            # Check for Drew mentions
            drew_mentions = map_data[map_data['Today'].str.contains('drew', case=False, na=False)]
            st.write(f"Case-insensitive 'drew' mentions: {len(drew_mentions)}")
            
            if len(drew_mentions) > 0:
                st.dataframe(drew_mentions[['Dog Name', 'Dog ID', 'Today']])
    else:
        st.error("❌ 'Today' column not found!")
        st.write("Available columns:", list(map_data.columns))
    
    # Test the exact script logic
    st.subheader("🧪 Test Script Logic")
    
    if st.button("Test Drew Dog Finding"):
        callout_driver = 'Drew'
        callout_pattern = f'{callout_driver}:'
        
        st.write(f"Looking for pattern: '{callout_pattern}'")
        
        # Exact script logic
        affected_dogs = map_data[map_data['Today'].str.contains(callout_pattern, na=False)]
        
        st.write(f"Found {len(affected_dogs)} affected dogs")
        
        if len(affected_dogs) > 0:
            st.success("✅ Script logic works!")
            st.dataframe(affected_dogs[['Dog Name', 'Dog ID', 'Today', 'Number of dogs']])
        else:
            st.error("❌ Script logic failed to find dogs")
            
            # Additional debugging
            st.write("**Debugging steps:**")
            st.write(f"1. Total rows in map_data: {len(map_data)}")
            st.write(f"2. Non-null Today values: {map_data['Today'].notna().sum()}")
            st.write(f"3. Contains 'Drew:': {map_data['Today'].str.contains('Drew:', na=False).sum()}")
            
            # Show first few Today values
            st.write("**First 10 Today values:**")
            st.write(map_data['Today'].head(10).tolist())

if __name__ == "__main__":
    debug_live_data()
