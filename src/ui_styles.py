import streamlit as st
st.markdown("""
<style>
    
    .stApp {
        background: linear-gradient(135deg, #0f2027, #203a43, #2c5364);
        color: #ffffff;
    }
    
    
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700&display=swap');
    html, body, [class*="css"]  {
        font-family: 'Outfit', sans-serif;
    }

    
    .stSidebar {
        background: rgba(25, 35, 45, 0.4) !important;
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        border-right: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    .stTabs [data-baseweb="tab-list"] {
        background: rgba(255, 255, 255, 0.05);
        border-radius: 12px;
        padding: 5px;
    }

    .stTabs [data-baseweb="tab"] {
        background: transparent;
        color: #ccc;
        border-radius: 8px;
        transition: all 0.3s ease;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(90deg, #00d2ff 0%, #3a7bd5 100%) !important;
        color: white !important;
        box-shadow: 0 4px 15px rgba(0, 210, 255, 0.3);
    }

    
    .stButton>button {
        background: linear-gradient(90deg, #ff416c 0%, #ff4b2b 100%) !important;
        color: white !important;
        border: none !important;
        border-radius: 8px !important;
        padding: 15px 24px !important;
        font-weight: 700 !important;
        font-size: 16px !important;
        transition: transform 0.2s, box-shadow 0.2s !important;
        box-shadow: 0 4px 15px rgba(255, 65, 108, 0.4) !important;
    }
    
    .stButton>button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 20px rgba(255, 65, 108, 0.6) !important;
    }

    
    .stSelectbox div[data-baseweb="select"], .stNumberInput input, .stMultiSelect div[data-baseweb="select"] {
        background: rgba(255, 255, 255, 0.05) !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        border-radius: 8px !important;
        color: white !important;
    }
    
    
    [data-testid="stMetricValue"] {
        font-size: 2.5rem !important;
        font-weight: 700 !important;
        background: -webkit-linear-gradient(#00d2ff, #3a7bd5);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    
    hr {
        border-color: rgba(255,255,255,0.1) !important;
    }
</style>
""" , unsafe_allow_html=True)
