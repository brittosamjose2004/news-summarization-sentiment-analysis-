import streamlit as st
import requests
import pandas as pd
import json
import os
import matplotlib.pyplot as plt
import seaborn as sns
import base64
from io import BytesIO
from PIL import Image, ImageEnhance
import time
import subprocess
from typing import Dict, Any, List
import threading

# Start the API in the background
def start_api():
    subprocess.Popen(["python", "api.py"])

# Start API when the app loads
if 'api_started' not in st.session_state:
    st.session_state.api_started = True
    threading.Thread(target=start_api).start()
    # Give the API time to start
    time.sleep(5)

# API Base URL - For Spaces deployment
API_BASE_URL = "http://localhost:8000"

# Import the rest of your app.py code here
# Copy everything else from app.py