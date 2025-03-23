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
from typing import Dict, Any, List

# API Base URL - Change this to match your deployment
API_BASE_URL = "http://localhost:8000"

# New function to generate the example output format
def generate_example_output(company_name: str) -> str:
    """
    Generate output in the example format for the given company.
    Returns the formatted JSON as a string.
    """
    try:
        # Make API request to get the analysis data
        url = f"{API_BASE_URL}/api/complete_analysis"
        response = requests.post(url, json={"company_name": company_name})
        response.raise_for_status()
        data = response.json()
        
        # Format the data to match the example output format exactly
        formatted_output = {
            "Company": data["Company"],
            "Articles": data["Articles"],
            "Comparative Sentiment Score": {
                "Sentiment Distribution": data["Comparative Sentiment Score"]["Sentiment Distribution"],
                "Coverage Differences": data["Comparative Sentiment Score"]["Coverage Differences"],
                "Topic Overlap": data["Comparative Sentiment Score"]["Topic Overlap"]
            },
            "Final Sentiment Analysis": data["Final Sentiment Analysis"],
            "Audio": "[Play Hindi Speech]" if data.get("Audio") else "No audio available"
        }
        
        # Convert to JSON string with proper formatting
        return json.dumps(formatted_output, indent=2)
        
    except Exception as e:
        return json.dumps({
            "error": str(e),
            "message": "Failed to generate example output"
        }, indent=2)

# Function to run in terminal mode
def run_terminal_mode():
    """Run the app in terminal mode to output JSON"""
    print("News Analysis Terminal Mode")
    company_name = input("Enter company name: ")
    print(f"Analyzing {company_name}...")
    output = generate_example_output(company_name)
    print(output)

# Check if run directly or imported
if __name__ == "__main__":
    # Check if terminal mode is requested via command line args
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "--terminal":
        run_terminal_mode()
    else:
        # Continue with the Streamlit app
        
        # App title and description
        st.set_page_config(
            page_title="News Summarization & TTS",
            page_icon="📰",
            layout="wide",
            initial_sidebar_state="expanded"
        )

# Custom CSS for better UI
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        font-weight: 600;
        color: #2563EB;
        margin-top: 1rem;
        margin-bottom: 0.5rem;
    }
    .card {
        padding: 1.5rem;
        border-radius: 0.5rem;
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        margin-bottom: 1rem;
    }
    .positive {
        color: #059669;
        font-weight: 600;
    }
    .negative {
        color: #DC2626;
        font-weight: 600;
    }
    .neutral {
        color: #6B7280;
        font-weight: 600;
    }
    .topic-tag {
        display: inline-block;
        padding: 0.25rem 0.5rem;
        border-radius: 2rem;
        background-color: #E5E7EB;
        color: #1F2937;
        font-size: 0.75rem;
        margin-right: 0.5rem;
        margin-bottom: 0.5rem;
    }
    .audio-container {
        width: 100%;
        padding: 1rem;
        background-color: #F3F4F6;
        border-radius: 0.5rem;
        margin-top: 1rem;
    }
    .info-text {
        font-size: 0.9rem;
        color: #4B5563;
    }
    .article-title {
        font-size: 1.2rem;
        font-weight: 600;
        color: #111827;
        margin-bottom: 0.5rem;
        margin-top: 0.5rem;
    }
    .article-summary {
        font-size: 0.9rem;
        color: #374151;
        margin-bottom: 0.5rem;
    }
    .article-meta {
        font-size: 0.8rem;
        color: #6B7280;
        margin-bottom: 0.5rem;
    }
    .section-divider {
        height: 1px;
        background-color: #E5E7EB;
        margin: 1.5rem 0;
    }
    .chart-container {
        background-color: white;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #E2E8F0;
    }
</style>
""", unsafe_allow_html=True)

# Function to make API requests
def make_api_request(endpoint: str, data: Dict[str, Any] = None, method: str = "POST") -> Dict[str, Any]:
    """Make API request to the backend."""
    url = f"{API_BASE_URL}{endpoint}"
    
    try:
        if method == "GET":
            response = requests.get(url)
        else:
            response = requests.post(url, json=data)
            
        response.raise_for_status()
        return response.json()
    except requests.exceptions.ConnectionError:
        st.error("⚠️ Connection Error: Cannot connect to the API server. Please ensure the API server is running at " + API_BASE_URL)
        return {}
    except requests.exceptions.Timeout:
        st.error("⚠️ Timeout Error: The request took too long to complete. Please try again with a different company name.")
        return {}
    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 404:
            st.error("⚠️ No articles found for this company. Please try another company name.")
        elif e.response.status_code == 500:
            # Try to get detailed error message
            try:
                error_detail = e.response.json().get("detail", "Unknown server error")
                st.error(f"⚠️ Server Error: {error_detail}")
            except:
                st.error("⚠️ Internal Server Error: Something went wrong on the server. Please try again later.")
        else:
            st.error(f"⚠️ HTTP Error: {str(e)}")
        return {}
    except Exception as e:
        st.error(f"⚠️ Error: {str(e)}")
        return {}

# Function to create sentiment color
def get_sentiment_color(sentiment: str) -> str:
    """Return CSS class for sentiment."""
    if sentiment == "Positive":
        return "positive"
    elif sentiment == "Negative":
        return "negative"
    else:
        return "neutral"

# Function to create visualization for sentiment distribution
def plot_sentiment_distribution(sentiment_data: Dict[str, int]):
    """Create and display a bar chart for sentiment distribution."""
    labels = ["Positive", "Neutral", "Negative"]
    values = [sentiment_data[label] for label in labels]
    colors = ["#059669", "#6B7280", "#DC2626"]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(labels, values, color=colors)
    ax.set_title("Sentiment Distribution", fontsize=16, fontweight='bold')
    ax.set_ylabel("Number of Articles", fontsize=12)
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Add value labels on top of bars
    for i, v in enumerate(values):
        ax.text(i, v + 0.1, str(v), ha='center', fontweight='bold')
    
    return fig

# Function to display article information
def display_article(article: Dict[str, Any], index: int):
    """Display article information in a card layout."""
    st.markdown(f"<div class='card'>", unsafe_allow_html=True)
    
    # Article title and sentiment
    sentiment = article.get("Sentiment", "Neutral")
    sentiment_class = get_sentiment_color(sentiment)
    
    st.markdown(f"<h3 class='article-title'>{index+1}. {article['Title']}</h3>", unsafe_allow_html=True)
    st.markdown(f"<span class='{sentiment_class}'>{sentiment}</span>", unsafe_allow_html=True)
    
    # Article summary
    st.markdown("<div class='article-summary'>", unsafe_allow_html=True)
    st.markdown(f"{article.get('Summary', 'No summary available.')}", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)
    
    # Topics
    if "Topics" in article and article["Topics"]:
        st.markdown("<div>", unsafe_allow_html=True)
        for topic in article["Topics"]:
            st.markdown(f"<span class='topic-tag'>{topic}</span>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
    
    st.markdown("</div>", unsafe_allow_html=True)

# App layout
st.markdown("<h1 class='main-header'>📰 News Summarization & Text-to-Speech</h1>", unsafe_allow_html=True)
st.markdown("""
<p class='info-text'>
This application extracts news articles about a company, performs sentiment analysis, conducts comparative analysis,
and generates a text-to-speech output in Hindi. Enter a company name to get started.
</p>
""", unsafe_allow_html=True)

# Sidebar
st.sidebar.image("https://cdn-icons-png.flaticon.com/512/2593/2593073.png", width=100)
st.sidebar.title("News Analysis Settings")

# Company selection
company_input_method = st.sidebar.radio(
    "Select company input method:",
    options=["Text Input", "Choose from List"]
)

if company_input_method == "Text Input":
    company_name = st.sidebar.text_input("Enter Company Name:", placeholder="e.g., Tesla")
else:
    companies = ["Apple", "Google", "Microsoft", "Amazon", "Tesla", "Meta", "Netflix", "Uber", "Airbnb", "Twitter"]
    company_name = st.sidebar.selectbox("Select Company:", companies)

# Analysis settings
max_articles = st.sidebar.slider("Maximum Articles to Analyze:", min_value=5, max_value=20, value=10)
st.sidebar.markdown("---")

# Analysis button
analyze_button = st.sidebar.button("Analyze Company News", type="primary")

# Audio playback settings
st.sidebar.markdown("## Audio Settings")
audio_speed = st.sidebar.select_slider("TTS Speech Speed:", options=["Slow", "Normal", "Fast"], value="Normal")
st.sidebar.markdown("---")

# Add option to see JSON in example format
st.sidebar.markdown("## Developer Options")
show_json = st.sidebar.checkbox("Show JSON output in example format")
st.sidebar.markdown("---")

# About section
with st.sidebar.expander("About This App"):
    st.markdown("""
    This application performs:
    - News extraction from multiple sources
    - Sentiment analysis of the content
    - Topic identification and comparative analysis
    - Text-to-speech conversion to Hindi
    
    Built with Streamlit, FastAPI, and various NLP tools.
    """)

# Main content area
if analyze_button and company_name:
    with st.spinner(f"Analyzing news for {company_name}... This may take a minute"):
        # Perform complete analysis
        response = make_api_request(
            "/api/complete_analysis", 
            {"company_name": company_name}
        )
        
        if not response:
            st.error("Failed to retrieve data. Please try again.")
        elif "detail" in response:
            st.error(response["detail"])
        else:
            # Display company header
            st.markdown(f"<h2 class='sub-header'>Analysis Results for {response['Company']}</h2>", unsafe_allow_html=True)
            
            # Display sentiment summary
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.markdown("<div class='card'>", unsafe_allow_html=True)
                st.markdown("<h3 class='sub-header'>Sentiment Overview</h3>", unsafe_allow_html=True)
                st.markdown(f"{response['Final Sentiment Analysis']}")
                st.markdown("</div>", unsafe_allow_html=True)
            
            with col2:
                sentiment_data = response["Comparative Sentiment Score"]["Sentiment Distribution"]
                fig = plot_sentiment_distribution(sentiment_data)
                st.pyplot(fig)
            
            st.markdown("<div class='section-divider'></div>", unsafe_allow_html=True)
            
            # Display Hindi TTS audio
            if "Audio" in response and response["Audio"]:
                st.markdown("<h3 class='sub-header'>Hindi Audio Summary</h3>", unsafe_allow_html=True)
                
                audio_message = response["Audio"]
                
                if audio_message == "Failed to generate audio":
                    st.warning("Hindi audio could not be generated. However, you can still read the Hindi text below.")
                else:
                    try:
                        # Check if the response contains the actual audio file path
                        audio_file_path = response.get("_audio_file_path")
                        
                        if audio_file_path:
                            # Extract the filename
                            audio_filename = os.path.basename(audio_file_path)
                            audio_url = f"{API_BASE_URL}/api/audio/{audio_filename}"
                        else:
                            # If no path is provided, just display a message
                            st.info("Audio is available but the path was not provided.")
                            audio_url = None
                        
                        if audio_url:
                            # Attempt to download the audio file
                            audio_response = requests.get(audio_url)
                            if audio_response.status_code == 200:
                                # Save temporarily
                                temp_audio_path = f"temp_audio_{os.path.basename(audio_url)}"
                                with open(temp_audio_path, "wb") as f:
                                    f.write(audio_response.content)
                                
                                # Play from local file
                                st.markdown("<div class='audio-container'>", unsafe_allow_html=True)
                                st.audio(temp_audio_path, format="audio/mp3")
                                
                                # Display audio download link
                                st.markdown(f"<a href='{audio_url}' download='hindi_summary.mp3'>Download Hindi Audio</a>", unsafe_allow_html=True)
                                
                                # Clean up temp file (optional)
                                # os.remove(temp_audio_path)  # Uncomment to delete after use
                            else:
                                st.warning(f"Unable to load audio file (HTTP {audio_response.status_code}). You can still read the Hindi text below.")
                        else:
                            st.info("Hindi audio summary would be available here.")
                    except Exception as e:
                        st.warning(f"Error playing audio: {str(e)}. You can still read the Hindi text below.")
                
                # Display the Hindi text with better formatting
                with st.expander("Show Hindi Text"):
                    hindi_text = response.get("Hindi Summary", "Hindi text not available.")
                    
                    # Format the text for better readability
                    paragraphs = hindi_text.split("। ")
                    
                    for paragraph in paragraphs:
                        if paragraph.strip():
                            # Add a period if it doesn't end with one
                            if not paragraph.strip().endswith("।"):
                                paragraph += "।"
                            st.markdown(f"<p style='font-size: 16px; margin-bottom: 10px;'>{paragraph}</p>", unsafe_allow_html=True)
                
                st.markdown("</div>", unsafe_allow_html=True)
                
                st.markdown("<div class='section-divider'></div>", unsafe_allow_html=True)
            
            # Display articles
            st.markdown("<h3 class='sub-header'>News Articles</h3>", unsafe_allow_html=True)
            articles = response.get("Articles", [])
            
            if not articles:
                st.info("No articles found for this company.")
            else:
                for i, article in enumerate(articles):
                    display_article(article, i)
            
            st.markdown("<div class='section-divider'></div>", unsafe_allow_html=True)
            
            # Display comparative analysis
            st.markdown("<h3 class='sub-header'>Comparative Analysis</h3>", unsafe_allow_html=True)
            
            # Display topic overlap
            topic_data = response["Comparative Sentiment Score"]["Topic Overlap"]
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("<div class='card'>", unsafe_allow_html=True)
                st.markdown("<h4>Common Topics</h4>", unsafe_allow_html=True)
                
                common_topics = topic_data.get("Common Topics Across All", [])
                if common_topics:
                    for topic in common_topics:
                        st.markdown(f"<span class='topic-tag'>{topic}</span>", unsafe_allow_html=True)
                else:
                    st.info("No common topics found across articles.")
                
                st.markdown("</div>", unsafe_allow_html=True)
            
            with col2:
                st.markdown("<div class='card'>", unsafe_allow_html=True)
                st.markdown("<h4>Coverage Comparison</h4>", unsafe_allow_html=True)
                
                comparisons = response["Comparative Sentiment Score"].get("Coverage Differences", [])
                if comparisons:
                    for i, comparison in enumerate(comparisons[:3]):  # Show only top 3 comparisons
                        st.markdown(f"<p><strong>{i+1}.</strong> {comparison.get('Comparison', '')}</p>", unsafe_allow_html=True)
                        st.markdown(f"<p class='info-text'>{comparison.get('Impact', '')}</p>", unsafe_allow_html=True)
                else:
                    st.info("No comparative insights available.")
                
                st.markdown("</div>", unsafe_allow_html=True)
            
            # Display full comparison in expander
            with st.expander("View All Comparisons"):
                comparisons = response["Comparative Sentiment Score"].get("Coverage Differences", [])
                for i, comparison in enumerate(comparisons):
                    st.markdown(f"<p><strong>{i+1}.</strong> {comparison.get('Comparison', '')}</p>", unsafe_allow_html=True)
                    st.markdown(f"<p class='info-text'>{comparison.get('Impact', '')}</p>", unsafe_allow_html=True)
                    st.markdown("<hr>", unsafe_allow_html=True)
            
            # Show JSON in example format if requested
            if show_json:
                st.markdown("<div class='section-divider'></div>", unsafe_allow_html=True)
                st.markdown("<h3 class='sub-header'>Example JSON Format</h3>", unsafe_allow_html=True)
                
                # Get the formatted JSON
                json_output = generate_example_output(company_name)
                
                # Display the JSON in a code block
                st.code(json_output, language="json")
else:
    # Display placeholder
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("<h3 class='sub-header'>Enter a Company Name to Begin Analysis</h3>", unsafe_allow_html=True)
    st.markdown("""
        <p class='info-text'>
        This application will:
        </p>
        <ul class='info-text'>
            <li>Extract news articles from multiple sources</li>
            <li>Analyze sentiment (positive, negative, neutral)</li>
            <li>Identify key topics in each article</li>
            <li>Perform comparative analysis across articles</li>
            <li>Generate Hindi speech output summarizing the findings</li>
        </ul>
    """, unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)
    
    # Sample output image
    st.image("https://miro.medium.com/max/1400/1*Ger-949PgQnaje2oa9XMdw.png", caption="Sample sentiment analysis visualization")

# Footer
st.markdown("<div class='section-divider'></div>", unsafe_allow_html=True)
st.markdown("<p class='info-text' style='text-align: center;'>News Summarization & Text-to-Speech Application | Developed with Streamlit and FastAPI</p>", unsafe_allow_html=True) 