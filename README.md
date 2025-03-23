# News Summarization and Text-to-Speech Application

A web-based application that extracts news articles related to companies, performs sentiment analysis, conducts comparative analysis, and generates a text-to-speech output in Hindi.

## Features 

- **News Extraction**: Scrapes at least 10 unique news articles about a given company using BeautifulSoup
- **Sentiment Analysis**: Analyzes the sentiment of each article (positive, negative, neutral)
- **Comparative Analysis**: Compares sentiment across articles to derive insights
- **Text-to-Speech**: Converts summarized content to Hindi speech
- **User Interface**: Simple web interface built with Streamlit
- **API Communication**: Backend and frontend communicate through APIs

## Project Structure

```
.
├── app.py              # Main Streamlit application
├── api.py              # API endpoints
├── utils.py            # Utility functions for scraping, sentiment analysis, etc.
├── healthcheck.py      # Script to verify all dependencies and services
├── requirements.txt    # Project dependencies
├── Dockerfile          # Docker configuration for deployment
├── Spacefile           # Hugging Face Spaces configuration
└── README.md           # Project documentation
```

## Setup Instructions

1. **Clone the repository**:
   ```
   git clone https://github.com/yourusername/news-summarization-tts.git
   cd news-summarization-tts
   ```

2. **Create a virtual environment** (recommended):
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```
   pip install -r requirements.txt
   ```

4. **Install system dependencies** (for text-to-speech functionality):
   - On Ubuntu/Debian:
     ```
     sudo apt-get install espeak ffmpeg
     ```
   - On Windows:
     Download and install espeak from http://espeak.sourceforge.net/download.html

5. **Run the healthcheck** (to verify all dependencies are working):
   ```
   python healthcheck.py
   ```

6. **Run the API server**:
   ```
   uvicorn api:app --reload
   ```

7. **Run the Streamlit application** (in a separate terminal):
   ```
   streamlit run app.py
   ```

## Models Used

- **News Summarization**: Extractive summarization using NLTK and NetworkX
- **Sentiment Analysis**: VADER for sentiment analysis and Hugging Face Transformers
- **Translation**: Google Translate API via deep-translator library
- **Text-to-Speech**: Google Text-to-Speech (gTTS) and pyttsx3 as fallback for Hindi conversion

## API Documentation

### Endpoints

- `POST /api/get_news`: Fetches news articles about a company
  - Request body: `{"company_name": "Tesla"}`
  - Returns a list of articles with metadata

- `POST /api/analyze_sentiment`: Performs sentiment analysis on articles
  - Request body: `{"articles": [article_list]}`
  - Returns sentiment analysis for each article

- `POST /api/generate_speech`: Converts text to Hindi speech
  - Request body: `{"text": "summarized_text"}`
  - Returns a URL to the generated audio file
  
- `POST /api/complete_analysis`: Performs complete analysis including fetching news, sentiment analysis, and generating speech
  - Request body: `{"company_name": "Tesla"}`
  - Returns complete analysis results

## Assumptions & Limitations

- The application scrapes publicly available news articles that don't require JavaScript rendering
- Sentiment analysis accuracy depends on the model used and may not capture context-specific nuances
- Hindi translation and TTS quality may vary based on technical terms
- The application requires an internet connection to fetch news articles and use cloud-based services

## Troubleshooting

If you encounter any issues:

1. Run the healthcheck script to verify all dependencies are working:
   ```
   python healthcheck.py
   ```

2. Check that you have all the required system dependencies installed (espeak, ffmpeg).

3. If you encounter issues with specific components:
   - Translation service requires an internet connection
   - Text-to-speech uses gTTS by default, but falls back to pyttsx3 if needed
   - Transformer models may take time to download on first run

## Deployment

This application is deployed on Hugging Face Spaces: [Link to deployment]

### Using Docker

You can also run the application using Docker:

```
docker build -t news-summarization-tts .
docker run -p 8501:8501 -p 8000:8000 news-summarization-tts
```

## Future Improvements

- Add support for more languages
- Implement advanced NLP techniques for better summarization
- Improve the user interface with more interactive visualizations
- Add historical data analysis for tracking sentiment over time
- Enhance TTS quality with dedicated Hindi speech models

## License

MIT 