# news-summarization-sentiment-analysis-
# News Summarizer

## Overview
A news summarization web app that extracts and summarizes news articles using NLP techniques. The app is built with Streamlit and leverages the BART model for text summarization and DistilBERT for sentiment analysis.

## Features
- Summarize news articles using BART.
- Perform sentiment analysis on news content.
- API support for summarization and sentiment analysis.
- Dockerized setup for easy deployment.

## Installation

### 1. Clone the Repository
```sh
 git clone https://github.com/your-username/news-summarizer.git
 cd news-summarizer
```

### 2. Set Up a Virtual Environment
```sh
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
```

### 3. Install Dependencies
```sh
pip install -r requirements.txt
```

## Usage

### Running the App
```sh
streamlit run app.py
```
The app will be available at: `http://localhost:8501`

### API Usage
Run the API server:
```sh
python api.py
```
Endpoints:
- `POST /summarize` - Summarizes an article.
- `POST /sentiment` - Analyzes sentiment of an article.

Example request:
```sh
curl -X POST http://localhost:5000/summarize -H "Content-Type: application/json" -d '{"text": "Your news article here."}'
```

## Docker Setup
Build and run the application using Docker:
```sh
docker build -t news-summarizer .
docker run -p 8501:8501 news-summarizer
```

## File Structure
```
news-summarizer/
│── app.py         # Main Streamlit app
│── api.py         # Flask API for summarization & sentiment analysis
│── utils.py       # Utility functions
│── requirements.txt  # Dependencies
│── README.md      # Documentation
│── Dockerfile     # Docker support
└── models/        # Pre-trained models
```

## License
This project is licensed under the MIT License.
