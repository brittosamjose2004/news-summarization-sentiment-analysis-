from fastapi import FastAPI, HTTPException, Response, File, UploadFile, Form
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import os
import json
import uuid
import asyncio
import uvicorn  
from utils import (search_news, analyze_article_sentiment, perform_comparative_analysis, 
                  translate_to_hindi, text_to_speech, prepare_final_report, NewsArticle)

# Initialize FastAPI app
app = FastAPI(
    title="News Summarization and TTS API",
    description="API for extracting news, performing sentiment analysis, and generating Hindi TTS audio",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods
    allow_headers=["*"],  # Allow all headers
)

# Define request/response models
class CompanyRequest(BaseModel):
    company_name: str

class TextToSpeechRequest(BaseModel):
    text: str
    output_filename: Optional[str] = None

class SentimentAnalysisRequest(BaseModel):
    articles: List[Dict[str, Any]]

class NewsResponse(BaseModel):
    articles: List[Dict[str, Any]]

class SentimentResponse(BaseModel):
    sentiment_analysis: Dict[str, Any]

class TextToSpeechResponse(BaseModel):
    audio_file: str
    text: str

# Create a directory for audio files if it doesn't exist
os.makedirs("audio_files", exist_ok=True)

# API endpoints
@app.get("/")
async def root():
    """Root endpoint to check if API is running."""
    return {"message": "News Summarization and TTS API is running"}

@app.post("/api/get_news", response_model=NewsResponse)
async def get_news(request: CompanyRequest):
    """Fetch news articles about a specific company."""
    try:
        company_name = request.company_name
        articles = search_news(company_name)
        
        if not articles:
            raise HTTPException(status_code=404, detail=f"No news articles found for {company_name}")
        
        # Convert NewsArticle objects to dictionaries
        article_data = [article.to_dict() for article in articles]
        
        return {"articles": article_data}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/analyze_sentiment", response_model=SentimentResponse)
async def analyze_sentiment(request: SentimentAnalysisRequest):
    """Analyze sentiment of provided articles."""
    try:
        # Convert dictionaries back to NewsArticle objects
        articles = []
        for article_dict in request.articles:
            article = NewsArticle(
                title=article_dict["title"],
                url=article_dict["url"],
                content=article_dict["content"],
                summary=article_dict.get("summary", ""),
                source=article_dict.get("source", ""),
                date=article_dict.get("date", ""),
                sentiment=article_dict.get("sentiment", ""),
                topics=article_dict.get("topics", [])
            )
            articles.append(article)
        
        # Perform detailed sentiment analysis for each article
        detailed_sentiment = [analyze_article_sentiment(article) for article in articles]
        
        # Perform comparative analysis
        comparative_analysis = perform_comparative_analysis(articles)
        
        return {
            "sentiment_analysis": {
                "detailed_sentiment": detailed_sentiment,
                "comparative_analysis": comparative_analysis
            }
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/generate_speech", response_model=TextToSpeechResponse)
async def generate_speech(request: TextToSpeechRequest):
    """Convert text to Hindi speech."""
    try:
        text = request.text
        
        # Generate a unique filename if not provided
        output_filename = request.output_filename
        if not output_filename:
            unique_id = uuid.uuid4().hex
            output_filename = f"audio_files/{unique_id}.mp3"
        elif not output_filename.startswith("audio_files/"):
            output_filename = f"audio_files/{output_filename}"
        
        # Translate text to Hindi
        hindi_text = translate_to_hindi(text)
        
        # Convert text to speech
        audio_file = text_to_speech(hindi_text, output_filename)
        
        if not audio_file:
            raise HTTPException(status_code=500, detail="Failed to generate audio file")
        
        return {
            "audio_file": audio_file,
            "text": hindi_text
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/complete_analysis")
async def complete_analysis(request: CompanyRequest):
    """Perform complete analysis for a company."""
    try:
        company_name = request.company_name
        
        # Log the start of analysis
        print(f"Starting complete analysis for company: {company_name}")
        
        # Step 1: Get news articles
        print("Step 1: Fetching news articles...")
        articles = search_news(company_name, num_articles=5)  # Increased from default 3 to 5
        print(f"Found {len(articles)} articles for {company_name}")
        
        if not articles:
            raise HTTPException(status_code=404, detail=f"No news articles found for {company_name}")
        
        # Step 2: Perform comparative analysis
        print("Step 2: Performing comparative analysis...")
        comparative_analysis = perform_comparative_analysis(articles)
        print("Comparative analysis completed")
        
        # Step 3: Prepare final report
        print("Step 3: Preparing final report...")
        final_report = prepare_final_report(company_name, articles, comparative_analysis)
        print("Final report prepared")
        
        # Step 4: Generate Hindi TTS
        print("Step 4: Generating Hindi TTS...")
        unique_id = uuid.uuid4().hex
        output_filename = f"audio_files/{unique_id}.mp3"
        
        # Use the Hindi summary for TTS
        hindi_text = final_report["Hindi Summary"]
        print(f"Converting Hindi text to speech (length: {len(hindi_text)} characters)")
        
        audio_file = text_to_speech(hindi_text, output_filename)
        
        # Format the response to match the example output exactly
        formatted_response = {
            "Company": company_name,
            "Articles": final_report["Articles"],
            "Comparative Sentiment Score": {
                "Sentiment Distribution": comparative_analysis["Sentiment Distribution"],
                "Coverage Differences": comparative_analysis["Coverage Differences"],
                "Topic Overlap": {
                    "Common Topics": comparative_analysis["Topic Overlap"]["Common Topics Across All"],
                }
            },
            "Final Sentiment Analysis": comparative_analysis["Final Sentiment Analysis"],
        }
        
        # Format the unique topics by article to match the expected output exactly
        unique_topics = comparative_analysis["Topic Overlap"]["Unique Topics By Article"]
        for article_idx, topics in unique_topics.items():
            article_num = int(article_idx) + 1
            formatted_response["Comparative Sentiment Score"]["Topic Overlap"][f"Unique Topics in Article {article_num}"] = topics
        
        # If we don't have more than 1 article, create some example comparisons to match format
        if len(articles) <= 1:
            formatted_response["Comparative Sentiment Score"]["Coverage Differences"] = [
                {
                    "Comparison": f"Only one article about {company_name} was found, limiting comparative analysis.",
                    "Impact": "Unable to compare coverage across multiple sources for more comprehensive insights."
                }
            ]
        
        # Add audio information
        if not audio_file:
            print("Warning: Failed to generate audio file")
            formatted_response["Audio"] = "Failed to generate audio"
        else:
            print(f"Audio file generated: {audio_file}")
            formatted_response["Audio"] = f"[Play Hindi Speech]"
            # Store the actual audio file path in a hidden field
            formatted_response["_audio_file_path"] = audio_file
        
        # Add the Hindi Summary to the response as well (needed for rendering in Streamlit)
        formatted_response["Hindi Summary"] = final_report["Hindi Summary"]
        
        print("Complete analysis finished successfully")
        return formatted_response
    
    except HTTPException as he:
        # Re-raise HTTP exceptions
        print(f"HTTP Exception: {he.detail}")
        raise
    
    except Exception as e:
        # For any other exception, provide detailed error information
        import traceback
        error_trace = traceback.format_exc()
        error_message = f"Error processing request: {str(e)}"
        print(f"ERROR: {error_message}")
        print(f"Traceback: {error_trace}")
        
        # Return a more user-friendly error message
        user_message = "An error occurred during analysis. "
        
        if "timeout" in str(e).lower():
            user_message += "There was a timeout when connecting to news sources. Please try again or try another company name."
        elif "connection" in str(e).lower():
            user_message += "There was a connection issue with one of the news sources. Please check your internet connection."
        elif "not found" in str(e).lower():
            user_message += f"No information could be found for {company_name}. Please try another company name."
        else:
            user_message += "Please try again with a different company name or check the server logs for more details."
        
        raise HTTPException(status_code=500, detail=user_message)

@app.get("/api/audio/{file_name}")
async def get_audio(file_name: str):
    """Serve audio files."""
    file_path = f"audio_files/{file_name}"
    
    # Make sure the audio_files directory exists
    os.makedirs("audio_files", exist_ok=True)
    
    if not os.path.exists(file_path):
        print(f"Audio file not found: {file_path}")
        # Check if any audio files exist in the directory
        audio_files = os.listdir("audio_files") if os.path.exists("audio_files") else []
        print(f"Available audio files: {audio_files}")
        raise HTTPException(status_code=404, detail=f"Audio file {file_name} not found")
    
    try:
        # Verify the file can be opened and is not corrupt
        with open(file_path, "rb") as f:
            file_size = os.path.getsize(file_path)
            print(f"Serving audio file: {file_path} (size: {file_size} bytes)")
            if file_size == 0:
                raise HTTPException(status_code=500, detail="Audio file is empty")
    except Exception as e:
        print(f"Error accessing audio file {file_path}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error accessing audio file: {str(e)}")
    
    # Set appropriate headers for audio file
    headers = {
        "Cache-Control": "no-cache, no-store, must-revalidate",
        "Pragma": "no-cache",
        "Expires": "0",
        "Content-Disposition": f"attachment; filename={file_name}"
    }
    
    # Determine the correct media type based on file extension
    media_type = "audio/mpeg"
    if file_name.lower().endswith(".wav"):
        media_type = "audio/wav"
    
    return FileResponse(
        path=file_path, 
        media_type=media_type, 
        headers=headers,
        filename=file_name
    )

@app.post("/api/example_format")
async def get_example_format(request: CompanyRequest):
    """
    Get analysis results in the example format specified.
    This endpoint provides results that exactly match the requested output format.
    """
    try:
        # Get the base analysis
        company_name = request.company_name
        result = await complete_analysis(request)
        
        # Format it to match the example output
        formatted_output = {
            "Company": result["Company"],
            "Articles": result["Articles"],
            "Comparative Sentiment Score": {
                "Sentiment Distribution": result["Comparative Sentiment Score"]["Sentiment Distribution"],
                "Coverage Differences": result["Comparative Sentiment Score"]["Coverage Differences"],
                "Topic Overlap": result["Comparative Sentiment Score"]["Topic Overlap"]
            },
            "Final Sentiment Analysis": result["Final Sentiment Analysis"],
            "Audio": "[Play Hindi Speech]" if result.get("Audio") else "No audio available"
        }
        
        return formatted_output
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating example format: {str(e)}")

if __name__ == "__main__":
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True) 