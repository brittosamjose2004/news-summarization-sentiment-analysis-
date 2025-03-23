"""
Healthcheck script to verify the functionality of all components of the application.
Run this script to check if all dependencies are correctly installed and working.
"""

import sys
import os
import time
import traceback

def run_checks():
    print("Starting health check for News Summarization and TTS Application...")
    checks_passed = 0
    checks_failed = 0
    
    # Check 1: Verify imports
    print("\n1. Checking imports...")
    try:
        # Standard libraries
        import json
        import re
        
        # Web and API dependencies
        import requests
        import fastapi
        import uvicorn
        import streamlit
        
        # Data processing
        import pandas
        import numpy
        import bs4
        
        # NLP
        import nltk
        import networkx
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
        
        # ML and Transformers
        import torch
        import transformers
        from transformers import pipeline
        
        # TTS and Translation
        import deep_translator
        from deep_translator import GoogleTranslator
        import gtts
        import pyttsx3
        
        print("✅ All imports successful.")
        checks_passed += 1
    except ImportError as e:
        print(f"❌ Import error: {str(e)}")
        print(f"Traceback: {traceback.format_exc()}")
        checks_failed += 1
    
    # Check 2: Verify NLTK data
    print("\n2. Checking NLTK data...")
    try:
        import nltk
        nltk.data.find('tokenizers/punkt')
        nltk.data.find('corpora/stopwords')
        print("✅ NLTK data verified.")
        checks_passed += 1
    except LookupError as e:
        print(f"❌ NLTK data error: {str(e)}")
        print("Trying to download necessary NLTK data...")
        try:
            nltk.download('punkt')
            nltk.download('stopwords')
            print("✅ NLTK data downloaded successfully.")
            checks_passed += 1
        except Exception as e:
            print(f"❌ Failed to download NLTK data: {str(e)}")
            checks_failed += 1
    
    # Check 3: Test translation
    print("\n3. Testing translation service...")
    try:
        from deep_translator import GoogleTranslator
        translator = GoogleTranslator(source='en', target='hi')
        text = "Hello, this is a test."
        translated = translator.translate(text)
        print(f"Original text: {text}")
        print(f"Translated text: {translated}")
        if translated and len(translated) > 0:
            print("✅ Translation service working.")
            checks_passed += 1
        else:
            print("❌ Translation returned empty result.")
            checks_failed += 1
    except Exception as e:
        print(f"❌ Translation error: {str(e)}")
        print(f"Traceback: {traceback.format_exc()}")
        checks_failed += 1
    
    # Check 4: Test TTS
    print("\n4. Testing Text-to-Speech service...")
    try:
        from gtts import gTTS
        test_text = "परीक्षण पाठ"  # "Test text" in Hindi
        test_file = 'test_audio.mp3'
        
        # Try gTTS
        tts = gTTS(text=test_text, lang='hi', slow=False)
        tts.save(test_file)
        
        if os.path.exists(test_file) and os.path.getsize(test_file) > 0:
            print("✅ gTTS service working.")
            # Clean up test file
            try:
                os.remove(test_file)
            except:
                pass
            checks_passed += 1
        else:
            print("❌ gTTS failed to generate a valid audio file.")
            checks_failed += 1
    except Exception as e:
        print(f"❌ Text-to-Speech error: {str(e)}")
        print(f"Traceback: {traceback.format_exc()}")
        checks_failed += 1
    
    # Check 5: Test sentiment analysis
    print("\n5. Testing sentiment analysis...")
    try:
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
        analyzer = SentimentIntensityAnalyzer()
        test_text = "This product is excellent and I love it!"
        scores = analyzer.polarity_scores(test_text)
        print(f"Sentiment scores for '{test_text}': {scores}")
        if 'compound' in scores:
            print("✅ Sentiment analysis working.")
            checks_passed += 1
        else:
            print("❌ Sentiment analysis returned unexpected result.")
            checks_failed += 1
    except Exception as e:
        print(f"❌ Sentiment analysis error: {str(e)}")
        print(f"Traceback: {traceback.format_exc()}")
        checks_failed += 1
    
    # Check 6: Test Transformers
    print("\n6. Testing Transformer models...")
    try:
        from transformers import pipeline
        sentiment_task = pipeline("sentiment-analysis", return_all_scores=False)
        result = sentiment_task("I love using this application!")
        print(f"Transformer sentiment analysis result: {result}")
        print("✅ Transformer models working.")
        checks_passed += 1
    except Exception as e:
        print(f"❌ Transformer models error: {str(e)}")
        print(f"Traceback: {traceback.format_exc()}")
        checks_failed += 1
    
    # Summary
    print("\n" + "="*50)
    print(f"Health Check Summary: {checks_passed} checks passed, {checks_failed} checks failed")
    
    if checks_failed == 0:
        print("\n✅ All systems operational! The application should run correctly.")
        return True
    else:
        print("\n❌ Some checks failed. Please review the errors above.")
        return False

if __name__ == "__main__":
    success = run_checks()
    if not success:
        sys.exit(1) 