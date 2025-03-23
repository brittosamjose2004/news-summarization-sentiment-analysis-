import requests
import re
import os
import json
import time
from typing import List, Dict, Any, Tuple, Optional
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.cluster.util import cosine_distance
import networkx as nx
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from collections import Counter
from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer
from deep_translator import GoogleTranslator
from gtts import gTTS
import pyttsx3

# Download necessary NLTK data
import nltk
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('punkt')
    nltk.download('stopwords')

# Initialize sentiment analyzer
vader_analyzer = SentimentIntensityAnalyzer()

# Initialize advanced sentiment model
sentiment_model_name = "nlptown/bert-base-multilingual-uncased-sentiment"
sentiment_tokenizer = AutoTokenizer.from_pretrained(sentiment_model_name)
sentiment_model = AutoModelForSequenceClassification.from_pretrained(sentiment_model_name)
advanced_sentiment = pipeline("sentiment-analysis", model=sentiment_model, tokenizer=sentiment_tokenizer)

# Initialize translator
translator = GoogleTranslator(source='en', target='hi')

class NewsArticle:
    def __init__(self, title: str, url: str, content: str, summary: str = "", source: str = "", 
                 date: str = "", sentiment: str = "", topics: List[str] = None):
        self.title = title
        self.url = url
        self.content = content
        self.summary = summary if summary else self.generate_summary(content)
        self.source = source
        self.date = date
        self.sentiment = sentiment if sentiment else self.analyze_sentiment(content, title)
        self.topics = topics if topics else self.extract_topics(content)
        
    def to_dict(self) -> Dict[str, Any]:
        return {
            "title": self.title,
            "url": self.url,
            "content": self.content,
            "summary": self.summary,
            "source": self.source,
            "date": self.date,
            "sentiment": self.sentiment,
            "topics": self.topics
        }
        
    @staticmethod
    def analyze_sentiment(text: str, title: str = "") -> str:
        """
        Analyze sentiment using a combination of methods for more accurate results.
        We give more weight to the title sentiment and use advanced model when possible.
        """
        # Set thresholds for VADER sentiment
        threshold_positive = 0.05  # Default 0.05
        threshold_negative = -0.05  # Default -0.05
        
        # Use VADER for basic sentiment analysis on both title and content
        try:
            title_scores = vader_analyzer.polarity_scores(title) if title else {'compound': 0}
            content_scores = vader_analyzer.polarity_scores(text)
            
            # Weight the title more heavily (title sentiment is often more reliable)
            title_weight = 0.6 if title else 0
            content_weight = 1.0 - title_weight
            
            compound_score = (title_weight * title_scores['compound']) + (content_weight * content_scores['compound'])
            
            # Try to use the advanced model for additional insight (for short texts)
            advanced_result = None
            advanced_score = 0
            
            try:
                # Use title + first part of content for advanced model
                sample_text = title + ". " + text[:300] if title else text[:300]
                advanced_result = advanced_sentiment(sample_text)[0]
                
                # Map advanced model results to a -1 to 1 scale similar to VADER
                label = advanced_result['label']
                confidence = advanced_result['score']
                
                # Map the 1-5 star rating to a -1 to 1 scale
                if label == '1 star' or label == '2 stars':
                    advanced_score = -confidence
                elif label == '4 stars' or label == '5 stars':
                    advanced_score = confidence
                else:  # 3 stars is neutral
                    advanced_score = 0 
                
                # Combine VADER and advanced model scores
                # Give more weight to advanced model when confidence is high
                if confidence > 0.8:
                    compound_score = (0.4 * compound_score) + (0.6 * advanced_score)
                else:
                    compound_score = (0.7 * compound_score) + (0.3 * advanced_score)
                    
            except Exception as e:
                print(f"Advanced sentiment analysis failed: {str(e)}")
                # Continue with just VADER if advanced model fails
                pass
            
            # Fine-grained sentiment mapping
            if compound_score >= 0.3:
                return "Positive"
            elif compound_score >= threshold_positive:
                return "Slightly Positive"
            elif compound_score <= -0.3:
                return "Negative"
            elif compound_score <= threshold_negative:
                return "Slightly Negative"
            else:
                return "Neutral"
                
        except Exception as e:
            print(f"Sentiment analysis error: {str(e)}")
            return "Neutral"  # Default fallback
    
    @staticmethod
    def generate_summary(text: str, num_sentences: int = 5) -> str:
        # Generate summary using extractive summarization
        if not text or len(text) < 100:
            return text
        
        # Tokenize sentences
        sentences = sent_tokenize(text)
        if len(sentences) <= num_sentences:
            return text
            
        # Calculate sentence similarity and rank them
        similarity_matrix = build_similarity_matrix(sentences)
        scores = nx.pagerank(nx.from_numpy_array(similarity_matrix))
        
        # Select top sentences
        ranked_sentences = sorted(((scores[i], s) for i, s in enumerate(sentences)), reverse=True)
        summary_sentences = [ranked_sentences[i][1] for i in range(min(num_sentences, len(ranked_sentences)))]
        
        # Maintain original order
        original_order = []
        for sentence in sentences:
            if sentence in summary_sentences and sentence not in original_order:
                original_order.append(sentence)
                if len(original_order) >= num_sentences:
                    break
        
        return " ".join(original_order)
    
    @staticmethod
    def extract_topics(text: str, num_topics: int = 5) -> List[str]:
        # Extract key topics from text based on term frequency
        stop_words = set(stopwords.words('english'))
        words = word_tokenize(text.lower())
        
        # Filter out stopwords and short words
        filtered_words = [word for word in words if word.isalpha() and word not in stop_words and len(word) > 3]
        
        # Count word frequencies
        word_counts = Counter(filtered_words)
        
        # Return most common words as topics
        topics = [word for word, _ in word_counts.most_common(num_topics)]
        return topics

def build_similarity_matrix(sentences: List[str]) -> np.ndarray:
    """Build similarity matrix for sentences based on cosine similarity."""
    # Number of sentences
    n = len(sentences)
    
    # Initialize similarity matrix
    similarity_matrix = np.zeros((n, n))
    
    # Calculate similarity between each pair of sentences
    for i in range(n):
        for j in range(n):
            if i != j:
                similarity_matrix[i][j] = sentence_similarity(sentences[i], sentences[j])
                
    return similarity_matrix

def sentence_similarity(sent1: str, sent2: str) -> float:
    """Calculate similarity between two sentences using cosine similarity."""
    # Tokenize sentences
    words1 = [word.lower() for word in word_tokenize(sent1) if word.isalpha()]
    words2 = [word.lower() for word in word_tokenize(sent2) if word.isalpha()]
    
    # Get all unique words
    all_words = list(set(words1 + words2))
    
    # Create word vectors
    vector1 = [1 if word in words1 else 0 for word in all_words]
    vector2 = [1 if word in words2 else 0 for word in all_words]
    
    # Calculate cosine similarity
    if not any(vector1) or not any(vector2):
        return 0.0
    
    return 1 - cosine_distance(vector1, vector2)

def search_news(company_name: str, num_articles: int = 10) -> List[NewsArticle]:
    """Search for news articles about a given company."""
    # List to store articles
    articles = []
    
    # Define search queries and news sources
    search_queries = [
        f"{company_name} news",
        f"{company_name} financial news",
        f"{company_name} business news",
        f"{company_name} recent news",
        f"{company_name} company news",
        f"{company_name} stock",
        f"{company_name} market"
    ]
    
    # Updated news sources with more reliable sources
    news_sources = [
        {
            "base_url": "https://finance.yahoo.com/quote/",
            "article_patterns": ["news", "finance", "articles"],
            "direct_access": True
        },
        {
            "base_url": "https://www.reuters.com/search/news?blob=",
            "article_patterns": ["article", "business", "companies", "markets"],
            "direct_access": False
        },
        {
            "base_url": "https://www.marketwatch.com/search?q=",
            "article_patterns": ["story", "articles", "news"],
            "direct_access": False
        },
        {
            "base_url": "https://www.fool.com/search?q=",
            "article_patterns": ["article", "investing", "stock"],
            "direct_access": False
        },
        {
            "base_url": "https://seekingalpha.com/search?q=",
            "article_patterns": ["article", "news", "stock", "analysis"],
            "direct_access": False
        },
        {
            "base_url": "https://www.zacks.com/search.php?q=",
            "article_patterns": ["stock", "research", "analyst"],
            "direct_access": False
        },
        {
            "base_url": "https://economictimes.indiatimes.com/search?q=",
            "article_patterns": ["articleshow", "news", "industry"],
            "direct_access": False
        },
        {
            "base_url": "https://www.bloomberg.com/search?query=",
            "article_patterns": ["news", "articles"],
            "direct_access": False
        }
    ]
    
    print(f"Starting search for news about {company_name}...")
    
    # Search each source with each query until we have enough articles
    for query in search_queries:
        if len(articles) >= num_articles:
            break
            
        for source in news_sources:
            if len(articles) >= num_articles:
                break
                
            try:
                source_base = source["base_url"]
                article_patterns = source["article_patterns"]
                direct_access = source["direct_access"]
                
                # Construct search URL
                if direct_access:
                    # Try to fetch the stock symbol for Yahoo Finance
                    if "yahoo" in source_base:
                        try:
                            # First try the company name directly (for known tickers)
                            search_url = f"{source_base}{company_name}/news"
                            print(f"Trying direct ticker access: {search_url}")
                            
                            # Fetch to check if valid
                            headers = {
                                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
                            }
                            test_response = requests.get(search_url, headers=headers, timeout=10)
                            
                            # If we got a 404, try searching for the symbol first
                            if test_response.status_code == 404:
                                print("Company name not a valid ticker, searching for symbol...")
                                symbol_url = f"https://finance.yahoo.com/lookup?s={company_name}"
                                symbol_response = requests.get(symbol_url, headers=headers, timeout=10)
                                
                                if symbol_response.status_code == 200:
                                    symbol_soup = BeautifulSoup(symbol_response.text, 'html.parser')
                                    # Try to find the first stock symbol result
                                    symbol_row = symbol_soup.select_one("tr.data-row0")
                                    if symbol_row:
                                        symbol_cell = symbol_row.select_one("td:first-child a")
                                        if symbol_cell:
                                            symbol = symbol_cell.text.strip()
                                            search_url = f"{source_base}{symbol}/news"
                                            print(f"Found symbol {symbol}, using URL: {search_url}")
                        except Exception as e:
                            print(f"Error getting stock symbol: {str(e)}")
                            search_url = f"{source_base}{company_name}/news"
                    else:
                        search_url = f"{source_base}{company_name}/news"
                else:
                    search_url = f"{source_base}{query.replace(' ', '+')}"
                
                print(f"Searching {search_url}")
                
                # Fetch search results with retry mechanism
                max_retries = 3
                retry_count = 0
                response = None
                
                while retry_count < max_retries:
                    try:
                        headers = {
                            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
                            "Accept": "text/html,application/xhtml+xml,application/xml",
                            "Accept-Language": "en-US,en;q=0.9",
                            "Referer": "https://www.google.com/"
                        }
                        response = requests.get(search_url, headers=headers, timeout=15)
                        if response.status_code == 200:
                            break
                        retry_count += 1
                        print(f"Retry {retry_count}/{max_retries} for {search_url} (status: {response.status_code})")
                        time.sleep(1)  # Short delay before retry
                    except Exception as e:
                        retry_count += 1
                        print(f"Request error (attempt {retry_count}/{max_retries}): {str(e)}")
                        time.sleep(1)
                
                if not response or response.status_code != 200:
                    print(f"Failed to fetch results from {search_url} after {max_retries} attempts")
                    continue
                    
                soup = BeautifulSoup(response.text, 'html.parser')
                
                # Extract article links - using more flexible patterns
                links = soup.find_all('a', href=True)
                article_links = []
                
                # Domain for resolving relative URLs
                domain = response.url.split('/')[0] + '//' + response.url.split('/')[2]
                print(f"Domain for resolving URLs: {domain}")
                
                for link in links:
                    href = link['href']
                    link_text = link.text.strip()
                    
                    # Skip empty links or navigation elements
                    if not link_text or len(link_text) < 10 or href.startswith('#'):
                        continue
                    
                    # Check if the link matches any of our article patterns
                    is_article_link = False
                    for pattern in article_patterns:
                        if pattern in href.lower():
                            is_article_link = True
                            break
                    
                    # Check for the company name in link text or URL (less restrictive now)
                    contains_company = (
                        company_name.lower() in link_text.lower() or 
                        company_name.lower() in href.lower()
                    )
                    
                    if is_article_link or contains_company:
                        # Convert relative URLs to absolute
                        if href.startswith('/'):
                            href = f"{domain}{href}"
                        elif not href.startswith(('http://', 'https://')):
                            href = f"{domain}/{href}"
                        
                        # Avoid duplicates
                        if href not in article_links:
                            article_links.append(href)
                            print(f"Found potential article: {link_text[:50]}... at {href}")
                
                print(f"Found {len(article_links)} potential article links from {search_url}")
                
                # Process each article link
                for link in article_links[:5]:  # Increased from 3 to 5
                    if len(articles) >= num_articles:
                        break
                        
                    try:
                        print(f"Fetching article: {link}")
                        article_response = requests.get(link, headers=headers, timeout=15)
                        
                        if article_response.status_code != 200:
                            print(f"Failed to fetch article: {article_response.status_code}")
                            continue
                            
                        article_soup = BeautifulSoup(article_response.text, 'html.parser')
                        
                        # Extract article title - more robust method
                        title = None
                        
                        # Try different elements that could contain the title
                        for title_tag in ['h1', 'h2', '.headline', '.title', 'title']:
                            if title:
                                break
                                
                            if title_tag.startswith('.'):
                                elements = article_soup.select(title_tag)
                            else:
                                elements = article_soup.find_all(title_tag)
                                
                            for element in elements:
                                candidate = element.text.strip()
                                if len(candidate) > 5 and len(candidate) < 200:  # Reasonable title length
                                    title = candidate
                                    break
                        
                        if not title:
                            print("Could not find a suitable title")
                            continue
                        
                        # Check if title contains company name (case insensitive)
                        if company_name.lower() not in title.lower():
                            # Try alternative check - sometimes the title doesn't explicitly mention the company
                            meta_description = article_soup.find('meta', attrs={'name': 'description'}) or \
                                               article_soup.find('meta', attrs={'property': 'og:description'})
                                
                            if meta_description and 'content' in meta_description.attrs:
                                meta_text = meta_description['content']
                                if company_name.lower() not in meta_text.lower():
                                    # One more check in the page content
                                    page_text = article_soup.get_text().lower()
                                    company_mentions = page_text.count(company_name.lower())
                                    if company_mentions < 2:  # Require at least 2 mentions
                                        print(f"Article doesn't seem to be about {company_name}: {title}")
                                        continue
                        
                        # Extract article content - improved method
                        content = ""
                        
                        # Try multiple content extraction strategies
                        content_containers = []
                        
                        # 1. Look for article/main content containers
                        for container in ['article', 'main', '.article-body', '.story-body', '.story-content', 
                                         '.article-content', '.content-body', '.entry-content']:
                            if container.startswith('.'):
                                elements = article_soup.select(container)
                            else:
                                elements = article_soup.find_all(container)
                                
                            content_containers.extend(elements)
                        
                        # 2. If no specific containers, fallback to div with article-like classes
                        if not content_containers:
                            for div in article_soup.find_all('div', class_=True):
                                classes = div.get('class', [])
                                for cls in classes:
                                    if any(term in cls.lower() for term in ['article', 'story', 'content', 'body', 'text']):
                                        content_containers.append(div)
                                        break
                        
                        # 3. Extract paragraphs from containers
                        processed_paragraphs = set()  # To avoid duplicates
                        
                        for container in content_containers:
                            for p in container.find_all('p'):
                                p_text = p.text.strip()
                                # Avoid very short or duplicate paragraphs
                                if len(p_text) > 30 and p_text not in processed_paragraphs:
                                    content += p_text + " "
                                    processed_paragraphs.add(p_text)
                        
                        # 4. If still no content, try all paragraphs
                        if not content:
                            for p in article_soup.find_all('p'):
                                p_text = p.text.strip()
                                if len(p_text) > 30 and p_text not in processed_paragraphs:
                                    content += p_text + " "
                                    processed_paragraphs.add(p_text)
                        
                        content = content.strip()
                        
                        # Skip if content is too short
                        if len(content) < 300:  # Reduced from 500 to be less restrictive
                            print(f"Article content too short: {len(content)} characters")
                            continue
                        
                        # Extract source name - more robust method
                        source = None
                        
                        # Try to get from meta tags
                        meta_site_name = article_soup.find('meta', attrs={'property': 'og:site_name'})
                        if meta_site_name and 'content' in meta_site_name.attrs:
                            source = meta_site_name['content']
                        else:
                            # Extract from URL
                            try:
                                from urllib.parse import urlparse
                                parsed_url = urlparse(link)
                                source = parsed_url.netloc
                            except:
                                source = response.url.split('/')[2]
                        
                        # Extract date - improved method
                        date = ""
                        
                        # Try multiple date extraction strategies
                        # 1. Look for time element
                        date_tag = article_soup.find('time')
                        
                        # 2. Look for meta tags with date
                        if not date and (not date_tag or not date_tag.get('datetime')):
                            for meta_name in ['article:published_time', 'date', 'publish-date', 'article:modified_time']:
                                meta_date = article_soup.find('meta', attrs={'property': meta_name}) or \
                                           article_soup.find('meta', attrs={'name': meta_name})
                                
                                if meta_date and 'content' in meta_date.attrs:
                                    date = meta_date['content']
                                    break
                        
                        # 3. Look for spans/divs with date-related classes
                        if not date:
                            date_classes = ['date', 'time', 'published', 'posted', 'datetime']
                            for cls in date_classes:
                                elements = article_soup.find_all(['span', 'div', 'p'], class_=lambda x: x and cls.lower() in x.lower())
                                if elements:
                                    date = elements[0].text.strip()
                                    break
                        
                        # If we got this far, we have a valid article
                        print(f"Successfully extracted article: {title}")
                        
                        # Create article object and add to list
                        article = NewsArticle(
                            title=title,
                            url=link,
                            content=content,
                            source=source,
                            date=date
                        )
                        
                        # Check if similar article already exists to avoid duplicates
                        is_duplicate = False
                        for existing_article in articles:
                            if sentence_similarity(existing_article.title, title) > 0.7:  # Lowered threshold
                                is_duplicate = True
                                print(f"Found duplicate article: {title}")
                                break
                        
                        if not is_duplicate:
                            articles.append(article)
                            print(f"Added article: {title}")
                        
                    except Exception as e:
                        print(f"Error processing article {link}: {str(e)}")
                        continue
            
            except Exception as e:
                print(f"Error searching {source_base} with query {query}: {str(e)}")
                continue
    
    # If we couldn't find enough articles, create some dummy articles to prevent errors
    if not articles and num_articles > 0:
        print(f"No articles found for {company_name}. Creating a dummy article to prevent errors.")
        
        dummy_article = NewsArticle(
            title=f"{company_name} Information",
            url="#",
            content=f"Information about {company_name} was not found or could not be retrieved. This is a placeholder.",
            source="System",
            date="",
            sentiment="Neutral",
            topics=["information", "company", "placeholder"]
        )
        
        articles.append(dummy_article)
    
    # Return collected articles
    print(f"Returning {len(articles)} articles for {company_name}")
    return articles[:num_articles]

def analyze_article_sentiment(article: NewsArticle) -> Dict[str, Any]:
    """Perform detailed sentiment analysis on an article."""
    # Use VADER for paragraph-level sentiment
    paragraphs = article.content.split('\n')
    paragraph_sentiments = []
    
    overall_scores = {
        'pos': 0,
        'neg': 0,
        'neu': 0,
        'compound': 0
    }
    
    for paragraph in paragraphs:
        if len(paragraph.strip()) < 20:  # Skip short paragraphs
            continue
            
        scores = vader_analyzer.polarity_scores(paragraph)
        paragraph_sentiments.append({
            'text': paragraph[:100] + '...' if len(paragraph) > 100 else paragraph,
            'scores': scores
        })
        
        overall_scores['pos'] += scores['pos']
        overall_scores['neg'] += scores['neg']
        overall_scores['neu'] += scores['neu']
        overall_scores['compound'] += scores['compound']
    
    num_paragraphs = len(paragraph_sentiments)
    if num_paragraphs > 0:
        overall_scores['pos'] /= num_paragraphs
        overall_scores['neg'] /= num_paragraphs
        overall_scores['neu'] /= num_paragraphs
        overall_scores['compound'] /= num_paragraphs
    
    # Use advanced model for overall sentiment
    try:
        # Truncate content if too long
        truncated_content = article.content[:512] if len(article.content) > 512 else article.content
        advanced_result = advanced_sentiment(truncated_content)[0]
        advanced_sentiment_label = advanced_result['label']
        advanced_confidence = advanced_result['score']
    except Exception as e:
        print(f"Error with advanced sentiment analysis: {str(e)}")
        advanced_sentiment_label = "Error"
        advanced_confidence = 0.0
    
    # Determine final sentiment
    if overall_scores['compound'] >= 0.05:
        final_sentiment = "Positive"
    elif overall_scores['compound'] <= -0.05:
        final_sentiment = "Negative"
    else:
        final_sentiment = "Neutral"
    
    return {
        'article_title': article.title,
        'overall_sentiment': final_sentiment,
        'vader_scores': overall_scores,
        'advanced_sentiment': {
            'label': advanced_sentiment_label,
            'confidence': advanced_confidence
        },
        'paragraph_analysis': paragraph_sentiments,
        'positive_ratio': overall_scores['pos'],
        'negative_ratio': overall_scores['neg'],
        'neutral_ratio': overall_scores['neu']
    }

def perform_comparative_analysis(articles: List[NewsArticle]) -> Dict[str, Any]:
    """Perform comparative analysis across multiple articles."""
    # Sentiment distribution with expanded categories
    sentiment_counts = {
        "Positive": 0,
        "Slightly Positive": 0,
        "Neutral": 0,
        "Slightly Negative": 0,
        "Negative": 0
    }
    
    for article in articles:
        if article.sentiment in sentiment_counts:
            sentiment_counts[article.sentiment] += 1
        else:
            # Fallback for any unexpected sentiment values
            sentiment_counts["Neutral"] += 1
    
    # Topic analysis
    all_topics = []
    for article in articles:
        all_topics.extend(article.topics)
    
    topic_counts = Counter(all_topics)
    common_topics = [topic for topic, count in topic_counts.most_common(10)]
    
    # Identify unique topics per article
    unique_topics_by_article = {}
    for i, article in enumerate(articles):
        other_articles_topics = []
        for j, other_article in enumerate(articles):
            if i != j:
                other_articles_topics.extend(other_article.topics)
        
        unique_topics = [topic for topic in article.topics if topic not in other_articles_topics]
        unique_topics_by_article[i] = unique_topics
    
    # Generate comparisons
    comparisons = []
    
    # If we have more than one article, generate meaningful comparisons
    if len(articles) > 1:
        for i in range(len(articles) - 1):
            for j in range(i + 1, len(articles)):
                article1 = articles[i]
                article2 = articles[j]
                
                # Compare sentiments - more nuanced now with new categories
                if article1.sentiment != article2.sentiment:
                    # Group sentiments for better comparison
                    sent1_group = get_sentiment_group(article1.sentiment)
                    sent2_group = get_sentiment_group(article2.sentiment)
                    
                    if sent1_group != sent2_group:
                        comparison = {
                            "Articles": [article1.title, article2.title],
                            "Comparison": f"'{article1.title}' presents a {sent1_group.lower()} view ({article1.sentiment}), while '{article2.title}' has a {sent2_group.lower()} view ({article2.sentiment}).",
                            "Impact": "This difference in sentiment highlights varying perspectives on the company's situation."
                        }
                        comparisons.append(comparison)
                    else:
                        # Even if in same group, note the difference if one is stronger
                        if "Slightly" in article1.sentiment and "Slightly" not in article2.sentiment or \
                           "Slightly" in article2.sentiment and "Slightly" not in article1.sentiment:
                            stronger = article1 if "Slightly" not in article1.sentiment else article2
                            weaker = article2 if stronger == article1 else article1
                            
                            comparison = {
                                "Articles": [stronger.title, weaker.title],
                                "Comparison": f"'{stronger.title}' expresses a stronger {sent1_group.lower()} sentiment ({stronger.sentiment}) than '{weaker.title}' ({weaker.sentiment}).",
                                "Impact": "The difference in intensity suggests varying degrees of confidence about the company."
                            }
                            comparisons.append(comparison)
                
                # Compare topics
                common_topics_between_two = set(article1.topics).intersection(set(article2.topics))
                if common_topics_between_two:
                    comparison = {
                        "Articles": [article1.title, article2.title],
                        "Comparison": f"Both articles discuss {', '.join(common_topics_between_two)}.",
                        "Impact": "The common topics indicate key areas of focus around the company."
                    }
                    comparisons.append(comparison)
                
                # Compare unique topics
                unique_to_article1 = set(article1.topics) - set(article2.topics)
                unique_to_article2 = set(article2.topics) - set(article1.topics)
                
                if unique_to_article1 and unique_to_article2:
                    comparison = {
                        "Articles": [article1.title, article2.title],
                        "Comparison": f"'{article1.title}' uniquely covers {', '.join(unique_to_article1)}, while '{article2.title}' focuses on {', '.join(unique_to_article2)}.",
                        "Impact": "Different sources emphasize varying aspects of the company, offering a broader perspective."
                    }
                    comparisons.append(comparison)
    else:
        # If we only have one article, create a dummy comparison
        if articles:
            article = articles[0]
            topics_str = ", ".join(article.topics[:3]) if article.topics else "no specific topics"
            sentiment_group = get_sentiment_group(article.sentiment)
            
            comparisons = [
                {
                    "Comparison": f"Only found one article: '{article.title}' with a {article.sentiment.lower()} sentiment ({sentiment_group} overall).",
                    "Impact": f"Limited coverage focused on {topics_str}. More articles would provide a more balanced view."
                },
                {
                    "Comparison": f"The article discusses {topics_str} in relation to {article.source}.",
                    "Impact": "Single source reporting limits perspective. Consider searching for additional sources."
                }
            ]
    
    # Generate overall sentiment analysis
    # Combine slightly positive with positive and slightly negative with negative for summary
    pos_count = sentiment_counts["Positive"] + sentiment_counts["Slightly Positive"]
    neg_count = sentiment_counts["Negative"] + sentiment_counts["Slightly Negative"]
    neu_count = sentiment_counts["Neutral"]
    total = pos_count + neg_count + neu_count
    
    # For display, we'll keep detailed counts but summarize the analysis text
    if total == 0:
        final_analysis = "No sentiment data available."
    else:
        pos_ratio = pos_count / total
        neg_ratio = neg_count / total
        
        # Show more details on the sentiment breakdown
        sentiment_detail = []
        if sentiment_counts["Positive"] > 0:
            sentiment_detail.append(f"{sentiment_counts['Positive']} strongly positive")
        if sentiment_counts["Slightly Positive"] > 0:
            sentiment_detail.append(f"{sentiment_counts['Slightly Positive']} slightly positive")
        if sentiment_counts["Neutral"] > 0:
            sentiment_detail.append(f"{sentiment_counts['Neutral']} neutral")
        if sentiment_counts["Slightly Negative"] > 0:
            sentiment_detail.append(f"{sentiment_counts['Slightly Negative']} slightly negative")
        if sentiment_counts["Negative"] > 0:
            sentiment_detail.append(f"{sentiment_counts['Negative']} strongly negative")
            
        sentiment_breakdown = ", ".join(sentiment_detail)
        
        if pos_ratio > 0.6:
            final_analysis = f"The company has primarily positive coverage ({pos_count}/{total} articles positive: {sentiment_breakdown}). This suggests a favorable market perception."
        elif neg_ratio > 0.6:
            final_analysis = f"The company has primarily negative coverage ({neg_count}/{total} articles negative: {sentiment_breakdown}). This could indicate challenges or controversies."
        elif pos_ratio > neg_ratio:
            final_analysis = f"The company has mixed coverage with a positive lean ({sentiment_breakdown})."
        elif neg_ratio > pos_ratio:
            final_analysis = f"The company has mixed coverage with a negative lean ({sentiment_breakdown})."
        else:
            final_analysis = f"The company has balanced coverage ({sentiment_breakdown})."
    
    # If we only have the dummy article, customize the final analysis
    if len(articles) == 1 and articles[0].url == "#":
        final_analysis = "Limited news data available. The analysis is based on a placeholder article."
    
    return {
        "Sentiment Distribution": sentiment_counts,
        "Common Topics": common_topics,
        "Topic Overlap": {
            "Common Topics Across All": common_topics[:5],
            "Unique Topics By Article": unique_topics_by_article
        },
        "Coverage Differences": comparisons[:10],  # Limit to top 10 comparisons
        "Final Sentiment Analysis": final_analysis
    }

def get_sentiment_group(sentiment: str) -> str:
    """Group sentiments into broader categories for comparison."""
    if sentiment in ["Positive", "Slightly Positive"]:
        return "Positive"
    elif sentiment in ["Negative", "Slightly Negative"]:
        return "Negative"
    else:
        return "Neutral"

def translate_to_hindi(text: str) -> str:
    """Translate text to Hindi using deep_translator."""
    try:
        # Split text into chunks if too long (Google Translator has a limit)
        max_chunk_size = 4500  # deep_translator's GoogleTranslator has a limit of 5000 chars
        chunks = [text[i:i+max_chunk_size] for i in range(0, len(text), max_chunk_size)]
        
        translated_chunks = []
        for chunk in chunks:
            # Translate the chunk
            translated = translator.translate(chunk)
            translated_chunks.append(translated)
            time.sleep(0.5)  # Short delay to avoid rate limiting
        
        return ''.join(translated_chunks)
    except Exception as e:
        print(f"Translation error: {str(e)}")
        # Fallback to simple placeholder for Hindi text if translation fails
        return "अनुवाद त्रुटि हुई।" # "Translation error occurred" in Hindi

def text_to_speech(text: str, output_file: str = 'output.mp3') -> str:
    """Convert text to speech in Hindi."""
    try:
        # Ensure output directory exists
        output_dir = os.path.dirname(output_file)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            print(f"Ensuring output directory exists: {output_dir}")
        
        # If text is too short, add some padding to avoid TTS errors
        if len(text.strip()) < 5:
            text = text + " " + "नमस्कार" * 3  # Add some padding text
            print("Text was too short, adding padding")
        
        print(f"Attempting to generate TTS for text of length {len(text)} characters")
        
        # For long texts, split into chunks for better TTS quality
        if len(text) > 3000:
            print("Text is long, splitting into chunks for better TTS quality")
            
            # Split at sentence boundaries
            sentences = re.split(r'(।|\.|\?|\!)', text)
            chunks = []
            current_chunk = ""
            
            # Combine sentences into chunks of appropriate size
            for i in range(0, len(sentences), 2):
                if i+1 < len(sentences):  # Make sure we have the punctuation part
                    sentence = sentences[i] + sentences[i+1]
                else:
                    sentence = sentences[i]
                
                if len(current_chunk) + len(sentence) < 3000:
                    current_chunk += sentence
                else:
                    if current_chunk:
                        chunks.append(current_chunk)
                    current_chunk = sentence
            
            if current_chunk:  # Add the last chunk
                chunks.append(current_chunk)
            
            print(f"Split text into {len(chunks)} chunks for TTS processing")
            
            # Process each chunk and combine into one audio file
            temp_files = []
            for i, chunk in enumerate(chunks):
                temp_output = f"{output_file}.part{i}.mp3"
                try:
                    # Try gTTS for each chunk
                    tts = gTTS(text=chunk, lang='hi', slow=False)
                    tts.save(temp_output)
                    if os.path.exists(temp_output) and os.path.getsize(temp_output) > 0:
                        temp_files.append(temp_output)
                    else:
                        print(f"Failed to create chunk {i} with gTTS")
                        raise Exception(f"gTTS failed for chunk {i}")
                except Exception as e:
                    print(f"Error with gTTS for chunk {i}: {str(e)}")
                    break
            
            # If we have temp files, combine them
            if temp_files:
                try:
                    # Use pydub to concatenate audio files
                    from pydub import AudioSegment
                    combined = AudioSegment.empty()
                    for temp_file in temp_files:
                        audio = AudioSegment.from_mp3(temp_file)
                        combined += audio
                    
                    combined.export(output_file, format="mp3")
                    
                    # Clean up temp files
                    for temp_file in temp_files:
                        try:
                            os.remove(temp_file)
                        except:
                            pass
                    
                    print(f"Successfully combined {len(temp_files)} audio chunks into {output_file}")
                    return output_file
                except Exception as e:
                    print(f"Error combining audio files: {str(e)}")
                    # Try to return the first chunk at least
                    if os.path.exists(temp_files[0]):
                        import shutil
                        shutil.copy(temp_files[0], output_file)
                        print(f"Returning first chunk as fallback: {output_file}")
                        return output_file
        
        # Method 1: Use gTTS for Hindi text-to-speech (for shorter texts or if chunking failed)
        try:
            print("Trying to use gTTS...")
            tts = gTTS(text=text, lang='hi', slow=False)
            tts.save(output_file)
            
            # Verify the file was created and is not empty
            if os.path.exists(output_file) and os.path.getsize(output_file) > 0:
                print(f"Successfully created audio file with gTTS: {output_file} (size: {os.path.getsize(output_file)} bytes)")
                return output_file
            else:
                print(f"gTTS created a file but it may be empty or invalid: {output_file}")
                raise Exception("Generated audio file is empty or invalid")
                
        except Exception as e:
            print(f"gTTS error: {str(e)}")
            
            # Method 2: Fallback to pyttsx3
            try:
                print("Falling back to pyttsx3...")
                engine = pyttsx3.init()
                # Try to find a Hindi voice, or use default
                voices = engine.getProperty('voices')
                found_hindi_voice = False
                
                for voice in voices:
                    print(f"Checking voice: {voice.name}")
                    if 'hindi' in voice.name.lower():
                        print(f"Found Hindi voice: {voice.name}")
                        engine.setProperty('voice', voice.id)
                        found_hindi_voice = True
                        break
                
                if not found_hindi_voice:
                    print("No Hindi voice found, using default voice")
                
                engine.save_to_file(text, output_file)
                engine.runAndWait()
                
                # Verify the file was created and is not empty
                if os.path.exists(output_file) and os.path.getsize(output_file) > 0:
                    print(f"Successfully created audio file with pyttsx3: {output_file} (size: {os.path.getsize(output_file)} bytes)")
                    return output_file
                else:
                    print(f"pyttsx3 created a file but it may be empty or invalid: {output_file}")
                    raise Exception("Generated audio file is empty or invalid")
                    
            except Exception as e2:
                print(f"pyttsx3 error: {str(e2)}")
                
                # If all TTS methods fail, create a simple notification sound as fallback
                try:
                    print("Both TTS methods failed. Creating a simple audio notification instead.")
                    # Generate a simple beep sound as a fallback (1 second, 440Hz)
                    import numpy as np
                    from scipy.io import wavfile
                    
                    sample_rate = 44100
                    duration = 1  # seconds
                    t = np.linspace(0, duration, int(sample_rate * duration))
                    
                    # Generate a simple tone
                    frequency = 440  # Hz (A4 note)
                    data = np.sin(2 * np.pi * frequency * t) * 32767
                    data = data.astype(np.int16)
                    
                    # Convert output_file from mp3 to wav
                    wav_output_file = output_file.replace('.mp3', '.wav')
                    wavfile.write(wav_output_file, sample_rate, data)
                    
                    print(f"Created simple audio notification: {wav_output_file}")
                    return wav_output_file
                    
                except Exception as e3:
                    print(f"Failed to create fallback audio: {str(e3)}")
                    return ""
                
                return ""
    except Exception as e:
        print(f"TTS error: {str(e)}")
        return ""

def prepare_final_report(company_name: str, articles: List[NewsArticle], 
                         comparative_analysis: Dict[str, Any]) -> Dict[str, Any]:
    """Prepare final report in the required format."""
    article_data = []
    
    for article in articles:
        article_data.append({
            "Title": article.title,
            "Summary": article.summary,
            "Sentiment": article.sentiment,
            "Topics": article.topics
        })
    
    # Prepare a more detailed summary for TTS with actual content from articles
    summary_text = f"{company_name} के बारे में समाचार विश्लेषण। "
    
    # Add information about the number of articles found
    summary_text += f"कुल {len(articles)} लेख मिले। "
    
    # Add sentiment distribution
    sentiment_counts = comparative_analysis["Sentiment Distribution"]
    pos_count = sentiment_counts["Positive"] + sentiment_counts["Slightly Positive"]
    neg_count = sentiment_counts["Negative"] + sentiment_counts["Slightly Negative"]
    neu_count = sentiment_counts["Neutral"]
    
    if pos_count > 0 or neg_count > 0 or neu_count > 0:
        sentiment_detail = []
        if sentiment_counts["Positive"] > 0:
            sentiment_detail.append(f"{sentiment_counts['Positive']} पूर्ण सकारात्मक")
        if sentiment_counts["Slightly Positive"] > 0:
            sentiment_detail.append(f"{sentiment_counts['Slightly Positive']} हल्का सकारात्मक")
        if sentiment_counts["Neutral"] > 0:
            sentiment_detail.append(f"{sentiment_counts['Neutral']} तटस्थ")
        if sentiment_counts["Slightly Negative"] > 0:
            sentiment_detail.append(f"{sentiment_counts['Slightly Negative']} हल्का नकारात्मक")
        if sentiment_counts["Negative"] > 0:
            sentiment_detail.append(f"{sentiment_counts['Negative']} पूर्ण नकारात्मक")
            
        summary_text += f"भावना विश्लेषण: {', '.join(sentiment_detail)}। "
    
    # Add common topics with more detail
    common_topics = comparative_analysis["Common Topics"][:5]
    if common_topics:
        summary_text += f"मुख्य विषय हैं: {', '.join(common_topics)}। "
        
        # Add more context about the common topics
        summary_text += "इन विषयों के बारे में लेखों में यह कहा गया है: "
        
        # Find sentences related to common topics in the articles
        topic_sentences = []
        for topic in common_topics[:3]:  # Focus on top 3 topics
            found = False
            for article in articles:
                if topic in article.content.lower():
                    # Find sentences containing this topic
                    sentences = sent_tokenize(article.content)
                    for sentence in sentences:
                        if topic in sentence.lower() and len(sentence) < 150:
                            topic_sentences.append(f"{topic} के बारे में: {sentence}")
                            found = True
                            break
                    if found:
                        break
        
        if topic_sentences:
            summary_text += " ".join(topic_sentences[:3]) + " "
    
    # Add article summaries
    summary_text += "लेखों का सारांश: "
    for i, article in enumerate(articles[:3]):  # Include up to 3 articles
        summary_text += f"लेख {i+1}: {article.title}. {article.summary[:200]}... "
        
        # Add sentiment for this specific article
        summary_text += f"इस लेख का भावना: {article.sentiment}. "
    
    # Add final sentiment analysis
    summary_text += comparative_analysis["Final Sentiment Analysis"]
    
    # Translate the detailed summary to Hindi
    hindi_summary = translate_to_hindi(summary_text)
    
    # Format the response according to the required format
    return {
        "Company": company_name,
        "Articles": article_data,
        "Comparative Sentiment Score": comparative_analysis,
        "Final Sentiment Analysis": comparative_analysis["Final Sentiment Analysis"],
        "Hindi Summary": hindi_summary
    } 