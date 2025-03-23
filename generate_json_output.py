#!/usr/bin/env python

import requests
import json
import sys

def generate_json_output(company_name, api_url="http://localhost:8000"):
    """
    Generate output in the example format for the given company.
    
    Args:
        company_name (str): Name of the company to analyze
        api_url (str): Base URL of the API
        
    Returns:
        str: Formatted JSON string
    """
    try:
        # Make API request to get the analysis data
        url = f"{api_url}/api/complete_analysis"
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

if __name__ == "__main__":
    # Get company name from command line arguments or prompt for it
    if len(sys.argv) > 1:
        company_name = sys.argv[1]
    else:
        company_name = input("Enter company name: ")
    
    print(f"Input:\nCompany Name: {company_name}")
    print("Output:", generate_json_output(company_name)) 