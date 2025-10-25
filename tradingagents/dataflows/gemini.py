import google.generativeai as genai
import os
from .config import get_config


def _initialize_gemini():
    """Initialize Gemini API with the API key from environment"""
    api_key = os.getenv("GOOGLE_API_KEY")
    if api_key:
        genai.configure(api_key=api_key)


def get_stock_news_gemini(query, start_date, end_date):
    """Get stock news using Gemini with Google Search grounding"""
    _initialize_gemini()
    config = get_config()
    
    model = genai.GenerativeModel(
        model_name=config["quick_think_llm"],
        tools='google_search_retrieval'
    )
    
    prompt = f"Can you search Social Media for {query} from {start_date} to {end_date}? Make sure you only get the data posted during that period."
    
    response = model.generate_content(
        prompt,
        generation_config=genai.types.GenerationConfig(
            temperature=1.0,
            max_output_tokens=4096,
            top_p=1.0,
        )
    )
    
    return response.text


def get_global_news_gemini(curr_date, look_back_days=7, limit=5):
    """Get global news using Gemini with Google Search grounding"""
    _initialize_gemini()
    config = get_config()
    
    model = genai.GenerativeModel(
        model_name=config["quick_think_llm"],
        tools='google_search_retrieval'
    )
    
    prompt = f"Can you search global or macroeconomics news from {look_back_days} days before {curr_date} to {curr_date} that would be informative for trading purposes? Make sure you only get the data posted during that period. Limit the results to {limit} articles."
    
    response = model.generate_content(
        prompt,
        generation_config=genai.types.GenerationConfig(
            temperature=1.0,
            max_output_tokens=4096,
            top_p=1.0,
        )
    )
    
    return response.text


def get_fundamentals_gemini(ticker, curr_date):
    """Get fundamental data using Gemini with Google Search grounding"""
    _initialize_gemini()
    config = get_config()
    
    model = genai.GenerativeModel(
        model_name=config["quick_think_llm"],
        tools='google_search_retrieval'
    )
    
    prompt = f"Can you search Fundamental for discussions on {ticker} during of the month before {curr_date} to the month of {curr_date}. Make sure you only get the data posted during that period. List as a table, with PE/PS/Cash flow/ etc"
    
    response = model.generate_content(
        prompt,
        generation_config=genai.types.GenerationConfig(
            temperature=1.0,
            max_output_tokens=4096,
            top_p=1.0,
        )
    )
    
    return response.text
