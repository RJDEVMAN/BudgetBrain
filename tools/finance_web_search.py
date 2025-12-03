# finance_web_search.py
from langchain_tavily import TavilySearch
from langchain_groq import ChatGroq
# unify tool import style
from langchain_core.tools import tool
from dotenv import load_dotenv
import os
import logging
load_dotenv()
logger = logging.getLogger(__name__)

GROQ_KEY = os.getenv("GROQ_API_KEY")
TAVILY_KEY = os.getenv("TAVILY_API_KEY")

if not GROQ_KEY:
    logger.warning("GROQ_API_KEY not set. LLM calls may fail.")
if not TAVILY_KEY:
    logger.warning("TAVILY_API_KEY not set. Tavily searches may fail.")

llm = ChatGroq(
    api_key=GROQ_KEY,
    model="llama-3.1-8b-instant",
    temperature=0.5
)

tavily_search = TavilySearch(
    max_results=5,
    tavily_api_key=TAVILY_KEY,
    topic="finance"
)


@tool
def search_financial_news(query: str) -> str:
    """Search latest financial news and analyze sentiment."""
    try:
        if not query or not isinstance(query, str):
            return "❌ Invalid query. Provide a search string."

        search_results = tavily_search.run(query)

        news_prompt = f"""
User Query: {query}
Search Results: {search_results}

Analyze the sentiment and relevancy of this news. Respond in natural language with empathy.
Also include short references (source names) for each important item.
"""
        response = llm.invoke(news_prompt)
        if hasattr(response, "content"):
            return response.content
        return str(response)
    except Exception as e:
        logger.error(f"Error in search_financial_news: {str(e)}")
        return f"❌ Error in search_financial_news: {str(e)}"

