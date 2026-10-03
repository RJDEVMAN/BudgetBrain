BudgetBrain – AI-Powered Financial Advisor Assistant

BudgetBrain is an intelligent, agentic financial advisory system designed to help users track budgets, analyze expenses, predict stock prices, detect financial trends, and provide personalized insights using AI.
It combines LangChain Agents, LLM-driven reasoning, financial APIs, voice interaction, and smart automation—all inside a clean UI.

🚀 Features:-

1. Smart Budget Analyzer:

Accepts natural language inputs (e.g., "Ruturaj, income 50,000, rent 15,000, food 5,000...").
Automatically parses income, expenses, savings.
Generates pie charts, breakdowns, surplus/deficit analysis.
Stores history for long-term trend analysis.

2. Stock Price Prediction:

Uses live market data via yfinance.
Forecasts future prices using ML (Linear Regression).
Provides:
Prediction graph
Range estimates
Trend (bullish/bearish)
Stop-loss calculation
Agent-generated insights

3. Financial News Insight Engine:

Fetches latest financial news.
Runs sentiment analysis to show:
How news may affect stocks
Market mood (positive/neutral/negative)
Provides summary and impact scores.

4. Agentic AI System:
Powered by LangChain:
Budget Agent → classifies budgets, finds anomalies
Investment Agent → checks stock trend & gives forecasts
News Agent → sentiment + insights
Everything orchestrated through create_agent().
🏗️ Project Architecture:-
BudgetBrain/
  tools/
    budget_tool.py
    finance_web_search.py
    stock_market_search.py

  budgetbrainnew_main.py

  .gitignore

  README.md

  requirements.txt

Installation:-
  1. Clone the Repo:
    git clone [https://github.com/RJDEVMAN/BudgetBrain.git]
    cd BudgetBrain
  2. Create Virtual Environment:
     python -m venv venv
     source venv/bin/activate  # Mac/Linux
     venv\Scripts\activate     # Windows
  3. Install Dependencies:
     pip install -r requirements.txt
  4. Set Environment Variables (.env):
     GROQ_API_KEY=your_key
     GEMINI_API_KEY=your_key
  5. Run the Application:
     python main.py

 How It Works (Internally)
1. User gives query →
2. Classifier decides:
Budget Query?
Stock Query?
News Query?
Mixed Query?
3. LangChain Agent calls the right tool
4. Prediction/Analysis generated
5. Result displayed + spoken back to user

🛠️ Tech Stack:-
Backend:
Python
LangChain
yfinance

Frontend:
Streamlit

AI/ML:
LLM (Gemini for orchestration & Groq for tools)

Sentiment Analysis:
prophet FB model for stock. 

📈 Future Roadmap:-

 Add LSTM-based forecasting
 Build mobile app UI
 Add PDF document analysis for financial statements
 Reinforcement Learning for investment optimization
 Add personal finance goal tracking
