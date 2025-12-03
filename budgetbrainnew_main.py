import streamlit as st
import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import create_agent
from datetime import datetime
import sys

# Import your custom tools
sys.path.append(os.path.dirname(__file__))
from tools.finance_web_search import search_financial_news
from tools.stock_market_search import (
    fetch_stock_data,
    predict_stock_price,
    analyze_trend,
    calculate_stoploss
)
from tools.budget_tool import analyze_budget

# Load environment variables
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="BudgetBrain - AI Financial Assistant",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for futuristic design
st.markdown("""
<style>
    /* Import futuristic font */
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;500;700;900&family=Rajdhani:wght@300;400;500;600;700&display=swap');
    
    /* Main background with gradient */
    .stApp {
        background: linear-gradient(135deg, #0a0e27 0%, #1a1f3a 50%, #0f1419 100%);
        font-family: 'Rajdhani', sans-serif;
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Header styling */
    .main-header {
        text-align: center;
        padding: 2rem 0 1rem 0;
        background: linear-gradient(90deg, #00d4ff 0%, #7b2ff7 50%, #f107a3 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        font-family: 'Orbitron', sans-serif;
        font-size: 3.5rem;
        font-weight: 900;
        letter-spacing: 3px;
        text-transform: uppercase;
        animation: glow 2s ease-in-out infinite alternate;
        margin-bottom: 0.5rem;
    }
    
    @keyframes glow {
        from {
            text-shadow: 0 0 10px #00d4ff, 0 0 20px #00d4ff, 0 0 30px #00d4ff;
            filter: brightness(1);
        }
        to {
            text-shadow: 0 0 20px #7b2ff7, 0 0 30px #7b2ff7, 0 0 40px #7b2ff7;
            filter: brightness(1.2);
        }
    }
    
    .tagline {
        text-align: center;
        color: #00d4ff;
        font-size: 1.2rem;
        font-weight: 300;
        letter-spacing: 2px;
        margin-bottom: 2rem;
        animation: fadeIn 1.5s ease-in;
    }
    
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(-10px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    /* Chat container */
    .chat-container {
        max-width: 900px;
        margin: 0 auto;
        padding: 1rem;
        height: 60vh;
        overflow-y: auto;
        background: rgba(15, 20, 35, 0.6);
        border-radius: 20px;
        border: 1px solid rgba(0, 212, 255, 0.3);
        backdrop-filter: blur(10px);
        box-shadow: 0 8px 32px 0 rgba(0, 212, 255, 0.2);
        margin-bottom: 1rem;
    }
    
    /* Scrollbar styling */
    .chat-container::-webkit-scrollbar {
        width: 8px;
    }
    
    .chat-container::-webkit-scrollbar-track {
        background: rgba(15, 20, 35, 0.4);
        border-radius: 10px;
    }
    
    .chat-container::-webkit-scrollbar-thumb {
        background: linear-gradient(180deg, #00d4ff, #7b2ff7);
        border-radius: 10px;
    }
    
    /* Message bubbles */
    .user-message {
        background: linear-gradient(135deg, #7b2ff7 0%, #f107a3 100%);
        color: white;
        padding: 1rem 1.5rem;
        border-radius: 20px 20px 5px 20px;
        margin: 1rem 0;
        margin-left: auto;
        max-width: 75%;
        float: right;
        clear: both;
        box-shadow: 0 4px 15px rgba(123, 47, 247, 0.4);
        animation: slideInRight 0.3s ease-out;
        font-size: 1rem;
        line-height: 1.6;
    }
    
    @keyframes slideInRight {
        from {
            opacity: 0;
            transform: translateX(20px);
        }
        to {
            opacity: 1;
            transform: translateX(0);
        }
    }
    
    .assistant-message {
        background: linear-gradient(135deg, #0f1419 0%, #1a2332 100%);
        color: #00d4ff;
        padding: 1rem 1.5rem;
        border-radius: 20px 20px 20px 5px;
        margin: 1rem 0;
        margin-right: auto;
        max-width: 75%;
        float: left;
        clear: both;
        border: 1px solid rgba(0, 212, 255, 0.3);
        box-shadow: 0 4px 15px rgba(0, 212, 255, 0.2);
        animation: slideInLeft 0.3s ease-out;
        font-size: 1rem;
        line-height: 1.6;
    }
    
    @keyframes slideInLeft {
        from {
            opacity: 0;
            transform: translateX(-20px);
        }
        to {
            opacity: 1;
            transform: translateX(0);
        }
    }
    
    .message-icon {
        font-size: 1.2rem;
        margin-right: 0.5rem;
        display: inline-block;
        vertical-align: middle;
    }
    
    .timestamp {
        font-size: 0.75rem;
        opacity: 0.7;
        margin-top: 0.5rem;
        font-style: italic;
    }
    
    /* Input area */
    .stTextInput > div > div > input {
        background: rgba(15, 20, 35, 0.8) !important;
        border: 2px solid rgba(0, 212, 255, 0.4) !important;
        border-radius: 25px !important;
        color: #00d4ff !important;
        padding: 1rem 1.5rem !important;
        font-size: 1rem !important;
        font-family: 'Rajdhani', sans-serif !important;
        transition: all 0.3s ease !important;
    }
    
    .stTextInput > div > div > input:focus {
        border-color: #7b2ff7 !important;
        box-shadow: 0 0 20px rgba(123, 47, 247, 0.4) !important;
        outline: none !important;
    }
    
    .stTextInput > div > div > input::placeholder {
        color: rgba(0, 212, 255, 0.5) !important;
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(135deg, #00d4ff 0%, #7b2ff7 100%) !important;
        color: white !important;
        border: none !important;
        border-radius: 25px !important;
        padding: 0.75rem 2rem !important;
        font-family: 'Orbitron', sans-serif !important;
        font-weight: 600 !important;
        font-size: 1rem !important;
        letter-spacing: 1px !important;
        cursor: pointer !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 4px 15px rgba(0, 212, 255, 0.4) !important;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 20px rgba(123, 47, 247, 0.6) !important;
    }
    
    /* Feature cards */
    .feature-card {
        background: rgba(15, 20, 35, 0.6);
        border: 1px solid rgba(0, 212, 255, 0.3);
        border-radius: 15px;
        padding: 1rem;
        margin: 0.5rem;
        text-align: center;
        transition: all 0.3s ease;
        backdrop-filter: blur(10px);
    }
    
    .feature-card:hover {
        transform: translateY(-5px);
        border-color: #7b2ff7;
        box-shadow: 0 8px 25px rgba(123, 47, 247, 0.4);
    }
    
    .feature-icon {
        font-size: 2rem;
        margin-bottom: 0.5rem;
    }
    
    .feature-title {
        color: #00d4ff;
        font-weight: 600;
        font-size: 1rem;
        margin-bottom: 0.3rem;
    }
    
    .feature-desc {
        color: #8ba3c7;
        font-size: 0.85rem;
    }
    
    /* Loading animation */
    .loading {
        display: inline-block;
        width: 20px;
        height: 20px;
        border: 3px solid rgba(0, 212, 255, 0.3);
        border-radius: 50%;
        border-top-color: #00d4ff;
        animation: spin 1s ease-in-out infinite;
    }
    
    @keyframes spin {
        to { transform: rotate(360deg); }
    }
    
    /* Welcome message */
    .welcome-message {
        text-align: center;
        color: #8ba3c7;
        padding: 2rem;
        font-size: 1.1rem;
        line-height: 1.8;
    }
    
    .capabilities {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
        gap: 1rem;
        margin-top: 2rem;
        padding: 0 2rem;
    }
    
    /* Clear chat button */
    .clear-chat-btn {
        position: fixed;
        top: 20px;
        right: 20px;
        z-index: 1000;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'messages' not in st.session_state:
    st.session_state.messages = []
    
if 'agent_executor' not in st.session_state:
    # Initialize LLM
    
    # Define tools
    
    
    # Create agent prompt with enhanced context awareness
    prompt = """You are BudgetBrain 🧠, an advanced AI Financial Intelligence Assistant created to revolutionize financial decision-making.

Your capabilities include:
- 📊 Real-time stock market analysis for Indian stocks (NSE)
- 📈 Price predictions using advanced ML models
- 📰 Latest financial news with sentiment analysis
- 💰 Budget planning and optimization
- 📉 Trend analysis and risk assessment
- 🎯 Stop-loss calculations for risk management

Core Principles:
1. Always maintain conversation context and remember previous interactions
2. Provide empathetic, human-like responses with appropriate emojis
3. Explain financial concepts clearly for all knowledge levels
4. Give actionable insights with specific reasoning
5. Use tools intelligently based on user intent
6. For Indian stocks, always use the .NS suffix (e.g., RELIANCE.NS).

TOOL SELECTION RULES (EXTREMELY IMPORTANT):

1. Budget Analysis → always use analyze_budget when the user mentions:
- income
- salary
- expenses
- savings
- budget
- spending
- monthly saving
- affordability
- purchase planning (car, house, bike, phone, laptop, etc.)
- “how much should I save”
- “can I afford”
- “how long will it take to save”
- “financial goal”
- “large purchase”
- calculating savings needed for future goal

2. NEVER use search_financial_news for personal finance questions.
Only use it when the user asks:
- “latest news”
- “current financial updates”
- “market news”
- “what's happening in finance”
- queries about external events, NOT personal budgeting.

3. Stock tools should ONLY be used for:
- stock names (RELIANCE, TCS, INFY) or any officially recognised stock symbol registered in NSE.
- words like “predict”, “forecast”, “trend”, “stoploss”, “price”, “chart” or any word relevant with Stock Market.
- Execute 'fetch_stock_data', 'predict_stock_price' when user asks only for prediction.
- Execute 'fetch_stock_data', 'predict_stock_price' and 'calculate_stoploss' when user asks to calculate stop loss.
- Execute 'fetch_stock_data', 'predict_stock_price', 'calculate_stoploss' and 'analyze_trend' when user asks for trend analysis as well including prediction and stop loss.
- Summarise and give informational insights to user

4. If none of the above tools are relevant → respond normally WITHOUT tools.

5. When in doubt between Budget v/s News v/s Stock -> Ask user clarifying question and create a friendly conversation with deep empathy to understand the exact issue.

Your job:
- understand context
- classify the intent correctly
- choose the right tool
- never hallucinate
- be empathetic and helpful

When analyzing user queries:
- Budget queries (income, expenses, savings) → use analyze_budget
- Stock data requests → use fetch_stock_data
- Price predictions → use predict_stock_price
- Trend analysis → use analyze_trend
- Stop-loss calculations → use calculate_stoploss
- Latest news → use search_financial_news

Always be conversational, supportive, and provide context-aware responses that reference previous messages when relevant."""
    tools = [
        search_financial_news,
        fetch_stock_data,
        predict_stock_price,
        analyze_trend,
        calculate_stoploss,
        analyze_budget
    ]

    llm = ChatGoogleGenerativeAI(
        api_key=os.getenv("GEMINI_API_KEY"),
        model="gemini-2.5-flash",
        temperature=0.7
    )

    # Create agent with safety wrapper
    try:
        if "agent" not in st.session_state:
            st.session_state.agent = create_agent(
                model=llm,
                tools=tools,
                system_prompt=prompt
            )
        if "messages" not in st.session_state:
            st.session_state.messages = []
        
    except Exception as e:
        # fallback: store None and show helpful error later
        st.session_state.agent = None
        st.error(f"⚠️ Could not initialize agent: {str(e)}")
# Header
st.markdown('<h1 class="main-header">🧠 BudgetBrain, created with 💓 by RJ</h1>', unsafe_allow_html=True)
st.markdown('<p class="tagline">Your AI-Powered Financial Intelligence Partner</p>', unsafe_allow_html=True)

# Clear chat button
col1, col2, col3 = st.columns([6, 1, 1])
with col3:
    if st.button("🗑️ Clear", key="clear_chat"):
        st.session_state.messages = []
        st.rerun()

# Display welcome message if no chat history
if len(st.session_state.messages) == 0:
    st.markdown("""
    <div class="welcome-message">
        <h2 style="color: #00d4ff; font-family: 'Orbitron', sans-serif;">Welcome to the Future of Financial Intelligence</h2>
        <p>I'm BudgetBrain, your AI companion for smart financial decisions. Let's explore what I can do for you:</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown('<div class="capabilities">', unsafe_allow_html=True)
    
    features = [
        ("📊", "Stock Analysis", "Get real-time data on Indian stocks"),
        ("📈", "Price Predictions", "AI-powered future price forecasts"),
        ("📰", "Financial News", "Latest news with sentiment analysis"),
        ("💰", "Budget Planning", "Optimize your income & expenses"),
        ("📉", "Trend Analysis", "Identify bullish/bearish patterns"),
        ("🎯", "Risk Management", "Calculate optimal stop-loss levels")
    ]
    
    for icon, title, desc in features:
        st.markdown(f"""
        <div class="feature-card">
            <div class="feature-icon">{icon}</div>
            <div class="feature-title">{title}</div>
            <div class="feature-desc">{desc}</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)
else:
    # Display chat history
    st.markdown('<div class="chat-container">', unsafe_allow_html=True)
    
    for message in st.session_state.messages:
        role = message["role"]
        content = message["content"]
        timestamp = message.get("timestamp", "")
        
        if role == "user":
            st.markdown(f"""
            <div class="user-message">
                <span class="message-icon">👤</span>
                <strong>You:</strong> {content}
                <div class="timestamp">{timestamp}</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="assistant-message">
                <span class="message-icon">🧠</span>
                <strong>BudgetBrain:</strong><br>{content}
                <div class="timestamp">{timestamp}</div>
            </div>
            """, unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

# Chat input
st.markdown("<br>", unsafe_allow_html=True)

# Create columns for input layout
col1, col2 = st.columns([9, 1])

with col1:
    user_input = st.text_input(
        "Message BudgetBrain...",
        key="user_input",
        placeholder="Ask me about stocks, budgets, predictions, or financial news...",
        label_visibility="collapsed"
    )

with col2:
    send_button = st.button("Send", key="send_btn", use_container_width=True)

# ---------------------- STREAMING RESPONSE HANDLING ----------------------
# ---------------------- FIXED STREAMING HANDLING ----------------------

if (user_input and send_button) or (user_input and user_input != st.session_state.get("last_input", "")):

    st.session_state.last_input = user_input

    timestamp = datetime.now().strftime("%I:%M %p")

    # Append user message
    st.session_state.messages.append({
        "role": "user",
        "content": user_input,
        "timestamp": timestamp
    })

    with st.chat_message("assistant"):
        message_placeholder = st.empty()

        try:
    # -------- DIRECT AGENT RESPONSE --------
            response = st.session_state.agent.invoke(
                {"messages": [{"role": m["role"], "content": m["content"]} for m in st.session_state.messages]}
            )

            # Extract the clean text content
            if isinstance(response, dict) and "messages" in response:
                last_message = response["messages"][-1]
                
                # Handle different content types (v1.0 uses content_blocks)
                if hasattr(last_message, 'content'):
                    content = last_message.content
                else:
                    content = last_message.get("content", "")
                
                # If content is a list of blocks, extract text from the first block
                if isinstance(content, list) and len(content) > 0:
                    if isinstance(content[0], dict) and 'text' in content[0]:
                        final_output = content[0]['text']  # Extract the text field
                    else:
                        final_output = str(content[0])
                else:
                    final_output = str(content)
                
                # Remove markdown asterisks and emojis for cleaner display
                import re
                final_output = re.sub(r'\*+', '', final_output)  # Remove all asterisks
                final_output = re.sub(r'[^\w\s\-\.\,\:\;\!\?\(\)₹]', '', final_output)  # Remove emojis
                
            else:
                final_output = str(response)

            # Display final output (no streaming)
            message_placeholder.markdown(final_output)

            # Save assistant message
            st.session_state.messages.append({
                "role": "assistant",
                "content": final_output,
                "timestamp": datetime.now().strftime("%I:%M %p")
            })

        except Exception as e:
            err = f"⚠️ Error: {str(e)}"
            message_placeholder.markdown(err)
            st.session_state.messages.append({
                "role": "assistant",
                "content": err,
                "timestamp": datetime.now().strftime("%I:%M %p")
            })

        st.rerun()


# Footer with example queries
st.markdown("<br><br>", unsafe_allow_html=True)
st.markdown("""
<div style="text-align: center; color: #8ba3c7; font-size: 0.85rem; padding: 1rem; border-top: 1px solid rgba(0, 212, 255, 0.2);">
    <strong>Try asking:</strong> "Analyze RELIANCE stock" • "Predict TCS price for 15 days" • "Latest AI news" • "My income is 50000 and expenses are 30000"
    <br><br>
    <span style="color: #7b2ff7;">⚡ Powered by Advanced AI • Built for Financial Excellence</span>
</div>
""", unsafe_allow_html=True)

