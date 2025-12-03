from langchain_core.tools import tool
import re
import logging
from langchain_groq import ChatGroq
import os
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

llm = ChatGroq(
    api_key=os.getenv("GROQ_API_KEY"),
    model="llama-3.1-8b-instant",
    temperature=0.5
)

# ---------------------------
# 1. UNIVERSAL CURRENCY REGEX
# ---------------------------

CURRENCY_REGEX = r"(₹|rs\.?|rupees|inr|\$|usd|dollars|€|eur|£|gbp)"

# Matches: "$40,000", "₹50000", "USD 3000", "40k", "40000"
AMOUNT_REGEX = rf"({CURRENCY_REGEX})?\s*([\d,.]+k|\d[\d,]*)"

# ---------------------------
# 2. Smart extraction engine
# ---------------------------

def extract_budget_brain(query: str):
    query_lower = query.lower()

    currency = None
    numbers = []

    # Extract all amounts + currency
    for cur, amt in re.findall(AMOUNT_REGEX, query_lower):
        clean_val = amt.replace(",", "").replace("k", "000")
        try:
            val = float(clean_val)
            numbers.append(val)
            if not currency and cur:
                currency = cur
        except:
            continue

    if not currency:
        currency = "₹"  # default

    income = expenses = savings = None
    future_expense = None

    # --- KEYWORD / semantic classification ---
    if "income" in query_lower or "salary" in query_lower:
        income = numbers[0] if numbers else None

    if "expense" in query_lower or "spend" in query_lower or "cost" in query_lower:
        # If explicit expenses appear
        if income and len(numbers) >= 2:
            expenses = numbers[1]
        elif len(numbers) >= 1:
            expenses = numbers[0]

    if "save" in query_lower or "savings" in query_lower:
        # Savings tends to be last mentioned amount
        if len(numbers) >= 3:
            savings = numbers[2]
        elif len(numbers) >= 2:
            savings = numbers[-1]

    # Detect future goals → treat as expense
    future_patterns = [
        "buy", "purchase", "invest in", "planning to buy",
        "want to buy", "saving for", "goal", "target"
    ]

    if any(word in query_lower for word in future_patterns):
        future_expense = numbers[-1]  # last amount is future goal
        if not expenses:
            expenses = future_expense

    # Auto-fill missing values
    if income and expenses and not savings:
        savings = income - expenses

    # If only 2 values are present, assume income > expenses
    if not savings and len(numbers) == 2:
        income, expenses = max(numbers), min(numbers)
        savings = income - expenses

    return {
        "currency": currency,
        "income": income,
        "expenses": expenses,
        "savings": savings,
        "future_expense": future_expense,
        "all_numbers": numbers
    }


# ---------------------------
# 3. Main Tool
# ---------------------------

@tool
def analyze_budget(query: str) -> str:
    """
    Super-flexible financial budgeting analysis tool.
    Extracts income, expenses, savings, currency, trends & future goals.
    """
    try:
        data = extract_budget_brain(query)

        income = data["income"]
        expenses = data["expenses"]
        savings = data["savings"]
        future_expense = data["future_expense"]
        currency = data["currency"]

        # If income still missing → ask user
        if not income:
            return "I couldn't detect your income. Please mention it (example: 'My income is $50000')."

        if not expenses:
            return "I detected your income but not expenses. Please specify the expenses."

        surplus = income - expenses
        savings_rate = (savings / income * 100) if savings else 0

        prompt = f"""
You are BrainBudget — a world-class financial analysis AI.

Extracted Budget:
- Income: {currency}{income}
- Expenses: {currency}{expenses}
- Savings: {currency}{savings}
- Surplus/Deficit: {currency}{surplus}
- Savings Rate: {savings_rate:.2f}%
- Future Goal (if any): {currency}{future_expense}

User Query:
{query}

Now produce a detailed financial analysis with:
1. Clear 3-point summary
2. Spending pattern analysis
3. Trend-based reasoning (e.g., surplus stability, investment readiness)
4. Ideal monthly savings plan (include exact currency values)
5. A goal-achievement roadmap (for future investments like car, home, etc.)
6. Risk analysis + advice on improving savings
"""

        response = llm.invoke(prompt)
        return response.content if hasattr(response, "content") else str(response)

    except Exception as e:
        logger.error(f"Budget tool error: {e}")
        return f"❌ Error in analyze_budget: {e}"




