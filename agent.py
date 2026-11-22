"""Shared LangGraph pipeline used by both the Streamlit app and the FastAPI service."""

import os
import warnings

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph
from openbb import obb

from classes import AppState, FinalReport, Ticker, TickerQuery, TimeFrame
from consts import top_crypto_dict
from utils import (
    add_indicators,
    calculate_50_percent,
    get_money_supply,
    get_news_data,
    get_price_data,
)

warnings.filterwarnings("ignore")

BASE_ADVISOR_PROMPT = (
    "You are an expert financial advisor with deep expertise in personal finance, "
    "investments, budgeting, taxation, and financial planning. Your goal is to provide "
    "precise, actionable, and reliable advice tailored to users' specific financial "
    "situations. Ensure your answers are accurate and relevant."
)

GENERAL_ADVISOR_PROMPT = f"""{BASE_ADVISOR_PROMPT}

If you do not know the answer to a question or if the query is unrelated to your expertise, humbly deny it and explain that you cannot provide an answer in that case. When providing advice, clearly communicate any risks, uncertainties, or potential downsides involved to help users make informed decisions. Always strive to answer the user's query in a clear, professional, and trustworthy manner.
"""

REPORT_ADVISOR_PROMPT = (
    BASE_ADVISOR_PROMPT
    + """

Additionally, there are pre-generated reports stored in variables that you should refer to when answering the user's query:
    • News Analyst Report: {news_report}
    • Price Analyst Report: {price_report}
    • Financial Report: {final_report}

Refer to these reports, if available, to ensure your responses are well-informed and data-driven.

If you do not know the answer to a question, if the query is unrelated to your expertise, or if there is insufficient information to provide an informed response, humbly deny it and explain why. When giving advice, always clearly communicate any risks, uncertainties, or potential downsides involved to help users make informed decisions. Strive to deliver clear, professional, and trustworthy responses to every query.
"""
)


def ticker_check(state: AppState) -> str:
    """Returns "yes" when a supported cryptocurrency was identified, else "no"."""
    ticker_name = state["ticker"].name
    return "yes" if ticker_name in top_crypto_dict and ticker_name != "NoCoin" else "no"


def build_app(model: str, override_env: bool = False):
    """Builds and compiles the advisor graph using the given Groq model."""
    load_dotenv(override=override_env)

    llm = ChatGroq(
        temperature=0,
        model_name=model,
        api_key=os.environ.get("GROQ_API_KEY"),
    )
    obb.account.login(pat=os.environ.get("OPENBB_API_KEY"))
    obb.user.preferences.output_type = "dataframe"

    def ticker_extractor(state: AppState):
        """Extracts the cryptocurrency ticker mentioned in the user's query."""
        extractor = llm.with_structured_output(TickerQuery)
        extraction = extractor.invoke([HumanMessage(state["user_query"])])
        return {"ticker": Ticker[extraction.ticker]}

    def news_retriever(state: AppState):
        """Retrieves news for the given ticker."""
        return {"news": get_news_data(state["ticker"])}

    def price_retriever(state: AppState):
        """Retrieves weekly price data with technical indicators for the given ticker."""
        price_df = get_price_data(state["ticker"], time_frame=TimeFrame.WEEKLY)
        price_df = add_indicators(price_df)
        return {"prices": price_df.tail(n=24)}

    def price_analyst(state: AppState):
        """Analyzes price data and generates a prediction report."""
        price_df = state["prices"]
        weeks_4_50_percent, _, _ = calculate_50_percent(price_df, n_weeks=4)
        weeks_12_50_percent, _, _ = calculate_50_percent(price_df, n_weeks=12)
        weeks_26_50_percent, _, _ = calculate_50_percent(price_df, n_weeks=26)

        money_supply_text = str(get_money_supply()["m2"])

        prompt = f"""You have extensive knowledge of the cryptocurrency market and historical data.
Think step-by-step and focus on the technical indicators.
Use the following weekly close price history and technical indicators for the particular currency:

price history:
{str(price_df[price_df.columns[2:]])}
M2 money supply history:
{money_supply_text}

4 weeks 50% level: {weeks_4_50_percent}
12 weeks 50% level: {weeks_12_50_percent}
26 weeks 50% level: {weeks_26_50_percent}

Predict the next 4 weekly prices based on the data. Put the predictions on separate line.
How certain are you of your predictions? Use a number between 0 (not certain at all) to 10 (sure thing).
What is the overall trend outlook? Explain your predictions in 1-3 sentences.

When creating your answer, focus on answering the user query:
{state["user_query"]}
"""
        response = llm.invoke([HumanMessage(prompt)])
        return {"price_analyst_report": response.content}

    def news_analyst(state: AppState):
        """Analyzes news sentiment and generates a sentiment score."""
        news_text = ""
        for date, row in list(state["news"].iterrows())[:20]:
            news_text += f"{date}\n{row.title}\n{row.body}\n---\n"

        prompt = f"""Choose a combined sentiment that best represents these news articles:

```
{news_text}
```

Each article is separated by `---`.

Pick a number for the sentiment between 0 and 100 where:

- 0 is extremely bearish
- 100 is extremely bullish

Reply only with the sentiment and a short explanation (1-2 sentences) of why.

When creating your answer, focus on answering the user query:
{state["user_query"]}
"""
        response = llm.invoke([HumanMessage(prompt)])
        return {"news_analyst_report": response.content}

    def financial_reporter(state: AppState):
        """Generates a final financial report based on price and news analyses."""
        prompt = f"""You're a senior cryptocurrency expert that makes extremely accurate predictions
about future prices and trend in the crypto market. You're well versed into technological advancements
and tokenomics of various projects.

You're working with two other agents that have created reports of the current state of the crypto market.

Report of the news analyst:

```
{state["news_analyst_report"]}
```

Report of the price analyst:

```
{state["price_analyst_report"]}
```

Based on the provided information, create a final report for the user.

When creating your answer, focus on answering the user query:
{state["user_query"]}
"""
        reporter = llm.with_structured_output(FinalReport)
        return {"final_report": reporter.invoke([HumanMessage(prompt)])}

    def final_answer(state: AppState):
        """Answers the user's query, using the generated reports when a ticker was found."""
        if ticker_check(state) == "no":
            prompt = GENERAL_ADVISOR_PROMPT
        else:
            prompt = REPORT_ADVISOR_PROMPT.format(
                news_report=state["news_analyst_report"],
                price_report=state["price_analyst_report"],
                final_report=state["final_report"],
            )

        message = llm.invoke(
            [SystemMessage(content=prompt), HumanMessage(state["user_query"])]
        )
        return {"final_response": [message]}

    graph = StateGraph(AppState)

    graph.add_node("ticker_extractor", ticker_extractor)
    graph.add_node("news_retriever", news_retriever)
    graph.add_node("price_retriever", price_retriever)
    graph.add_node("price_analyst", price_analyst)
    graph.add_node("news_analyst", news_analyst)
    graph.add_node("financial_reporter", financial_reporter)
    graph.add_node("final_answer", final_answer)

    graph.add_conditional_edges(
        "ticker_extractor",
        ticker_check,
        {"yes": "price_retriever", "no": "final_answer"},
    )
    graph.add_edge("price_retriever", "news_retriever")
    graph.add_edge("price_retriever", "price_analyst")
    graph.add_edge("news_retriever", "news_analyst")
    graph.add_edge("news_analyst", "financial_reporter")
    graph.add_edge("financial_reporter", "final_answer")

    graph.set_entry_point("ticker_extractor")
    graph.set_finish_point("final_answer")
    return graph.compile()
