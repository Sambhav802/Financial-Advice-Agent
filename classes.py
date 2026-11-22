import operator
from enum import Enum, auto
from typing import Annotated, Literal, TypedDict

import pandas as pd
from langchain_core.messages import AnyMessage
from pydantic import BaseModel, Field

from consts import top_crypto_dict


class TimeFrame(Enum):
    DAILY = auto()
    WEEKLY = auto()
    MONTHLY = auto()


Ticker = Enum("Ticker", {symbol: symbol for symbol in top_crypto_dict})
TickerSymbols = Literal[tuple(top_crypto_dict)]


class TickerQuery(BaseModel):
    """Ticker symbol requested by the user"""

    ticker: TickerSymbols = Field(
        description="Ticker symbol for the chosen cryptocurrency. Choose NoCoin if no crypto coin name does not explicitly exist in query",
    )


class FinalReport(BaseModel):
    """Report created by the financial reporter"""

    action: Literal["BUY", "HODL", "SELL"] = Field(
        description="Action to take with the chosen cryptocurrency"
    )
    score: int = Field(
        description="Bullishness market score between 0 (extremely bearish) and 100 (extremely bullish)"
    )
    trend: Literal["UP", "NEUTRAL", "DOWN"] = Field(
        description="Price trend for the chosen cryptocurrency",
    )
    sentiment: Literal["GREED", "NEUTRAL", "FEAR"] = Field(
        description="Sentiment from the news for the chosen cryptocurrency"
    )
    price_predictions: list[float] = Field(
        description="Price predictions for 1, 2, 3 and 4 weeks ahead"
    )
    summary: str = Field(
        description="Summary of the current market conditions (1-3 sentences)"
    )


class AppState(TypedDict):
    user_query: str
    ticker: Ticker
    news: pd.DataFrame
    prices: pd.DataFrame
    price_analyst_report: str
    news_analyst_report: str
    final_report: FinalReport
    final_response: Annotated[list[AnyMessage], operator.add]
