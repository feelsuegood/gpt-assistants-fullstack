import streamlit as st
from typing import Type
from langchain_openai import ChatOpenAI
from langchain.schema import SystemMessage
from langchain.tools import BaseTool
from pydantic import BaseModel, Field
from langchain.agents import initialize_agent, AgentType
from langchain_community.utilities import DuckDuckGoSearchAPIWrapper
import requests

# Get the API key saved in Home.py and use it
openai_api_key = st.session_state.api_keys.get("OPENAI_API_KEY")
if not openai_api_key:
    st.error("Please enter your OpenAI API key in the home page")
    st.stop()

alpha_vantage_api_key = st.session_state.api_keys.get("ALPHA_VANTAGE_API_KEY")
if not alpha_vantage_api_key:
    st.error("Please enter your Alpha Vantage API key in the home page")
    st.stop()

llm = ChatOpenAI(
    temperature=0.1,
    model="gpt-4.1-nano",
)


class StockMarketSymbolSearchToolArgsSchema(BaseModel):
    query: str = Field(description="The query you will search for")


class StockMarketSymbolSearchTool(BaseTool):

    args_schema: Type[StockMarketSymbolSearchToolArgsSchema] = (
        StockMarketSymbolSearchToolArgsSchema
    )

    def _run(self, query):
        ddg = DuckDuckGoSearchAPIWrapper(backend="html")
        result = ddg.run(query)
        return result


class CompanyArgsSchema(BaseModel):

    symbol: str = Field(description="Stock symbol of the company.Example: AAPL, TSLA")


class CompanyOverviewTool(BaseTool):

    args_schema: Type[CompanyArgsSchema] = CompanyArgsSchema

    def _run(self, symbol):
        url = f"https://www.alphavantage.co/query?function=OVERVIEW&symbol={symbol}&apikey={alpha_vantage_api_key}"
        r = requests.get(url)
        return r.json()


class CompanyIncomeStatementTool(BaseTool):

    args_schema: Type[CompanyArgsSchema] = CompanyArgsSchema

    def _run(self, symbol):
        url = f"https://www.alphavantage.co/query?function=INCOME_STATEMENT&symbol={symbol}&apikey={alpha_vantage_api_key}"
        r = requests.get(url)
        return r.json()
        # return r.json()["annualReports"]


class CompanyStockPerformanceTool(BaseTool):

    args_schema: Type[CompanyArgsSchema] = CompanyArgsSchema

    def _run(self, symbol):
        url = f"https://www.alphavantage.co/query?function=TIME_SERIES_WEEKLY&symbol={symbol}&apikey={alpha_vantage_api_key}"
        r = requests.get(url)
        response = r.json()
        return list(response["Weekly Time Series"].items())[:100]


agent = initialize_agent(
    llm=llm,
    verbose=True,
    agent=AgentType.OPENAI_FUNCTIONS,
    handle_parsing_errors=True,
    tools=[
        StockMarketSymbolSearchTool(
            name="StockMarketSymbolSearchTool",
            description="""
            Use this tool to find the stock market symbol of a company.
            It takes a query as an argument.
            Example query: Stock Market Symbol for Apple Company
            """,
        ),
        CompanyOverviewTool(
            name="CompanyOverviewTool",
            description="""
            Must use this first to get an overview of the financials of the company.
            You should enter a stock symbol.
            IMPORTANT: After getting the overview, you MUST use CompanyIncomeStatementTool to analyze detailed financials.
            This tool alone is not sufficient for a complete analysis.
            """,
        ),
        CompanyIncomeStatementTool(
            name="CompanyIncomeStatementTool",
            description="""
            Must use this after CompanyOverviewTool to get detailed income statements.
            This is CRUCIAL for understanding the company's:
            - Revenue growth trends
            - Profit margins
            - Operating expenses
            - Net income trends
            
            You should enter a stock symbol.
            IMPORTANT: After analyzing income statements, you MUST use CompanyStockPerformanceTool to complete the analysis.
            """,
        ),
        CompanyStockPerformanceTool(
            name="CompanyStockPerformanceTool",
            description="""
            Must use this as the final step after CompanyIncomeStatementTool.
            This tool provides crucial stock price performance data including:
            - Recent price trends
            - Weekly performance
            - Trading volumes
            
            This information is ESSENTIAL for making a final investment recommendation.
            After using this tool, provide your final investment analysis and recommendation.
            """,
        ),
    ],
    agent_kwargs={
        "system_message": SystemMessage(
            content="""
            You are a thorough and methodical hedge fund manager who ALWAYS follows a strict analysis process:

            1. First, find the company's stock symbol (Tool name: StockMarketSymbolSearchTool)
            2. Then, get a company overview to understand basic metrics (Tool name: CompanyOverviewTool)
            3. Next, ALWAYS analyze income statements for detailed financial performance (Tool name: CompanyIncomeStatementTool)
            4. Finally, ALWAYS check recent stock performance data (Tool name: CompanyStockPerformanceTool)

            You must complete ALL these steps before making any investment recommendation.
            Your analysis should include:
            - Company overview and market position
            - Financial health from income statements
            - Stock performance trends 
            - Clear buy/sell recommendation with detailed reasoning

            Never skip any analysis steps - all tools must be used for a complete evaluation.

            Be assertive in your judgement and recommend the stock or advise the user against it.
            """
        )
    },
)

st.set_page_config(
    page_title="InvestorGPT",
    page_icon="📈",
)

st.markdown(
    """
# InvestorGPT  

Welcome to InvestorGPT.

Write down the name of a company and our Agent will do the research for you.
"""
)

company = st.text_input("Write the name of the company you are interested in.")

if company:
    result = agent.invoke({"input": company})

    st.write(result["output"].replace("$", "\$"))
