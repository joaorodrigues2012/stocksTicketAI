import json
import os
from datetime import datetime, timedelta

import yfinance as yf

from crewai import Agent, Task, Crew, Process

from langchain.tools import Tool
from langchain_openai import ChatOpenAI
from langchain_community.tools import DuckDuckGoSearchResults

import streamlit as st

import matplotlib.pyplot as plt
import io
import base64

START_DATE = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')
END_DATE = datetime.now().strftime('%Y-%m-%d')

def plot_stock_price(ticket):
    stock = fetch_stock_price(ticket)
    if isinstance(stock, str):  # Erro ao buscar dados
        return stock

    fig, ax = plt.subplots(figsize=(10, 5))
    stock['Close'].plot(ax=ax, title=f"{ticket} Stock Price")
    ax.set_xlabel("Date")
    ax.set_ylabel("Price")
    ax.grid(True)

    # Salvar o gráfico em um buffer de memória e codificá-lo em base64
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    img_str = base64.b64encode(buf.getvalue()).decode('utf-8')
    buf.close()
    plt.close(fig)

    return img_str

def fetch_stock_price(ticket):
    try:
        stock = yf.download(ticket, start=START_DATE, end=END_DATE)
        if stock.empty:
            raise ValueError(f"No data found for ticker: {ticket}")
        return stock
    except Exception as e:
        return f"Error fetching stock price: {e}"

yahoo_finance_tool = Tool(
    name="Yahoo Finance Tool",
    description="Fetches stocks prices for {ticket} from the last year about a specific company from Yahoo Finance API",
    func=fetch_stock_price
)

os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
llm = ChatOpenAI(model="gpt-3.5-turbo")

stockPriceAnalyst = Agent(
    role="Senior stock price analyst",
    goal="Find the {ticket} stock price and analyze trends",
    backstory="""You're highly experienced in analyzing stock prices and making predictions about future prices""",
    llm=llm,
    max_iter=5,
    memory=True,
    tools=[yahoo_finance_tool],
    verbose=True,
    allow_delegation=False
)

getStockPrice = Task(
    description="Analyze the stock {ticket} price history and create a trend analysis",
    expected_output="""Specify the current trend stock price - up, down or sideways.
        eg. stock='AAPL, price UP'""",
    agent=stockPriceAnalyst
)

search_tools = DuckDuckGoSearchResults(backend="news", num_results=10)

newsAnalyst = Agent(
    role="Stock News Analyst",
    goal="""Create a summary of market news related to the stock {ticket}. Specify the current trend with news context and provide a fear/greed score.""",
    backstory="""You're highly experienced in analyzing market trends and news with over 10 years of experience. You consider news sources critically.""",
    llm=llm,
    max_iter=10,
    memory=True,
    tools=[search_tools],
    verbose=True,
    allow_delegation=False
)

get_news = Task(
    description=f"""Search news for the stock and include BTC. Compose results into a report. Current date: {datetime.now()}""",
    expected_output="""A summary of the overall market and one-sentence summaries for each asset with fear/greed scores.""",
    agent=newsAnalyst
)

stockAnalystWrite = Agent(
    role="Senior Stock Analyst Writer",
    goal="""Analyze the trends, price, and news to write a compelling 3-paragraph newsletter based on the stock report and price trend.""",
    backstory="""You're known for creating compelling narratives that resonate with audiences, combining macro factors and multiple theories.""",
    verbose=True,
    llm=llm,
    max_iter=5,
    memory=True,
    allow_delegation=True
)

writeAnalyses = Task(
    description="""Use the stock price trend and news report to create an analysis and write a newsletter about the {ticket} company. Highlight important points and future considerations. Include a recommendation to buy, sell, or hold the stock based on the analysis.""",
    expected_output="""A 3-paragraph newsletter formatted in markdown with:
    - 3 bullet points executive summary
    - Introduction
    - Main part with news summary, fear/greed scores, and recommendation
    - Summary and trend prediction""",
    agent=stockAnalystWrite,
    context=[getStockPrice, get_news]
)

crew = Crew(
    agents=[stockPriceAnalyst, newsAnalyst, stockAnalystWrite],
    tasks=[getStockPrice, get_news, writeAnalyses],
    verbose=True,
    process=Process.hierarchical,
    full_output=True,
    share_crew=False,
    manager_llm=llm,
    max_iter=15
)

# results = crew.kickoff(inputs={'ticket': 'AAPL'})

with st.sidebar:
    st.header('Enter the Stock to Research')

    with st.form(key='research_form'):
        ticket = st.text_input('Select the ticket')
        submit_button = st.form_submit_button(label='Run Research')

if submit_button:
    if not ticket:
        st.error('Please fill the ticket field')
    else:
        try:
            results = crew.kickoff(inputs={'ticket': ticket})
            if 'final_output' in results and results['final_output']:
                st.subheader('Results of your research:')
                st.write(results['final_output'])

                # Mostrar o gráfico
                img_str = plot_stock_price(ticket)
                if not img_str.startswith("Error"):
                    st.image(f"data:image/png;base64,{img_str}", caption=f"{ticket} Stock Price")
                else:
                    st.error(img_str)

                # Exemplo de recomendação (isso deve ser baseado na análise)
                recommendation = "BUY"  # Ou "SELL" ou "HOLD", baseado na análise do texto
                st.write(f"Recommendation: {recommendation}")

            else:
                st.error('No results available for the selected ticket.')
        except Exception as e:
            st.error(f"An error occurred: {e}")



# streamlit run crewai-stocks.py - para rodar aplicação
