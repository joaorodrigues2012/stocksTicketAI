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
import matplotlib.dates as mdates
import io
import base64

START_DATE = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')
END_DATE = datetime.now().strftime('%Y-%m-%d')

def fetch_stock_info(ticket):
    try:
        stock = yf.Ticker(ticket)
        info = stock.info

        stock_data = {
            "Abertura": info.get("open"),
            "P/L": info.get("trailingPE"),
            "Vol.": info.get("volume"),
            "Alta": info.get("dayHigh"),
            "Máx. de 52 sem.": info.get("fiftyTwoWeekHigh"),
            "Vol. Médio": info.get("averageVolume"),
            "Baixa": info.get("dayLow"),
            "Mín. de 52 sem.": info.get("fiftyTwoWeekLow"),
            "Cap. Merc.": info.get("marketCap"),
            "Fechamento": info.get("previousClose")
        }

        return stock_data
    except Exception as e:
        return f"Error fetching stock info: {e}"

def plot_stock_price(ticket):
    stock = fetch_stock_price(ticket)
    if isinstance(stock, str):  # Verifica se houve um erro
        return stock

    # Calcula as médias móveis de 20 e 50 dias
    stock['SMA20'] = stock['Close'].rolling(window=20).mean()
    stock['SMA50'] = stock['Close'].rolling(window=50).mean()

    fig, ax1 = plt.subplots(figsize=(12, 6))

    # Gráfico do preço de fechamento e médias móveis
    ax1.plot(stock.index, stock['Close'], label="Close Price", color='blue')
    ax1.plot(stock.index, stock['SMA20'], label="20-Day SMA", color='orange')
    ax1.plot(stock.index, stock['SMA50'], label="50-Day SMA", color='green')
    ax1.set_xlabel("Date")
    ax1.set_ylabel("Price")
    ax1.set_title(f"{ticket} Stock Price and Moving Averages")
    ax1.grid(True)
    ax1.legend()

    # Adicionar o volume de negociação
    ax2 = ax1.twinx()
    ax2.bar(stock.index, stock['Volume'], color='gray', alpha=0.3, label='Volume')
    ax2.set_ylabel("Volume")
    ax2.legend(loc='upper left')

    # Formatação das datas no eixo x
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    ax1.xaxis.set_major_locator(mdates.MonthLocator())
    fig.autofmt_xdate()

    # Salvar o gráfico localmente
    image_path = f'{ticket}_stock_plot.png'
    plt.savefig(image_path)
    plt.close(fig)

    stock_info = fetch_stock_info(ticket)

    abertura = stock['Open'].iloc[0]
    fechamento = stock['Close'].iloc[-1]

    # Calcula a valorização em Reais e porcentagem
    stock_info["Valorização (R$)"] = fechamento - abertura
    stock_info["Valorização (%)"] = ((fechamento - abertura) / abertura) * 100

    return image_path, stock_info

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
                    st.image(img_str, caption=f"{ticket} Stock Price", use_column_width=True)
                else:
                    st.error(img_str)

                st.subheader(f"Informações Adicionais de {ticket}")
                st.write(f"Abertura: {stock_info['Abertura']}")
                st.write(f"P/L: {stock_info['P/L']}")
                st.write(f"Vol.: {stock_info['Vol.']}")
                st.write(f"Alta: {stock_info['Alta']}")
                st.write(f"Mín. de 52 sem.: {stock_info['Mín. de 52 sem.']}")
                st.write(f"Máx. de 52 sem.: {stock_info['Máx. de 52 sem.']}")
                st.write(f"Vol. Médio: {stock_info['Vol. Médio']}")
                st.write(f"Baixa: {stock_info['Baixa']}")
                st.write(f"Fechamento: {stock_info['Fechamento']}")
                st.write(f"Valorização (R$): {stock_info['Valorização (R$)']:.2f}")
                st.write(f"Valorização (%): {stock_info['Valorização (%)']:.2f}%")
                st.write(f"Cap. Merc.: {stock_info['Cap. Merc.']}")

            else:
                st.error('No results available for the selected ticket.')
        except Exception as e:
            st.error(f"An error occurred: {e}")



# streamlit run crewai-stocks.py - para rodar aplicação
