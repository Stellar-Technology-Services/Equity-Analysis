#importing required libraries

#ui libraries
import streamlit as st
import pandas as pd

#env libraries for api key
import os
from dotenv import load_dotenv

#libraries to get genuine data about stock options
import yfinance as yf
from yahooquery import Ticker
from datetime import datetime, timedelta

#handle pdf file
import PyPDF2

#langchain modules
from langchain.llms import OpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA

#loading the api key
openai_api_key=os.getenv("OPENAI_API_KEY")
load_dotenv()

def format_large_number(num):
    if abs(num) >= 1_000_000_000_000:
        return f"${num / 1_000_000_000_000:.2f}T"
    elif abs(num) >= 1_000_000_000:
        return f"${num / 1_000_000_000:.2f}B"
    elif abs(num) >= 1_000_000:
        return f"${num / 1_000_000:.2f}M"
    else:
        return str(num)

#generating recommendations
def recommendations(company, query):
    text = ''

    llm=OpenAI(temperature=0.5, openai_api_key=openai_api_key)

    pdfreader = PyPDF2.PdfReader(company["name10k"]+'.pdf')

    for i, page in enumerate(pdfreader.pages):
        content = page.extract_text()
        if content:
            text += content

    text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 1000,
    chunk_overlap  = 200,
    length_function = len,
    )

    texts=text_splitter.split_text(text)
    embeddings=OpenAIEmbeddings(openai_api_key=openai_api_key)
    doc=FAISS.from_texts(texts, embeddings)

    qa=RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=doc.as_retriever())
    response=qa.run(query)

    return response.translate(str.maketrans("", "", "_*"))

#stock options
stocks = {
    "Apple - 'AAPL'": {"name": "APPLE INC", "symbol": "AAPL", "cik": "0000320193", "name10k":"apple10k"},
    "Alphabet - 'GOOG'": {"name": "Alphabet Inc.", "symbol": "GOOG", "cik": "0001652044", "name10k":"alphabet10k"},
    "Facebook - 'META'": {"name": "META PLATFORMS INC", "symbol": "META", "cik": "0001326801", "name10k":"meta10k"},
    "Amazon - 'AMZN'": {"name": "AMAZON COM INC", "symbol": "AMZN", "cik": "0001018724", "name10k":"amazon10k"},
    "Netflix - 'NFLX'": {"name": "NETFLIX INC", "symbol": "NFLX", "cik": "0001065280", "name10k":"netflix10k"},
    "Microsoft - 'MSFT'": {"name": "MICROSOFT CORP", "symbol": "MSFT", "cik": "0000789019", "microsoft10k":"apple10k"},
    "Tesla - 'TSLA'": {"name": "TESLA INC", "symbol": "TSLA", "cik": "0001318605", "name10k":"tesla10k"},
}

#page config
st.set_page_config(page_title="Equity Analysis report", layout="wide")


#title and logo
with st.container():
    left_head, _, right_head=st.columns([1, 0.2, 2])
    
    with left_head:
        st.image("stellar_logo.jpeg", width=200)

    with right_head:
        st.title("Equity Analysis report")

with st.container():
    left_body, _, right_body=st.columns([1, 0.2, 2])

    with left_body:
        #company options drop down
        global company
        company=st.selectbox("", options=list(stocks.keys()), index=0)

        ###Code take from github repo- https://github.com/mrspiggot/FinGPT/blob/master/main.py regarding stock recommendation, prices etc starts###
        # Get stock data from yfinance
        ticker = yf.Ticker(stocks[company]["symbol"])

        # Calculate the date range for the last 30 days
        end_date = datetime.now()
        start_date = end_date - timedelta(days=360)

        # Get the closing prices for the selected stock in the last 30 days
        data = ticker.history(start=start_date, end=end_date)
        closing_prices = data["Close"]

        # Plot the line chart in the first column
        st.line_chart(closing_prices, use_container_width=True)

        # Get the company long description
        long_description = ticker.info["longBusinessSummary"]

        # Use yahooquery to get earnings and revenue
        ticker_yq = Ticker(stocks[company]["symbol"])
        earnings = ticker_yq.earnings

        financials_data = earnings[stocks[company]["symbol"]]['financialsChart']['yearly']

        df_financials = pd.DataFrame(financials_data)
        df_financials['revenue'] = df_financials['revenue']
        df_financials['earnings'] = df_financials['earnings']
        df_financials = df_financials.rename(columns={'earnings': 'yearly earnings', 'revenue': 'yearly revenue'})

        numeric_cols = ['yearly earnings', 'yearly revenue']
        df_financials[numeric_cols] = df_financials[numeric_cols].applymap(format_large_number)
        df_financials['date'] = df_financials['date'].astype(str)
        df_financials.set_index('date', inplace=True)

        # Display earnings and revenue in the first column
        st.write(df_financials)

        summary_detail = ticker_yq.summary_detail[stocks[company]["symbol"]]

        obj = yf.Ticker(stocks[company]["symbol"])

        pe_ratio = '{0:.2f}'.format(summary_detail["trailingPE"])
        price_to_sales = summary_detail["fiftyTwoWeekLow"]
        target_price = summary_detail["fiftyTwoWeekHigh"]
        market_cap = summary_detail["marketCap"]
        ebitda = ticker.info["ebitda"]
        tar = ticker.info["targetHighPrice"]
        rec = ticker.info["recommendationKey"].upper()

        # Format large numbers
        market_cap = format_large_number(market_cap)
        ebitda = format_large_number(ebitda)

        # Create a dictionary for additional stock data
        additional_data = {
            "P/E Ratio": pe_ratio,
            "52 Week Low": price_to_sales,
            "52 Week High": target_price,
            "Market Capitalisation": market_cap,
            "EBITDA": ebitda,
            "Price Target": tar,
            "Recommendation": rec
        }

        # Display additional stock data in the first column
        for key, value in additional_data.items():
            st.write(f"{key}: {value}")
        ###Code take from Github repo ends###

    with right_body:
        st.header("Overview")
        ticker = yf.Ticker(stocks[company]["symbol"])
        st.write(ticker.info["longBusinessSummary"])
        st.header("Opportunities for potential investors")
        st.write(recommendations(stocks[company], "What are this firm's key products and services?"))
        st.write(recommendations(stocks[company], "What are the new products and services that the company has to offer?"))
        st.write(recommendations(stocks[company], "What are the company's unique strengths?"))
        st.write(recommendations(stocks[company], "Who are this firms key competitors? What are the principal threats?"))
