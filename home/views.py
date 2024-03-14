from django.shortcuts import render
import yfinance as yf
from django.http import JsonResponse


def index(request):
    # Page from the theme
    return render(request, 'pages/dashboard.html')



def get_stock_names(request):
    # List of tickers for demonstration
    tickers = ['AAPL', 'MSFT', 'GOOGL']  # You can replace this with your desired list of tickers

    # Instantiate the Tickers class with the list of tickers
    all_tickers = yf.Tickers(tickers)

    # Initialize list to store stock names
    stock_names = []

    # Fetch stock names for each ticker
    for ticker in all_tickers.tickers:
        ticker_obj = yf.Ticker(ticker)
        if 'shortName' in ticker_obj.info:
            stock_names.append(ticker_obj.info['shortName'])

    return JsonResponse({'stock_names': stock_names})