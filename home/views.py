from django.shortcuts import render
import yfinance as yf
from django.http import HttpResponseServerError, HttpResponseRedirect
from django.http import HttpResponse
import pandas as pd

import plotly.graph_objs as go
from plotly.offline import plot
# from backend import fetch_data

from backend2 import (
    fetch_data,
    calculate_sma,
    calculate_ema,
    calculate_rsi,
    calculate_macd,
    calculate_bollinger_bands,
    create_lstm_dataset,
    create_lstm_model,
    train_lstm_model,
    predict_lstm
)


# Create your views here.

# def index(request):
#     if request.method == 'POST' and 'get_started' in request.POST:
#         request.session['get_started_clicked'] = True
#         return HttpResponseRedirect(request.path)  # Redirect to the same page to refresh content

#     return render(request, 'pages/dashboard.html')


def index(request):
    # Read the CSV file into a DataFrame
    df = pd.read_csv('tsx_tickers.csv')

    tickers_data = df.set_index('Ticker')['Name'].to_dict()

    # Convert DataFrame columns to lists
    tickers = df['Ticker'].tolist()
    names = df['Name'].tolist()

    # Zip tickers and names into a list of tuples
    tickers_with_names = zip(tickers, names)

    # Pass the tickers_with_names to your template context
    context = {
        'tickers': tickers_with_names, 
        # 'tickers2': tickers,
        # 'names': names,
        'tickers_data': tickers_data
        
    }

    # Return only the dropdown menu portion of the page
    return render(request, 'pages/dashboard.html', context)

# def stock_display(request):
    
#     # Fetch data for both stocks
#     stock1 = request.GET.get('stock1')
#     stock2 = request.GET.get('stock2')
#     # Calculate SMA and EMA for both stocks
#     sma1 = calculate_sma(stock1, window=50)
#     ema1 = calculate_ema(stock1, span=50)
#     sma2 = calculate_sma(stock2, window=50)
#     ema2 = calculate_ema(stock2, span=50)
 
#     # Add graph data into context
#     context = {
#         'stock1': stock1,
#         'stock2': stock2,
#         'sma1': sma1,
#         'ema1': ema1,
#         'sma2': sma2,
#         'ema2': ema2
#     }

#     # Return only the dropdown menu portion of the page
#     return render(request, 'pages/stock_display.html', context)



def stock_display(request):
    stock1 = request.GET.get('stock1')
    stock2 = request.GET.get('stock2')
 
    # Use your backend functions to fetch the data for the stocks
    data1 = fetch_data(stock1)
    data2 = fetch_data(stock2)
 
    # Create Plotly graphs for the fetched data
    fig1 = go.Figure(data=[go.Scatter(x=data1.index, y=data1['Close'], mode='lines', name=stock1)])
    fig2 = go.Figure(data=[go.Scatter(x=data2.index, y=data2['Close'], mode='lines', name=stock2)])
 
    # Convert the figures to HTML div strings
    div1 = plot(fig1, output_type='div', include_plotlyjs=False)
    div2 = plot(fig2, output_type='div', include_plotlyjs=False)
 
    context = {
        'graph1': div1,
        'graph2': div2,
        'stock1': stock1,
        'stock2': stock2,
    }
 
    return render(request, 'pages/stock_display.html', context)




    # Return only the dropdown menu portion of the page

def stock_dropdown1(request):
    # Read the CSV file into a DataFrame
    df = pd.read_csv('tsx_tickers.csv')

    # Convert DataFrame columns to lists
    tickers = df['Ticker'].tolist()
    names = df['Name'].tolist()

    # Zip tickers and names into a list of tuples
    tickers_with_names = zip(tickers, names)

    # Pass the tickers_with_names to your template context
    context = {
        'tickers': tickers_with_names
    }

    # Return only the dropdown menu portion of the page
    return render(request, 'includes/stock_dropdown.html', context)


def stock_dropdown2(request):
    # Read the CSV file into a DataFrame
    df = pd.read_csv('tsx_tickers.csv')

    # Convert DataFrame columns to lists
    tickers = df['Ticker'].tolist()
    names = df['Name'].tolist()

    # Zip tickers and names into a list of tuples
    tickers_with_names = zip(tickers, names)

    # Pass the tickers_with_names to your template context
    context = {
        'tickers': tickers_with_names
    }

    # Return only the dropdown menu portion of the page
    return render(request, 'includes/stock_dropdown.html', context)
