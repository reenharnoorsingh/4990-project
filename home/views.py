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

    # This is correct if data1 and data2 are DataFrames returned by the fetch_data function
    sma_1 = calculate_sma(data1, 50)
    sma_2 = calculate_sma(data2, 50)
    print(type(data1.index))
    # sma_3 = calculate_sma(data1,200)
    # sma_4 = calculate_sma(data2,200)
    ema_1 = calculate_ema(data1,50) 
    ema_2 = calculate_ema(data2,50)
    rsi_1 = calculate_rsi(data1) 
    rsi_2 = calculate_rsi(data2)
    macd_1, signal_1 = calculate_macd(data1, 12, 26, 9)
    macd_2, signal_2 = calculate_macd(data2, 12, 26, 9)
    bollinger_bands_1 = calculate_bollinger_bands(data1, 20)
    bollinger_bands_2 = calculate_bollinger_bands(data2, 20)
    # lstm_dataset_1 = create_lstm_dataset(stock1, time_step)
    # lstm_dataset_2 = create_lstm_dataset(stock2, time_step)
    # train_lstm_model_1 = create_lstm_model(stock1)
    # train_lstm_model_2 = create_lstm_model(stock2)
    # create_lstm_model_1 = train_lstm_model(stock1)
    # create_lstm_model_2 = train_lstm_model(stock2)
    # lstm_predction_1 = predict_lstm(stock1)
    # lstm_predction_2 = predict_lstm(stock2)
    print(type(data1.index))
    # Create Plotly graphs for the fetched data
    fig1 = go.Figure(data=[go.Scatter(x=data1.index, y=data1['Close'], mode='lines', name=stock1)])
    fig2 = go.Figure(data=[go.Scatter(x=data2.index, y=data2['Close'], mode='lines', name=stock2)])
    fig3 = go.Figure(data=[go.Scatter(x=data1.index, y=sma_1, mode='lines', name='SMA '+stock1)])
    fig4 = go.Figure(data=[go.Scatter(x=data2.index, y=sma_2, mode='lines', name='SMA '+stock2)])
    fig5 = go.Figure(data=[go.Scatter(x=data1.index, y=ema_1, mode='lines', name='EMA '+stock1)])
    fig6 = go.Figure(data=[go.Scatter(x=data2.index, y=ema_2, mode='lines', name='EMA '+stock2)])
    fig7 = go.Figure(data=[go.Scatter(x=data1.index, y=rsi_1, mode='lines', name='RSI '+stock1)])
    fig8 = go.Figure(data=[go.Scatter(x=data2.index, y=rsi_2, mode='lines', name='RSI '+stock2)])
    fig9 = go.Figure(data=[go.Scatter(x=data1.index, y=macd_1, mode='lines', name='MACD ' + stock1)])
    fig10 = go.Figure(data=[go.Scatter(x=data2.index, y=macd_2, mode='lines', name='MACD ' + stock2)])
    fig11 = go.Figure(data=[go.Scatter(x=data1.index, y=bollinger_bands_1, mode='lines', name=stock1)])
    fig12 = go.Figure(data=[go.Scatter(x=data2.index, y=bollinger_bands_2, mode='lines', name=stock2)])
    # fig13 = go.Figure(data=[go.Scatter(x=lstm_dataset_1.index, y=lstm_dataset_1['Close'], mode='lines', name=stock1)])
    # fig14 = go.Figure(data=[go.Scatter(x=lstm_dataset_2.index, y=lstm_dataset_2['Close'], mode='lines', name=stock2)])
    # fig15 = go.Figure(data=[go.Scatter(x= train_lstm_model_1.index, y= train_lstm_model_1['Close'], mode='lines', name=stock1)])
    # fig16 = go.Figure(data=[go.Scatter(x= train_lstm_model_2.index, y= train_lstm_model_2['Close'], mode='lines', name=stock2)])
    # fig17 = go.Figure(data=[go.Scatter(x=create_lstm_model_1.index, y=create_lstm_model_1['Close'], mode='lines', name=stock1)])
    # fig18 = go.Figure(data=[go.Scatter(x=create_lstm_model_2.index, y=create_lstm_model_2['Close'], mode='lines', name=stock2)])
    # fig19 = go.Figure(data=[go.Scatter(x=lstm_predction_1.index, y=lstm_predction_1['Close'], mode='lines', name=stock2)])
    # fig20 = go.Figure(data=[go.Scatter(x=lstm_predction_2.index, y=lstm_predction_2['Close'], mode='lines', name=stock2)])
 
    # Convert the figures to HTML div strings
    div1 = plot(fig1, output_type='div', include_plotlyjs=False)
    div2 = plot(fig2, output_type='div', include_plotlyjs=False)
    div3 = plot(fig3, output_type='div', include_plotlyjs=False)
    div4 = plot(fig4, output_type='div', include_plotlyjs=False)
    div5 = plot(fig5, output_type='div', include_plotlyjs=False)
    div6 = plot(fig6, output_type='div', include_plotlyjs=False)
    div7 = plot(fig7, output_type='div', include_plotlyjs=False)
    div8 = plot(fig8, output_type='div', include_plotlyjs=False)
    div9 = plot(fig9, output_type='div', include_plotlyjs=False)
    div10 = plot(fig10, output_type='div', include_plotlyjs=False)
    div11 = plot(fig11, output_type='div', include_plotlyjs=False)
    div12 = plot(fig12, output_type='div', include_plotlyjs=False)
    # div13 = plot(fig13, output_type='div', include_plotlyjs=False)
    # div14 = plot(fig14, output_type='div', include_plotlyjs=False)
    # div15 = plot(fig15, output_type='div', include_plotlyjs=False)
    # div16 = plot(fig16, output_type='div', include_plotlyjs=False)
    # div17 = plot(fig17, output_type='div', include_plotlyjs=False)
    # div18 = plot(fig18, output_type='div', include_plotlyjs=False)
    # div19 = plot(fig19, output_type='div', include_plotlyjs=False)
    # div20 = plot(fig20, output_type='div', include_plotlyjs=False)
   

    context = {
        
        'stock1': stock1,
        'stock2': stock2,
        
        'graph1': div1,
        'graph2': div2,
        
        'sma1': div3,
        'sma2': div4,

        'ema1': div5,
        'ema2': div6,

        'rsi1': div7,
        'rsi2': div8,

        'macd1': div9,
        'macd2' :div10,

        'bollinger_bands1': div11,
        'bollinger_bands2' :div12,

        #'lstm_dataset1' : div13,
        #'lstm_dataset2' : div14,

        # 'train_lstm_model1' : div15,
        # 'train_lstm_model2' : div16,

        # 'create_lstm_model1' : div17,
        # 'create_lstm_model2' : div18,
        
        # 'lstm_predction1' : div19,
        # 'lstm_predction2' : div20
 
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
