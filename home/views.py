from django.shortcuts import render
import yfinance as yf
from django.http import HttpResponseServerError, HttpResponseRedirect
from django.http import HttpResponse
import pandas as pd
import plotly.graph_objs as go
from plotly.offline import plot

from backend2 import (
    fetch_data,
    calculate_sma,
    calculate_ema,
    calculate_rsi,
    calculate_macd,
    calculate_bollinger_bands,
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
    #bollinger_bands_1 = calculate_bollinger_bands(data1, 20)
    # Bollinger Bands calculations
    upper_band_1, lower_band_1 = calculate_bollinger_bands(data1, 20)
    middle_band_1 = calculate_sma(data1, 20)
    #bollinger_bands_2 = calculate_bollinger_bands(data2, 20)
    upper_band_2, lower_band_2 = calculate_bollinger_bands(data2, 20)
    middle_band_2 = calculate_sma(data2, 20)
    

    # Create Plotly graphs for the fetched data
    
    # Create the closing price plots
    fig1 = go.Figure(data=[go.Scatter(x=data1.index, y=data1['Close'], mode='lines', name='Closing Price')])
    fig1.update_layout(title='Closing Price - ' + stock1, xaxis_title='Date', yaxis_title='Price', template='plotly_white')

    fig2 = go.Figure(data=[go.Scatter(x=data2.index, y=data2['Close'], mode='lines', name='Closing Price')])
    fig2.update_layout(title='Closing Price - ' + stock2, xaxis_title='Date', yaxis_title='Price', template='plotly_white')

    # Create the SMA plots
    fig3 = go.Figure()
    fig3.add_trace(go.Scatter(x=data1.index, y=data1['Close'], mode='lines', name='Close Price'))
    fig3.add_trace(go.Scatter(x=data1.index, y=sma_1, mode='lines', name='50-day SMA'))
    fig3.update_layout(title='Simple Moving Average (SMA) - ' + stock1, xaxis_title='Date', yaxis_title='Price', template='plotly_white')

    fig4 = go.Figure()
    fig4.add_trace(go.Scatter(x=data2.index, y=data2['Close'], mode='lines', name='Close Price'))
    fig4.add_trace(go.Scatter(x=data2.index, y=sma_2, mode='lines', name='50-day SMA'))
    fig4.update_layout(title='Simple Moving Average (SMA) - ' + stock2, xaxis_title='Date', yaxis_title='Price', template='plotly_white')

    # Create the EMA plots
    fig5 = go.Figure()
    fig5.add_trace(go.Scatter(x=data1.index, y=data1['Close'], mode='lines', name='Close Price'))
    fig5.add_trace(go.Scatter(x=data1.index, y=ema_1, mode='lines', name='50-day EMA'))
    fig5.update_layout(title='Exponential Moving Average (EMA) - ' + stock1, xaxis_title='Date', yaxis_title='Price', template='plotly_white')

    fig6 = go.Figure()
    fig6.add_trace(go.Scatter(x=data2.index, y=data2['Close'], mode='lines', name='Close Price'))
    fig6.add_trace(go.Scatter(x=data2.index, y=ema_2, mode='lines', name='50-day EMA'))
    fig6.update_layout(title='Exponential Moving Average (EMA) - ' + stock2, xaxis_title='Date', yaxis_title='Price', template='plotly_white')

    # Create the RSI plots
    fig7 = go.Figure()
    fig7.add_trace(go.Scatter(x=data1.index, y=rsi_1, mode='lines', name='RSI'))
    fig7.add_hline(y=70, line_dash="dot", annotation_text="Overbought", annotation_position="bottom right")
    fig7.add_hline(y=30, line_dash="dot", annotation_text="Oversold", annotation_position="bottom right")
    fig7.update_layout(title='Relative Strength Index (RSI) - ' + stock1, xaxis_title='Date', yaxis_title='RSI', template='plotly_white')

    fig8 = go.Figure()
    fig8.add_trace(go.Scatter(x=data2.index, y=rsi_2, mode='lines', name='RSI'))
    fig8.add_hline(y=70, line_dash="dot", annotation_text="Overbought", annotation_position="bottom right")
    fig8.add_hline(y=30, line_dash="dot", annotation_text="Oversold", annotation_position="bottom right")
    fig8.update_layout(title='Relative Strength Index (RSI) - ' + stock2, xaxis_title='Date', yaxis_title='RSI', template='plotly_white')

    # Create the MACD plots
    fig9 = go.Figure()
    fig9.add_trace(go.Scatter(x=data1.index, y=macd_1, mode='lines', name='MACD'))
    fig9.add_trace(go.Scatter(x=data1.index, y=signal_1, mode='lines', name='Signal'))
    fig9.update_layout(title='Moving Average Convergence Divergence (MACD) - ' + stock1, xaxis_title='Date', yaxis_title='MACD', template='plotly_white')

    fig10 = go.Figure()
    fig10.add_trace(go.Scatter(x=data2.index, y=macd_2, mode='lines', name='MACD'))
    fig10.add_trace(go.Scatter(x=data2.index, y=signal_2, mode='lines', name='Signal'))
    fig10.update_layout(title='Moving Average Convergence Divergence (MACD) - ' + stock2, xaxis_title='Date', yaxis_title='MACD', template='plotly_white')

    # Create the Bollinger Bands plots
    fig11 = go.Figure()
    fig11.add_trace(go.Scatter(x=data1.index, y=upper_band_1, mode='lines', name='Upper Band'))
    fig11.add_trace(go.Scatter(x=data1.index, y=middle_band_1, mode='lines', name='Middle Band'))
    fig11.add_trace(go.Scatter(x=data1.index, y=lower_band_1, mode='lines', name='Lower Band'))
    fig11.update_layout(title='Bollinger Bands - ' + stock1, xaxis_title='Date', yaxis_title='Price', template='plotly_white')

    fig12 = go.Figure()
    fig12.add_trace(go.Scatter(x=data2.index, y=upper_band_2, mode='lines', name='Upper Band'))
    fig12.add_trace(go.Scatter(x=data2.index, y=middle_band_2, mode='lines', name='Middle Band'))
    fig12.add_trace(go.Scatter(x=data2.index, y=lower_band_2, mode='lines', name='Lower Band'))
    fig12.update_layout(title='Bollinger Bands - ' + stock2, xaxis_title='Date', yaxis_title='Price', template='plotly_white')
    
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
