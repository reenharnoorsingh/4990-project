from django.shortcuts import render
import yfinance as yf
from django.http import HttpResponseServerError, HttpResponseRedirect
from django.http import HttpResponse
import pandas as pd


# Create your views here.

def index(request):
    if request.method == 'POST' and 'get_started' in request.POST:
        request.session['get_started_clicked'] = True
        return HttpResponseRedirect(request.path)  # Redirect to the same page to refresh content

    return render(request, 'pages/dashboard.html')


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
