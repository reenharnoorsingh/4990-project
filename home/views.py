from django.shortcuts import render
import yfinance as yf
from django.http import HttpResponseServerError
from django.http import HttpResponse

# Create your views here.

def index(request):

    # Page from the theme 
    return render(request, 'pages/dashboard.html')


import yfinance as yf
from django.shortcuts import render
from django.http import HttpResponseServerError

def stock_dropdown(request):
    if request.method == 'POST':
        # If the form is submitted, retrieve the selected stock symbol
        selected_stock = request.POST.get('stock1')
        try:
            # Fetch available stock options using yfinance
            stock_options = yf.Tickers().tickers

            # Extract the symbols from the stock options
            symbols = [stock_info.info['symbol'] for stock_info in stock_options]

            # Pass the symbols to the template
            return render(request, 'pages/dashboard.html', {'symbols': symbols, 'selected_stock': selected_stock})
        except yf.YFinanceError as yf_error:
            # Handle yfinance specific errors
            error_message = f"YFinanceError: {yf_error}"
            return render(request, 'error.html', {'error_message': error_message})
        except Exception as e:
            # Handle other unexpected errors gracefully
            error_message = f"An error occurred while fetching stock options: {str(e)}"
            return render(request, 'error.html', {'error_message': error_message})
    else:
        # If the request method is not POST, render the form with available stock options
        try:
            # Fetch available stock options using yfinance
            stock_options = yf.Tickers().tickers

            # Extract the symbols from the stock options
            symbols = [stock_info.info['symbol'] for stock_info in stock_options]

            # Pass the symbols to the template
            return render(request, 'pages/dashboard.html', {'symbols': symbols})
        except yf.YFinanceError as yf_error:
            # Handle yfinance specific errors
            error_message = f"YFinanceError: {yf_error}"
            return render(request, 'error.html', {'error_message': error_message})
        except Exception as e:
            # Handle other unexpected errors gracefully
            error_message = f"An error occurred while fetching stock options: {str(e)}"
            return render(request, 'error.html', {'error_message': error_message})
