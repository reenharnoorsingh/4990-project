import logging
import csv
import os

from django.shortcuts import render
import yfinance as yf
from django.http import JsonResponse, HttpResponse
from get_all_tickers import get_tickers as gt
import logging

from core import settings

logger = logging.getLogger(__name__)
def index(request):
    # Page from the theme
    return render(request, 'pages/dashboard.html')




def get_stock_names(request):
    print("get_stock_names function called.")  # Print statement to verify function call

    # Get the path to the CSV file
    csv_file_path = os.path.join(settings.BASE_DIR, 'tsx_tickers.csv')

    # Read tickers from the CSV file, skipping the header row
    tickers = []
    with open(csv_file_path, 'r') as file:
        reader = csv.reader(file)
        next(reader)  # Skip the header row
        for row in reader:
            tickers.append(row[0])

    # Print tickers to the terminal for verification
    # print("Tickers:", tickers)

    # Return JSON response with tickers
    return JsonResponse({'tickers': tickers})


def stock_chart(request):
    symbol = 'AAPL'  # Example symbol, you can change it as per your requirement
    data = yf.download(symbol, start='2023-01-01', end='2023-12-31')
    dates = data.index.strftime('%Y-%m-%d').tolist()
    prices = data['Close'].tolist()
    return render(request, 'stocks/stock_chart.html', {'dates': dates, 'prices': prices})


def submit_stock_form(request):
    print("submit_stock_form function called.")  # Debug statement to verify function call

    if request.method == 'POST':
        selected_stock1 = request.POST.get('stock1')
        selected_stock2 = request.POST.get('stock2')

        print("Selected Stock 1:", selected_stock1)  # Debug statement to print selected stock 1
        print("Selected Stock 2:", selected_stock2)  # Debug statement to print selected stock 2

        # You can perform further processing here

    return HttpResponse("Stock values printed to console")  # Placeholder response



