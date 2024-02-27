import requests
import pandas as pd

def get_tsx_tickers():
    url = "https://www.tsx.com/json/company-directory/search/tsx/^*"
    response = requests.get(url)
    data = response.json()
    tickers = [item['symbol'] for item in data['results']]
    return tickers

tsx_tickers = get_tsx_tickers()

# Convert the list to a DataFrame
df = pd.DataFrame(tsx_tickers, columns=['Ticker'])

# Save the DataFrame to a CSV file
df.to_csv('tsx_tickers.csv', index=False)

print("Tickers saved to tsx_tickers.csv")