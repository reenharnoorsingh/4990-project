document.addEventListener('DOMContentLoaded', function() {
    // Fetch stock data and render graphs
    fetchStockDataAndRenderGraphs();
});

function fetchStockDataAndRenderGraphs() {
    // Get the stock symbols from the URL parameters
    const urlParams = new URLSearchParams(window.location.search);
    const stock1 = urlParams.get('stock1');
    const stock2 = urlParams.get('stock2');

    // Fetch stock data from the backend
    fetch(`/stock_display?stock1=${stock1}&stock2=${stock2}`)
        .then(response => {
            if (!response.ok) {
                throw new Error('Network response was not ok');
            }
            return response.json();
        })
        .then(data => {
            // Check if the data is valid JSON
            if (data && typeof data === 'object') {
                // Render graphs using Plotly
                renderStockGraph('chartBig1', data.stock1_dates, data.stock1_open, data.stock1_close, data.stock1);
                renderStockGraph('chartBig2', data.stock2_dates, data.stock2_open, data.stock2_close, data.stock2);
            } else {
                console.error('Invalid data received:', data);
            }
         })
        .catch(error => {
            console.error('Error fetching stock data:', error);
        });
}
