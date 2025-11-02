from flask import Flask, render_template, request
from model import predict_stock_price, get_buy_suggestion, save_plot
import os

app = Flask(__name__)

# Ensure static folder exists for plots
if not os.path.exists("static"):
    os.makedirs("static")

@app.route('/', methods=['GET', 'POST'])
def home():
    prediction = None
    symbol = None
    suggestion = None
    reasoning = None
    plot_path = None
    accuracy = None

    if request.method == 'POST':
        symbol = request.form['symbol'].upper().strip()
        try:
            # Get prediction WITH accuracy from LSTM training
            prediction, last_close, data_recent, accuracy = predict_stock_price(symbol)
            
            # Get buy/sell suggestion
            suggestion, reasoning = get_buy_suggestion(last_close, prediction)

            # Save plot for frontend
            plot_path = f"static/{symbol}_plot.png"
            save_plot(symbol, prediction, data_recent, plot_path)

        except Exception as e:
            prediction = f"Error: {str(e)}"
            suggestion = ""
            reasoning = ""
            plot_path = None
            accuracy = None

    return render_template(
        'index.html',
        prediction=prediction,
        symbol=symbol,
        suggestion=suggestion,
        reasoning=reasoning,
        plot_path=plot_path,
        accuracy=accuracy
    )

if __name__ == '__main__':
    app.run(debug=True)