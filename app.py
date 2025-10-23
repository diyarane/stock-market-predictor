from flask import Flask, render_template, request
from model import predict_stock_price, get_buy_suggestion, save_plot

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        symbol = request.form['symbol'].upper()
        predicted_price, last_close, data_recent = predict_stock_price(symbol)
        suggestion, reasoning = get_buy_suggestion(last_close, predicted_price)

        # Save plot for frontend
        plot_path = f"static/{symbol}_plot.png"
        save_plot(symbol, predicted_price, data_recent, plot_path)

        return render_template(
            'index.html',
            prediction=predicted_price,
            symbol=symbol,
            suggestion=suggestion,
            reasoning=reasoning
        )

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)





'''

Before presenting:

Open your terminal → activate your virtual environment:

cd ~/Desktop/stock-market-predictor
source venv/bin/activate
python -m flask run --host=0.0.0.0 --port=8080


Open browser at:

http://localhost:8080/
'''