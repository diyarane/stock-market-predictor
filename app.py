from flask import Flask, render_template, request
from model import predict_stock_price, get_buy_suggestion, save_plot, calculate_accuracy
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
            # Get prediction
            prediction, last_close, data_recent = predict_stock_price(symbol)
            print(f"✅ Prediction successful: {prediction}")
            
            # Get buy/sell suggestion
            suggestion, reasoning = get_buy_suggestion(last_close, prediction)
            print(f"✅ Suggestion successful: {suggestion}")
            
            # Calculate model accuracy
            print(f"🔍 Calculating accuracy for {symbol}...")
            accuracy = calculate_accuracy(symbol, days_back=30)
            print(f"✅ Accuracy calculated: {accuracy}%")

            # Save plot for frontend
            plot_path = f"static/{symbol}_plot.png"
            save_plot(symbol, prediction, data_recent, plot_path)
            print(f"✅ Plot saved")

        except Exception as e:
            print(f"❌ Error occurred: {str(e)}")
            import traceback
            traceback.print_exc()
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

'''
cd ~/Desktop/stock-market-predictor

source venv/bin/activate

python -m flask run --host=0.0.0.0 --port=8080
'''