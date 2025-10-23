from flask import Flask, render_template, request
from model import predict_stock
import os

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    symbol = request.form['symbol']
    prediction, plot_path = predict_stock(symbol)
    return render_template('index.html', prediction=prediction, plot_path=plot_path, symbol=symbol)

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