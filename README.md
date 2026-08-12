# LSTM-Stock-Price-Prediction
A PyTorch LSTM model that predicts Apple stock prices using 30-day historical sequences. Utilises time-series forecasting, recurrent neural networks, and financial data preprocessing.

## Key Concepts

### Sequences

We make predictions on price using a sequential pattern: we use 30 day sequences to predict day 31. This creates overlapping windows ([days 1-30→31], [days 2-31→32], etc.) to generate multiple training examples from the continuous dataset.

### Data Scaling

Stock prices ($50-$250) are normalised to (-2, 2) using StandardScaler. This is because large values cause unstable gradients. Our predictions are rescaled back to actual prices for interpretation.

### Train vs Test Split

First 80% of data trains the model, last 20% tests it. We ensure we use a time ordered split so we never use future data to predict the past.

### LSTM Architecture

We apply a stacked LSTM to learn different levels of pattern abstraction. Each of the layers has 32 hidden units balancing the trade-off between capacity and overfitting risk.

### Training Process
- Conduct forward pass to feed sequences and get predictions
- Compute MSE loss (mean squared error)
- Carry out backward pass to calculate gradients
- Update weights using Adam optimiser
- Repeat the process over 200 epochs

### Results

Train RMSE: $4.71 | Test RMSE: $10.44

On training data, predictions are on average within about $4.71 of the true stock price.
On test data, predictions are on average within about $10.44 of the true stock price.

The gap shows the issue of overfitting as the model memorised patterns in the training data but struggles to capture trends on unseen data. However, the model does perform fairly/moderately well. One could potentially explore the lowering of the learning rate and if this would result in an improvement.


## How to Use

### Prerequisites
- Python 3.10+
- Stock data downloads from `yfinance`

### Installation

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Run

```bash
jupyter notebook Stock_price_prediction_project.ipynb
```

Change the `ticker` variable to predict different stocks (e.g., 'GLD', 'TSLA').

## Technologies

- **PyTorch** — Construction of neural network framework
- **LSTM** — Recurrent layer for sequences
- **yfinance** — Stock data
- **scikit-learn** — Data/numerical scaling
- **Matplotlib** — Visualisation using plots


## Results

Model captures price trends reasonably well but does not respond quickly during volatile periods such as sudden market crashes. In fact, the model has a lag affect when it comes to these sudden market events. Higher test error than train error indicates overfitting. 

## Potential Improvements

Add OHLCV data (Open, High, Low, Close, Volume) whilst lowering the learning rate to reduce overfitting. One could potentially experiment with dropout layers and increasing the sequence length (seq_length=60 or 90). The model could then be tested on other stocks and in different timeframes. 


## Project Files

- `Stock_price_prediction_project.ipynb` — Notebook with all code
- `requirements.txt` — Dependencies

