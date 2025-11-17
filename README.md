This README is *really* solid already – it genuinely looks like a mini-paper.
Only a few small fixes / clean-ups:

1. **Remove the outer code fence** (` ````markdown` … ``` `) – in the actual README you want plain markdown starting with `# ts-forecasting-classical-vs-deep`.
2. In **Project structure**, close the block with **three** backticks, not four.
3. In **Forecast plot**, you’re still putting the `<img>` tag *inside* a code block, so GitHub will show it as text, not as an image. Put it outside the ``` block (or better, use the local file).
4. Double-check your username in the `git clone` command (`imaneelhy` vs `imanelhy`). It must exactly match your GitHub account.

Here’s your README with those fixes applied and using the image from your repo instead of the long attachment URL (cleaner and safer):

````markdown
# ts-forecasting-classical-vs-deep

Comparing **classical time-series models** and a **neural network (LSTM)** for
one-step-ahead forecasting of daily minimum temperatures in Melbourne.

We evaluate:

- Naïve baseline (last observed value)
- Moving average baseline
- ARIMA(5,1,0)
- LSTM neural network

and compare their performance using MSE, MAE, and MAPE on a held-out test set.

---

## Abstract

Deep learning models are increasingly used for time-series forecasting, even in
simple univariate settings where classical statistical models have traditionally
performed well. In this project, we compare a small LSTM neural network to
classical forecasting methods on the daily minimum temperature series in
Melbourne. We evaluate four models: a naïve “last value” baseline, a moving
average baseline, an ARIMA(5,1,0) model, and an LSTM trained on sliding windows
of the normalized series. Performance is measured using MSE, MAE, and MAPE on a
chronologically held-out test period. Our experiments show that ARIMA slightly
outperforms the LSTM and both only modestly improve over simple baselines,
suggesting that for short univariate series with clear seasonality, classical
models remain highly competitive and deep learning is not strictly necessary.

---

## Project structure

```text
ts-forecasting-classical-vs-deep/
│
├─ src/
│   ├─ __init__.py
│   ├─ prepare_data.py       # dataset loading, scaling, train/val/test split
│   ├─ models_classical.py   # naïve, moving average, ARIMA(5,1,0) with rolling forecasts
│   └─ models_neural.py      # LSTM model and training loop
│
├─ notebooks/
│   └─ eda_and_visualization.ipynb   # Colab/Jupyter notebook using the src/ code
│
├─ results/
│   ├─ example_forecasts.png         # test forecasts vs ground truth
│   └─ metrics.json                  # MSE / MAE / MAPE per model (JSON)
│
├─ train_and_evaluate.py   # runs full experiment from the command line
├─ requirements.txt
└─ README.md
````

---

## Dataset

* **Daily minimum temperatures in Melbourne, Australia**
* 3650 daily observations (1981–1990)
* Source: public dataset `daily-min-temperatures.csv`

We perform a **chronological split**:

* 70% train
* 15% validation
* 15% test

Target task: **one-step-ahead forecasting** – predict tomorrow’s temperature
from the previous `window_size` days (here: 14 days).

To avoid data leakage, the MinMax scaler is fit on the **training split only**
and then applied to validation and test data.

---

## Models

### Baselines

* **Naïve:**
  Forecast = last observed value.

* **Moving average:**
  Forecast = mean of the last `window_size` observations (14 days).

Both baselines operate directly on the normalized test segment.

### Classical time-series model

* **ARIMA(5,1,0)**

  * Fitted on the concatenated train + validation series (normalized).
  * Used in a **rolling one-step-ahead** fashion on the test set: for each
    test time step, we fit on all data up to that point, forecast one step,
    then update the history with the true test observation.

### Neural network (deep learning)

* **LSTM forecaster**

  * Input: sliding window of the last 14 normalized values.
  * Architecture: 1 LSTM layer (hidden size 32, 1 layer) followed by a
    linear output layer.
  * Loss: mean squared error (MSE).
  * Optimizer: Adam with learning rate 1e-3.
  * Training: 20 epochs on the training set; validation MSE monitored each epoch.

All model predictions are inverse-transformed back to degrees Celsius before
computing metrics.

---

## Quantitative results (test set)

The table below shows performance on the held-out test period
(temperature in °C):

| Model          | MSE  | MAE  | MAPE (%) |
| -------------- | ---- | ---- | -------- |
| Naive          | 6.69 | 2.05 | 22.09    |
| Moving average | 6.35 | 1.96 | 21.00    |
| ARIMA (5,1,0)  | 5.34 | 1.80 | 19.49    |
| LSTM           | 5.41 | 1.81 | 20.29    |

Results are also saved as a JSON file in `results/metrics.json`.

### Results & discussion

On the held-out test period, all models achieve reasonably low errors, but
classical approaches perform surprisingly well. The naïve baseline (forecast =
last observed value) already yields an MSE of 6.69 and a MAPE of 22.1%, while
a simple moving-average baseline reduces the error slightly. The ARIMA(5,1,0)
model achieves the best overall performance (MSE 5.34, MAE 1.80, MAPE 19.5%),
closely followed by the LSTM (MSE 5.41, MAE 1.81, MAPE 20.3%). The differences
between ARIMA and the LSTM are small, and both models only modestly improve
over the moving-average baseline. Visually, all methods track the seasonal
pattern of the temperature series, with larger deviations mainly around sharp
spikes. These results suggest that for short univariate series with clear
seasonality, classical time-series models can match or slightly outperform
neural networks, and deep learning is not strictly necessary to obtain
competitive forecasts.

---

## Forecast plot

An example comparison of predictions on the test set:

![Forecast comparison](results/example_forecasts.png)

---

## How to run

### 1. From the command line

```bash
# clone the repo
git clone https://github.com/imaneelhy/ts-forecasting-classical-vs-deep.git
cd ts-forecasting-classical-vs-deep

# install dependencies
pip install -r requirements.txt

# run full experiment
python train_and_evaluate.py
```

This will:

1. Download and prepare the dataset.
2. Split and normalize the series.
3. Train the LSTM model.
4. Compute forecasts for all models (naive, moving average, ARIMA, LSTM).
5. Save metrics to `results/metrics.json`.
6. Save the forecast plot to `results/example_forecasts.png`.

### 2. Using the notebook (EDA & visualization)

Open `notebooks/eda_and_visualization.ipynb` in Jupyter or Google Colab to:

* Explore the time series (trend, seasonality, autocorrelation).
* Visualize train/val/test splits.
* Run the models interactively and inspect predictions.

The notebook uses the same `src/` modules as the script.

---

## Conclusion

This comparison between classical and neural models on a short univariate
temperature series shows that deep learning does not automatically dominate
traditional approaches. A simple ARIMA(5,1,0) model achieved the best overall
accuracy on the test set, with the LSTM performing very similarly but not
clearly better. Both models only moderately improved over a moving-average
baseline, while a naïve “last value” forecast was already competitive. These
results highlight that in low-dimensional, small-data settings, careful
classical modelling can be just as effective as more complex neural networks
and may be preferable due to its simplicity, interpretability, and lower
computational cost.

---

## Future work

Possible extensions of this project include:

* Evaluating additional classical models such as seasonal ARIMA or Prophet.
* Comparing alternative neural architectures (GRU, 1D CNNs, Seq2Seq models).
* Studying multi-step forecasting (e.g., 7-day or 30-day horizons) rather than
  only one-step-ahead predictions.
* Repeating the comparison on multiple datasets (energy demand, traffic volume,
  financial time series) to see whether the conclusions generalize.
* Incorporating exogenous variables (e.g., weather covariates) and assessing
  whether neural networks provide larger gains in multivariate settings.

---

## License

This project is released under the [MIT License](LICENSE).

```


```
