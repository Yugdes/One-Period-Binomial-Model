# One-Period Binomial Model

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://your-streamlit-app-url.streamlit.app)

This project implements a **One-Period Binomial Model** for pricing financial options. It includes an interactive web application built with Streamlit and a Jupyter Notebook for detailed analysis. The project demonstrates core quantitative finance concepts, such as option pricing, real-time data fetching, and trading signals based on model vs. market price comparisons.

This was developed as a group project for the course ES418 (Financial Engineering/Mathematics).

## Features

- **Interactive Streamlit Web App**: An easy-to-use interface for pricing Call and Put options.
- **Real-Time Market Data**: Integrates with `yfinance` to fetch live spot prices, option chains, and calculate realized volatility.
- **One-Period Binomial Pricing**: Calculates up/down factors, risk-neutral probabilities, replicating portfolios, and the theoretical model price.
- **Trading Signals**: Compares the calculated model price with the live market price to generate Buy / Sell / Hold signals based on a defined threshold.
- **Visualizations**: Interactive Plotly charts mapping the binomial tree.

## Files

- `streamlit_app.py`: The main Streamlit web application.
- `Binomial_model.ipynb`: Jupyter notebook containing model implementation and analysis.
- `ES418_Group_4_Project.pdf`: Final project report detailing the theoretical background and methodology.
- `ES_418_Group_4_Project_Presentation.pdf`: Presentation slides for the project.

## Requirements

Ensure you have Python installed along with the following libraries:

```bash
pip install numpy pandas plotly streamlit yfinance
```

## Usage

To run the interactive Streamlit application:

```bash
streamlit run streamlit_app.py
```

## Contributing

Contributions, issues, and feature requests are welcome!

## License

This project is open-sourced under the MIT License.
