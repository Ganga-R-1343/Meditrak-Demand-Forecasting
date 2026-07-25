# Meditrak-Demand-Forecasting

# 💊 Meditrak: Multi-Store Demand Forecasting & Inventory Optimization

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/streamlit-1.28+-FF4B4B.svg)](https://streamlit.io/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-1.3+-F7931E.svg)](https://scikit-learn.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**Meditrak** is a machine learning solution for retail pharmacy chains to prevent stockouts of critical medications, minimize drug expiration waste, and optimize safety stock levels across global branches.

---

## ✨ Features

* **🔮 Real-Time Demand Prediction:** Projects unit sales per product line across store locations using a Random Forest Regressor.
* **🛡️ Dynamic Safety Stock Buffers:** Automatically calculates safety stock ($20\%$ buffer) to prevent stockouts.
* **📋 Smart Reorder Targets:** Generates purchase order quantities using:
  $$\text{Reorder Target} = \text{Predicted Demand} + \text{Safety Stock}$$
* **📊 Regional Analytics:** Visualizes demand trends across drug categories and store locations.

---

## 🚀 Quick Start Guide

### 1. Clone the Repository

```bash
git clone [https://github.com/your-username/Meditrak-Demand-Forecasting.git](https://github.com/your-username/Meditrak-Demand-Forecasting.git)
cd Meditrak-Demand-Forecasting

📊 Model Performance MetricsMetricScoreMAE (Mean Absolute Error)~2.15 unitsMSE (Mean Squared Error)~7.80 unitsRMSE (Root Mean Squared Error)~2.79 units$R^2$ Score~0.91
