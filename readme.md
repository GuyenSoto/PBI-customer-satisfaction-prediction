# Customer Satisfaction Prediction System

A machine learning-based system for predicting customer satisfaction with delivery services before the delivery occurs, using ensemble modeling techniques and Streamlit for visualization. 

## Overview

This project implements a predictive model that helps businesses anticipate customer satisfaction levels before delivery, allowing for proactive measures to improve service quality. The system uses historical customer data along with delivery logistics information to identify potential issues and suggest targeted interventions.

## Features

- **Predictive Analytics**: Uses an ensemble of machine learning models to predict satisfaction with >85% accuracy
- **Interactive Interface**: User-friendly Streamlit application for real-time predictions
- **Proactive Recommendations**: Suggests specific actions based on risk factors
- **Enhanced Logistics Integration**: Incorporates delivery distance and time for improved accuracy
- **Visualization**: Intuitive visual representation of satisfaction probability
- **Power BI Integration**: Includes a Power BI dashboard for business intelligence visualization

## Screenshots

### Basic Prediction Application
![Basic Prediction App Main Screen](https://github.com/GuyenSoto/customer-satisfaction-prediction/raw/main/OUTPUT/prediction_app1_2025-05-02%20101351.jpg)
*Main interface of the basic prediction application*

![Basic Prediction App Results](https://github.com/GuyenSoto/customer-satisfaction-prediction/raw/main/OUTPUT/prediction_app2_2025-05-02%20101351.jpg)
*Prediction results with satisfaction probability visualization*

### Enhanced Prediction Application
![Enhanced Prediction App Interface](https://github.com/GuyenSoto/customer-satisfaction-prediction/raw/main/OUTPUT/prediction_app_enhanced3_2025-05-02%20101351.jpg)
*Enhanced application with logistics variables*

![Enhanced Prediction App Recommendations](https://github.com/GuyenSoto/customer-satisfaction-prediction/raw/main/OUTPUT/prediction_app_enhanced4_2025-05-02%20101351.jpg)
*Risk factors and specific action recommendations*

### Power BI Dashboard
![Happy Customers Dashboard Overview](https://github.com/GuyenSoto/customer-satisfaction-prediction/raw/main/OUTPUT/Happy_Customers1_2025-05-03%20105036.jpg)
*Main dashboard overview with key metrics and visualizations*

![Happy Customers Detailed Analytics](https://github.com/GuyenSoto/customer-satisfaction-prediction/raw/main/OUTPUT/Happy_Customers2_2025-05-03%20105036.jpg)
*Detailed analytics view showing customer satisfaction trends*

![Happy Customers Interactive Reports](https://github.com/GuyenSoto/customer-satisfaction-prediction/raw/main/OUTPUT/Happy_Customers3_2025-05-03%20105036.jpg)
*Interactive reports for business intelligence insights*

## Project Structure

- `ensemble_model.py`: Core implementation of the voting ensemble model
- `train_and_save_model_enhanced.py`: Script to train the enhanced model with logistics features
- `prediction_app_enhanced.py`: Streamlit app for the enhanced model with logistics variables
- `Happy_Customers.pbix`: Power BI dashboard with informative visuals about customer satisfaction

## Model Details

The system uses a voting ensemble that combines multiple algorithms:
- Random Forest
- Gradient Boosting
- LightGBM
- Support Vector Machine
- Logistic Regression

The enhanced version incorporates additional features:
- Delivery distance
- Estimated delivery time
- Feature interactions with logistics variables

## Installation

1. Clone this repository:
```bash
git clone https://github.com/GuyenSoto/customer-satisfaction-prediction.git
cd customer-satisfaction-prediction
```

2. Create a conda environment:
```bash
conda create -n satisfaction_prediction python=3.9
conda activate satisfaction_prediction
```

3. Install the required packages:
```bash
pip install -r requirements.txt
```

## Usage

### Training the Model

To train the enhanced model:
```bash
python train_and_save_model_enhanced.py
```

### Running the Prediction App

For the enhanced app with logistics features:
```bash
streamlit run prediction_app_enhanced.py
```

### Using the Power BI Dashboard

Open the `Happy_Customers.pbix` file with Power BI Desktop to explore interactive visualizations of customer satisfaction data and insights.

## Input Variables

The model uses the following customer satisfaction indicators:

| Variable | Description |
|----------|-------------|
| X1 | My order was delivered on time |
| X2 | Contents of my order was as I expected |
| X3 | I ordered everything I wanted to order |
| X4 | I paid a good price for my order |
| X5 | I am satisfied with my courier |
| X6 | The app makes ordering easy for me |

The enhanced model additionally uses:
- Delivery distance (km)
- Estimated delivery time (minutes)
- Order complexity (1-10)
- Courier rating (1-5)

## Requirements

- Python 3.9+
- pandas
- numpy
- scikit-learn
- LightGBM
- Streamlit
- Matplotlib
- Seaborn
- Power BI Desktop (to view the .pbix file)

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- The ACME-HappinessSurvey2020 dataset provided the foundation for model training
- Built with Streamlit for intuitive data application development