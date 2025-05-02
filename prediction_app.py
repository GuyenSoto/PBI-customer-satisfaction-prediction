# prediction_app.py
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler

# Title and description
st.title("Customer Satisfaction Predictor")
st.write("This application predicts whether a customer will be satisfied with their delivery service")

# Load the trained model (previously saved)
@st.cache_resource
def load_model():
    try:
        with open('ensemble_model.pkl', 'rb') as file:
            model = pickle.load(file)
        with open('scaler.pkl', 'rb') as file:
            scaler = pickle.load(file)
        with open('feature_info.pkl', 'rb') as file:
            feature_info = pickle.load(file)
        return model, scaler, feature_info
    except FileNotFoundError:
        st.error("Model files not found. Please run the training script first.")
        return None, None, None

model, scaler, feature_info = load_model()

# Check if the model loaded correctly
if model is not None:
    # Sidebar with information
    st.sidebar.header("Model Information")
    st.sidebar.write("Model: Voting Ensemble")
    st.sidebar.write("Model accuracy: 85.2%")
    
    # Display important features
    st.sidebar.header("Important Features")
    importance_df = pd.DataFrame({
        'Feature': feature_info['features'],
        'Importance': feature_info['importance']
    }).sort_values('Importance', ascending=False)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x='Importance', y='Feature', data=importance_df, ax=ax)
    st.sidebar.pyplot(fig)
    
    # Add feature description legend
    st.sidebar.header("Feature Descriptions")
    feature_descriptions = {
        'X1': "My order was delivered on time",
        'X2': "Contents of my order was as I expected",
        'X3': "I ordered everything I wanted to order",
        'X4': "I paid a good price for my order",
        'X5': "I am satisfied with my courier",
        'X6': "The app makes ordering easy for me"
    }
    
    for feature, description in feature_descriptions.items():
        st.sidebar.write(f"**{feature}**: {description}")
    
    # Main form
    st.header("Enter Customer Data")
    
    col1, col2 = st.columns(2)
    
    with col1:
        x1 = st.slider("Previous delivery timeliness (X1)", 1.0, 5.0, 3.0, 0.1,
                      help="How punctual has the service been for this customer in the past?")
        
        x2 = st.slider("Previous product quality (X2)", 1.0, 5.0, 3.0, 0.1,
                      help="How good has the quality of products delivered to this customer been?")
        
        x3 = st.slider("Previous order completeness (X3)", 1.0, 5.0, 3.0, 0.1,
                      help="How complete have this customer's previous orders been?")
    
    with col2:
        x4 = st.slider("Price perception (X4)", 1.0, 5.0, 3.0, 0.1,
                      help="How does the customer perceive the price of the products?")
        
        x5 = st.slider("Previous courier satisfaction (X5)", 1.0, 5.0, 3.0, 0.1,
                      help="How has the previous satisfaction with the courier service been?")
        
        x6 = st.slider("App ease of use (X6)", 1.0, 5.0, 3.0, 0.1,
                      help="How easy does the customer find using the app for ordering?")
    
    # Additional information about the current order
    st.header("Additional Current Order Information")
    col3, col4 = st.columns(2)
    
    with col3:
        distance = st.number_input("Delivery distance (km)", 0.1, 20.0, 5.0, 0.1)
        estimated_time = st.number_input("Estimated delivery time (minutes)", 10, 120, 30, 5)
    
    with col4:
        order_complexity = st.slider("Order complexity", 1, 10, 5, 
                                     help="1: Very simple, 10: Extremely complex")
        courier_rating = st.slider("Average courier rating", 1.0, 5.0, 4.0, 0.1)
    
    # Button to make the prediction
    if st.button("Predict Satisfaction"):
        # Prepare data for prediction
        input_data = pd.DataFrame({
            'X1': [x1],
            'X2': [x2],
            'X3': [x3],
            'X4': [x4],
            'X5': [x5],
            'X6': [x6]
        })
        
        # Scale the data
        input_scaled = scaler.transform(input_data)
        
        # Make prediction
        prediction = model.predict(input_scaled)[0]
        probability = model.predict_proba(input_scaled)[0][1]
        
        # Display results
        st.header("Results")
        
        # Create a visualization of the satisfaction level
        fig, ax = plt.subplots(figsize=(10, 2))
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_title("Satisfaction Probability")
        ax.add_patch(plt.Rectangle((0, 0), probability, 0.5, color='green', alpha=0.3))
        ax.add_patch(plt.Rectangle((probability, 0), 1-probability, 0.5, color='red', alpha=0.3))
        ax.text(probability/2, 0.25, f"{probability:.1%}", ha='center', va='center')
        ax.text(probability + (1-probability)/2, 0.25, f"{1-probability:.1%}", ha='center', va='center')
        ax.set_yticks([])
        ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
        ax.set_xticklabels(['0%', '25%', '50%', '75%', '100%'])
        st.pyplot(fig)
        
        # Show interpretation
        if prediction == 1:
            st.success(f"Prediction: Customer SATISFIED (Probability: {probability:.1%})")
        else:
            st.error(f"Prediction: Customer UNSATISFIED (Probability: {probability:.1%})")
            
        # Recommendations based on analysis
        st.header("Recommendations")
        
        if prediction == 0:
            st.write("To improve the customer experience:")
            
            # Identify the weakest features
            input_data_with_means = input_data.copy()
            for col in input_data.columns:
                mean_happy = feature_info['avg_happy'].get(col, 0)
                current_val = input_data[col].values[0]
                if current_val < mean_happy:
                    st.write(f"- **Improve {col}**: Current value ({current_val:.1f}) is below average for satisfied customers ({mean_happy:.1f}).")
            
            # Specific recommendations based on the current order
            if distance > 8:
                st.write("- **Distance Warning**: This order has a greater than average distance, consider adjusting delivery time expectations.")
            
            if order_complexity > 7:
                st.write("- **Complex Order**: This order has high complexity. Consider an additional verification before shipping.")
            
            if estimated_time > 45:
                st.write("- **Long Delivery Time**: Consider offering a small discount or additional benefit for the wait time.")
        else:
            st.write("The customer will likely be satisfied, but you can further improve:")
            
            if probability < 0.8:
                st.write("- The satisfaction probability is not very high. Consider adding an extra detail to ensure a good experience.")
            
            st.write("- Keep the customer informed about the status of their order.")
            st.write("- Make sure the order is complete and well packed.")
else:
    st.warning("Could not load the model. Please run the training script first.")