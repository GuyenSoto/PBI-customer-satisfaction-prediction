# prediction_app_enhanced.py
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
import os
import datetime

# Define output directory for images
OUTPUT_DIR = "OUTPUT"
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# Function to save figures with timestamp
def save_figure(fig, base_name):
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H%M%S")
    filename = f"{base_name}_{timestamp}.jpg"
    filepath = os.path.join(OUTPUT_DIR, filename)
    fig.savefig(filepath, dpi=300, bbox_inches='tight')
    return filepath

# Title and description
st.title("Enhanced Customer Satisfaction Predictor")
st.write("This application predicts whether a customer will be satisfied with their delivery service, including logistics variables")

# Load the trained model (previously saved)
@st.cache_resource
def load_model():
    try:
        with open('ensemble_model_enhanced.pkl', 'rb') as file:
            model = pickle.load(file)
        with open('scaler_enhanced.pkl', 'rb') as file:
            scaler = pickle.load(file)
        with open('feature_info_enhanced.pkl', 'rb') as file:
            feature_info = pickle.load(file)
        return model, scaler, feature_info
    except FileNotFoundError:
        st.error("Enhanced model files not found. Please run the enhanced training script first.")
        return None, None, None

# Load feature columns to ensure consistency
@st.cache_resource
def load_feature_columns():
    try:
        with open('feature_columns_enhanced.pkl', 'rb') as file:
            feature_columns = pickle.load(file)
        return feature_columns
    except FileNotFoundError:
        st.error("Feature columns file not found. Please run the training script first.")
        return None

model, scaler, feature_info = load_model()
feature_columns = load_feature_columns()

# Check if the model loaded correctly
if model is not None and feature_columns is not None:
    # Sidebar with information
    st.sidebar.header("Model Information")
    st.sidebar.write("Model: Voting Ensemble with Delivery Metrics")
    st.sidebar.write("Model accuracy: 87.5%")
    
    # Display important features
    st.sidebar.header("Important Features")
    importance_df = pd.DataFrame({
        'Feature': feature_info['features'],
        'Importance': feature_info['importance']
    }).sort_values('Importance', ascending=False)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x='Importance', y='Feature', data=importance_df[:10], ax=ax)
    st.sidebar.pyplot(fig)
    
    # Save the feature importance figure
    save_figure(fig, "feature_importance_sidebar")
    
    # Add feature description legend
    st.sidebar.header("Feature Descriptions")
    feature_descriptions = {
        'X1': "My order was delivered on time",
        'X2': "Contents of my order was as I expected",
        'X3': "I ordered everything I wanted to order",
        'X4': "I paid a good price for my order",
        'X5': "I am satisfied with my courier",
        'X6': "The app makes ordering easy for me",
        'DeliveryDistance': "Distance from distribution center to customer in km",
        'DeliveryTime': "Total estimated time from order confirmation to delivery"
    }
    
    for feature, description in feature_descriptions.items():
        st.sidebar.write(f"**{feature}**: {description}")
    
    # Main form
    st.header("Enter Customer Data")
    
    # Organize inputs in three columns
    col1, col2, col3 = st.columns(3)
    
    with col1:
        x1 = st.slider("Previous delivery timeliness (X1)", 1.0, 5.0, 3.0, 0.1,
                      help="How punctual has the service been for this customer in the past?")
        
        x2 = st.slider("Previous product quality (X2)", 1.0, 5.0, 3.0, 0.1,
                      help="How good has the quality of products delivered to this customer been?")
    
    with col2:
        x3 = st.slider("Previous order completeness (X3)", 1.0, 5.0, 3.0, 0.1,
                      help="How complete have this customer's previous orders been?")
        
        x4 = st.slider("Price perception (X4)", 1.0, 5.0, 3.0, 0.1,
                      help="How does the customer perceive the price of the products?")
    
    with col3:
        x5 = st.slider("Previous courier satisfaction (X5)", 1.0, 5.0, 3.0, 0.1,
                      help="How has the previous satisfaction with the courier service been?")
        
        x6 = st.slider("App ease of use (X6)", 1.0, 5.0, 3.0, 0.1,
                      help="How easy does the customer find using the app for ordering?")
    
    # Current order logistics information
    st.header("Current Order Logistics Information")
    
    # Use two columns for logistics information
    log_col1, log_col2 = st.columns(2)
    
    with log_col1:
        distance = st.number_input("Delivery distance (km)", 0.1, 20.0, 5.0, 0.1,
                                  help="Distance in kilometers from distribution center to customer")
        
        estimated_time = st.number_input("Estimated delivery time (minutes)", 10, 120, 30, 5,
                                       help="Total estimated time from order confirmation to delivery")
    
    with log_col2:
        order_complexity = st.slider("Order complexity", 1, 10, 5, 
                                     help="1: Very simple, 10: Extremely complex")
        
        courier_rating = st.slider("Average courier rating", 1.0, 5.0, 4.0, 0.1,
                                 help="Historical average rating of the assigned courier")
    
    # Button to make the prediction
    if st.button("Predict Satisfaction"):
        # Prepare data for prediction
        input_data = pd.DataFrame({
            'X1': [x1],
            'X2': [x2],
            'X3': [x3],
            'X4': [x4],
            'X5': [x5],
            'X6': [x6],
            'DeliveryDistance': [distance],
            'DeliveryTime': [estimated_time]
        })
        
        # Scale the data
        input_scaled = scaler.transform(input_data)
        input_scaled_df = pd.DataFrame(input_scaled, columns=input_data.columns)
        
        # Create a DataFrame with all the required columns, initialized to 0
        input_enhanced = pd.DataFrame(0, index=np.arange(1), columns=feature_columns)
        
        # Fill the basic columns with scaled values
        for col in input_scaled_df.columns:
            if col in input_enhanced.columns:
                input_enhanced[col] = input_scaled_df[col].values
        
        # Add the same interaction features as in the training model
        input_enhanced['DeliveryDistance_x_Distance'] = input_scaled_df['DeliveryDistance'] * input_scaled_df['DeliveryDistance']
        input_enhanced['DeliveryDistance_x_Time'] = input_scaled_df['DeliveryDistance'] * input_scaled_df['DeliveryTime']
        input_enhanced['DeliveryTime_x_Distance'] = input_scaled_df['DeliveryTime'] * input_scaled_df['DeliveryDistance']
        input_enhanced['DeliveryTime_x_Time'] = input_scaled_df['DeliveryTime'] * input_scaled_df['DeliveryTime']
        
        # Make sure columns are in the same order as during training
        input_enhanced = input_enhanced[feature_columns]
        
        # Make prediction
        prediction = model.predict(input_enhanced)[0]
        probability = model.predict_proba(input_enhanced)[0][1]
        
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
        
        # Save the results visualization
        save_figure(fig, "prediction_result")
        
        # Show interpretation
        if prediction == 1:
            st.success(f"Prediction: Customer SATISFIED (Probability: {probability:.1%})")
        else:
            st.error(f"Prediction: Customer UNSATISFIED (Probability: {probability:.1%})")
            
        # Recommendations based on analysis
        st.header("Recommendations")
        
        # Determine what factors most affect the prediction
        risk_factors = []
        
        # Check risk factors based on distance and time
        if distance > 10:
            risk_factors.append(("Long Distance", 
                                  "The customer is more than 10km away, which significantly increases the risk of dissatisfaction."))
        
        if estimated_time > 45:
            risk_factors.append(("Extended Wait Time", 
                                  "The delivery time exceeds 45 minutes, which may cause impatience."))
        
        if order_complexity > 7 and estimated_time < 45:
            risk_factors.append(("Insufficient Time for Complex Order", 
                                  "The order is complex but the estimated time seems optimistic."))
        
        # Check previous experience factors
        if x1 < 3.5:
            risk_factors.append(("History of Late Deliveries", 
                                  "This customer has experienced delays in the past and may be more sensitive to timing."))
        
        if x5 < 3.0:
            risk_factors.append(("Previous Courier Dissatisfaction", 
                                  "The customer has had negative experiences with the delivery service."))
        
        # Show specific recommendations based on analysis
        if prediction == 0 or probability < 0.7:
            st.write("⚠️ **Risk factors have been detected in this order:**")
            
            for factor, description in risk_factors:
                st.write(f"- **{factor}**: {description}")
            
            st.write("**Recommended Actions:**")
            
            # Specific recommendations for each risk factor
            if distance > 10:
                st.write("- **Adjust Expectations**: Clearly communicate the expected delivery time.")
                st.write("- **Prioritize Dispatch**: Process this order with priority to compensate for distance.")
            
            if estimated_time > 45:
                st.write("- **Active Tracking**: Provide updates during delivery.")
                st.write("- **Wait Incentive**: Consider offering a small discount or additional product.")
            
            if order_complexity > 7:
                st.write("- **Double Verification**: Perform an additional check of the order before dispatch.")
                st.write("- **Experienced Courier**: Assign a courier with experience in complex orders.")
            
            # Add specific recommendations for history of late deliveries
            if x1 < 3.5:
                st.write("- **Proactive Communication**: Establish a more frequent communication pattern with this customer.")
                st.write("- **Time Buffer**: Add a small buffer to the estimated delivery time to ensure on-time delivery.")
            
            if x5 < 3.0 and courier_rating < 4.5:
                st.write("- **Premium Courier**: Assign a courier with a rating higher than 4.5 for this customer.")
            
            # General recommendation if many risk factors
            if len(risk_factors) >= 3:
                st.write("- **VIP Attention**: This order requires special supervision. Consider assigning a manager to monitor the entire process.")
                
            # Save the recommendations view
            plt.figure(figsize=(10, 6))
            plt.text(0.5, 0.5, "Risk Factors & Recommendations", fontsize=20, ha='center')
            plt.axis('off')
            plt.tight_layout()
            fig_rec = plt.gcf()
            save_figure(fig_rec, "prediction_recommendations")
            
        else:
            st.write("👍 **The customer will likely be satisfied, but you can improve even more:**")
            
            # Some general recommendations for low-risk orders
            st.write("- Keep the customer informed about the status of their order.")
            st.write("- Make sure the order is complete and well packed.")
            
            if courier_rating < 4.8 and probability > 0.9:
                st.write("- The order seems low risk, it may be a good opportunity to assign a courier in training.")
else:
    st.warning("Could not load the enhanced model. Please run the training script first.")