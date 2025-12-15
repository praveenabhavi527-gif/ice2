import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import seaborn as sns

# -----------------------------------------------------------------------------
# 1. APP CONFIGURATION & STYLING
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Ice Cream Sales Predictor",
    page_icon="🍦",
    layout="wide"
)

# Custom CSS for a better look
st.markdown("""
<style>
    .main {
        background-color: #f8f9fa;
    }
    .stButton>button {
        width: 100%;
        background-color: #ff4b4b;
        color: white;
        font-weight: bold;
        border-radius: 10px;
    }
    .metric-card {
        background-color: #ffffff;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        text-align: center;
    }
    h1 {
        color: #ff4b4b;
    }
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 2. DATA LOADING / GENERATION
# -----------------------------------------------------------------------------
@st.cache_data
def load_data():
    """
    Since we don't have the original CSV file from the screenshot,
    we generate synthetic data that mimics the statistical properties
    seen in the PDF (Temp: 20-40, Sales: 280-630).
    """
    np.random.seed(42)
    n_samples = 60
    
    # Generate Temperature between 20 and 40 (as per PDF min/max)
    temperature = np.random.uniform(20, 40, n_samples)
    
    # Generate Sales with a linear relationship + some noise
    # Slope ~17, Intercept ~ -70 estimated to match the prediction of 680 at 45 degrees
    sales = (17 * temperature) - 80 + np.random.normal(0, 25, n_samples)
    
    df = pd.DataFrame({
        'Temperature_C': temperature,
        'IceCream_Sales': sales
    })
    return df

# Load the data
try:
    # If you have the actual CSV, uncomment the line below and comment out the synthetic generation
    # ice = pd.read_csv('ice_cream_temp_sales.csv')
    ice = load_data()
except Exception as e:
    st.error(f"Error loading data: {e}")
    st.stop()

# -----------------------------------------------------------------------------
# 3. MODEL TRAINING
# -----------------------------------------------------------------------------
X = ice[['Temperature_C']]
y = ice['IceCream_Sales']

model = LinearRegression()
model.fit(X, y)

# -----------------------------------------------------------------------------
# 4. SIDEBAR - USER INPUT
# -----------------------------------------------------------------------------
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/1046/1046769.png", width=100)
    st.title("Settings")
    st.markdown("Adjust the temperature to predict revenue.")
    
    # Input Slider
    input_temp = st.slider(
        "Temperature (°C)", 
        min_value=10.0, 
        max_value=50.0, 
        value=45.0, # Default value from PDF
        step=0.1
    )
    
    st.markdown("---")
    st.write("Based on Scikit-Learn Linear Regression")

# -----------------------------------------------------------------------------
# 5. MAIN CONTENT
# -----------------------------------------------------------------------------
st.title("🍦 Ice Cream Sales Forecaster")
st.markdown("### Predict daily revenue based on weather conditions")

# Create two columns: Prediction & Stats
col1, col2 = st.columns([1, 2])

# -- Prediction Logic --
prediction = model.predict(pd.DataFrame([[input_temp]], columns=['Temperature_C']))[0]

with col1:
    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
    st.markdown("### Predicted Revenue")
    st.markdown(f"<h1 style='font-size: 3.5rem; color: #2e7d32;'>${prediction:,.2f}</h1>", unsafe_allow_html=True)
    st.markdown(f"at **{input_temp}°C** Temperature")
    st.markdown('</div>', unsafe_allow_html=True)
    
    st.info(f"""
    **Model Insights:**
    - Dataset size: {len(ice)} days
    - Temp Range: {ice['Temperature_C'].min():.1f}°C - {ice['Temperature_C'].max():.1f}°C
    """)

# -- Visualization --
with col2:
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot training data
    sns.scatterplot(x='Temperature_C', y='IceCream_Sales', data=ice, color='#888', alpha=0.6, s=100, label='Historical Data', ax=ax)
    
    # Plot Regression Line
    line_x = np.linspace(10, 50, 100).reshape(-1, 1)
    line_y = model.predict(pd.DataFrame(line_x, columns=['Temperature_C']))
    plt.plot(line_x, line_y, color='#ff4b4b', linewidth=2, label='Regression Line')
    
    # Plot the User Prediction Point
    plt.scatter([input_temp], [prediction], color='#2e7d32', s=200, zorder=5, edgecolor='black', label='Your Prediction')
    
    # Styling the plot
    ax.set_title(f"Revenue vs. Temperature", fontsize=15)
    ax.set_xlabel("Temperature (°C)", fontsize=12)
    ax.set_ylabel("Sales ($)", fontsize=12)
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.legend()
    
    # Use standard Streamlit plot display
    st.pyplot(fig)

# -----------------------------------------------------------------------------
# 6. DATA INSPECTION (Expandable)
# -----------------------------------------------------------------------------
with st.expander("🔍 View Raw Training Data"):
    st.dataframe(ice.style.background_gradient(cmap="Reds"), use_container_width=True)
How to run this:
Make sure you have Python installed.
Install the requirements:
code
Bash
pip install -r requirements.txt
Run the application:
code
Bash
streamlit run app.py
What this code does (Visual Explanation):
Replicates the PDF Logic: It uses sklearn.LinearRegression, takes temperature as input, and predicts sales.
Solves the "File Not Found" issue: Since I don't have your specific CSV, the load_data() function creates 60 random data points that look exactly like the data in your screenshots (Range 20-40C).
Design:
It uses a "Wide" layout.
It includes a Sidebar for the slider input.
It displays the prediction in a large, styled "Metric Card".
It draws a Scatter Plot (using Matplotlib/Seaborn) showing the original data points, the red regression line, and a large green dot representing the specific prediction you just made.
Model
ThinkingThoughts
Expand to view model thoughts

chevron_right
To do this, we need to split the process into two steps:
train_model.py: A script you run once to train the model and save it as a file (model.pkl).
app.py: The Streamlit app that loads that file and uses it.
Here is the exact code for both.
Step 1: Create train_model.py
Run this file once to generate the model.pkl file.
code
Python
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import pickle

# 1. Load Data (Using synthetic data to match your PDF stats)
# If you have your CSV, use: df = pd.read_csv('ice_cream_temp_sales (1).csv')
def get_data():
    np.random.seed(42)
    n_samples = 60
    temperature = np.random.uniform(20, 40, n_samples)
    sales = (17 * temperature) - 80 + np.random.normal(0, 25, n_samples)
    return pd.DataFrame({'Temperature_C': temperature, 'IceCream_Sales': sales})

ice = get_data()

# 2. Prepare Data
X = ice[['Temperature_C']]
y = ice['IceCream_Sales']

# 3. Train Model
LR = LinearRegression()
LR.fit(X, y)

# 4. Save model to pickle file
with open('model.pkl', 'wb') as file:
    pickle.dump(LR, file)

print("Success! Model saved as 'model.pkl'")
