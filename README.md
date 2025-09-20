# Multi-Disease Early Prediction System

This project is a machine learning-based system to predict **Diabetes, Heart Disease, Chronic Kidney Disease, Parkinson's Disease, and Breast Cancer**. Users can input features for a patient and get early risk predictions via a Streamlit web app.


## Project Structure

- `data/` : Contains CSV datasets used for training
- `models/` : Pre-trained machine learning models (.pkl)
- `scalers/` : Saved feature scalers (.pkl) for input standardization
- `app.py` : Streamlit web application for predictions
- `requirements.txt` : Python dependencies


## How to Run

1. Clone the repository:
git clone https://github.com/namanguptaa24/multi_disease_project.git
2. Navigate to the project folder:
cd multi_disease_project
3. Install dependencies:
pip install -r requirements.txt
4. Run the Streamlit app:
streamlit run app.py
5. Open the browser and use the app.
