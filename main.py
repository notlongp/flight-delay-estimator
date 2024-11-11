import streamlit as st
import pickle
import pandas as pd
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, split
from datetime import datetime
from pyspark.ml.pipeline import PipelineModel

# Initialize PySpark session
spark = SparkSession.builder.appName("FlightPrediction").getOrCreate()

classfier_path = "decision_tree_c.model"
regressor_path = "decision_tree_r.model"

classifier = PipelineModel.load(classfier_path)
regressor = PipelineModel.load(regressor_path)

# Function to preprocess input data for prediction
def preprocess_data(origin_airport, date_of_flight, distance, airline):
    # Convert the date string to a datetime object
    date = datetime.strptime(date_of_flight, '%Y-%m-%d')
    
    # Create a dataframe (adjust the features based on your model's expected input)
    data = {
        'FL_MONTH': [date],
        'ORIGIN': [origin_airport],
        'DISTANCE': [distance],
        'AIRLINE_NAME': [airline]
    }
    
    df = pd.DataFrame(data)
    spark_df = spark.createDataFrame(df)

    # Assuming the model expects specific feature transformations (e.g., one-hot encoding, date processing)
    # Here we are just converting the columns into features. Adjust as necessary.
    spark_df = spark_df.withColumn('FL_MONTH_str', col('FL_MONTH').cast('string'))
    spark_df = spark_df.withColumn('FL_MONTH_str', split(col('FL_MONTH_str'), '-')[1].cast('int'))
    spark_df = spark_df.withColumn('FL_MONTH_str', col('FL_MONTH_str').cast('string'))    
    spark_df = spark_df.withColumn('DISTANCE', col('DISTANCE').cast('float'))

    # Add other transformations based on model feature engineering

    return spark_df

# Streamlit UI for user input
st.title("Flight Prediction Dashboard")
st.write("Enter flight details to predict flight-related information:")

# Input fields for the dashboard
origin_airport = st.text_input("Origin Airport", "JFK")
distance = st.number_input("Distance between airports", 0)
date_of_flight = st.date_input("Date of Flight", datetime.today())

# Airline dropdown selection
airline_list = [
    "American Airlines Inc.",
    "Delta Air Lines Inc.",
    "Frontier Airlines Inc.",
    "JetBlue Airways",
    "Southwest Airlines Co.",
    "Spirit Air Lines",
    "United Air Lines Inc.",
]
airline = st.selectbox("Airline", airline_list)

# Convert the input date to a string format
date_of_flight = date_of_flight.strftime('%Y-%m-%d')

# Predict button
if st.button("Predict"):
    # Preprocess the input data
    processed_data = preprocess_data(origin_airport, date_of_flight, distance, airline)

    pred_c_df = classifier.transform(processed_data)
    print(pred_c_df.select('prediction').show())
    pred_c = pred_c_df.select('prediction').first()[0]
    pred_r_df = regressor.transform(processed_data)
    pred_r = pred_r_df.select('prediction').first()[0]

    st.write("Your flight is likely not going to be delayed" 
             if int(pred_c) == 0 
             else "Your flight is likely going to be delayed")
    if (int(pred_c) == 1):
      st.write(f"Estimated Delay time: {pred_r*0.75:.2f} to {pred_r*1.25:.2f} minutes")
