# UPI Transaction Fraud Detection 

This project implements a fraud detection model using a Random Forest Classifier to predict fraudulent UPI transactions.
The code is written in Python and uses scikit-learn, pandas, and other essential libraries for data preprocessing, model training, and evaluation.

# Key Features
Automatic Preprocessing

Detects numeric features automatically.

Converts non-numeric features to numeric when possible.

Drops rows with missing values in required columns.

Balanced Class Handling

Uses class_weight='balanced' in the Random Forest model to handle class imbalance.

Automatic Train-Test Split

Uses stratified splitting (if both classes have at least 2 samples).

Performance Metrics

Prints accuracy and a detailed classification report.

# Future Enhancements
Deploy the model as a web API or web application.

Improve fraud detection with ensemble learning or deep learning models.
