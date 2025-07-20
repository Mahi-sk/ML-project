import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score


df = pd.read_csv('upi_transactions_2024.csv')


print("First 5 rows of the dataset:")
print(df.head())
print("\nInformation about the dataset:")
print(df.info())
print(f"\nInitial DataFrame shape: {df.shape}")

# Check for 'is_fraud' column, if not found, create a dummy one for demonstration
if 'is_fraud' not in df.columns:
    print("\n'is_fraud' column not found. Creating a dummy 'is_fraud' column for demonstration.")
    # Create a dummy 'is_fraud' column (e.g., set ~10% of transactions as fraud)
    df['is_fraud'] = (df.index % 10 == 0).astype(int)
else:
    print("\n'is_fraud' column found.")

# Selecting numeric features
numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
feature_columns = [col for col in numeric_cols if col not in ['is_fraud', 'fraud_flag']] 


if not feature_columns:
    print("\nNo numeric feature columns found automatically. Using all columns except 'transaction_id', 'timestamp', 'is_fraud', 'fraud_flag' as features.")
    feature_columns = [col for col in df.columns if col not in ['transaction_id', 'timestamp', 'is_fraud', 'fraud_flag']]
    for col in feature_columns:
        if df[col].dtype == 'object':
            try:
                df[col] = pd.to_numeric(df[col], errors='coerce')
                print(f"Converted column '{col}' to numeric.")
            except:
                print(f"Could not convert column '{col}' to numeric.")

# Dropping rows with any NaN values in the selected feature columns or target
original_shape_before_na = df.shape
df.dropna(subset=feature_columns + ['is_fraud'], inplace=True)
print(f"DataFrame shape after dropping NaN values: {df.shape}")
print(f"Number of rows dropped due to NaN: {original_shape_before_na[0] - df.shape[0]}")

# if the DataFrame is empty after dropping NaNs
if df.empty:
    print("\nDataFrame is empty after preprocessing. Cannot proceed with model training.")
else:
    X = df[feature_columns]
    y = df['is_fraud']

    print(f"\nShape of X (features) before split: {X.shape}")
    print(f"Shape of y (target) before split: {y.shape}")

    value_counts = y.value_counts()
    if (value_counts < 2).any():
        print("\nCannot use stratify=y because at least one class has fewer than 2 samples. Splitting without stratification.")
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    else:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)


    model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')

    model.fit(X_train, y_train)

 
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)

    print(f"\nModel Accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(report)