import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error, r2_score

features_and_target_df = pd.read_excel('features_and_target.xlsx')
features_and_target_df = features_and_target_df.fillna(0)


# Split the dataset into features (X) and target (y)
X = features_and_target_df.drop(columns=['log_price_spread'])
y = features_and_target_df['log_price_spread']

# Split the dataset into training and testing subsets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the SVM regression model
svm_regressor = SVR(kernel='linear')  # You can try other kernels like 'rbf', 'poly', etc.
svm_regressor.fit(X_train, y_train)

# Make predictions on the test set
y_pred = svm_regressor.predict(X_test)

# Calculate the Mean Absolute Error (MAE)
mae = mean_absolute_error(y_test, y_pred)

# Calculate the R-squared (R²) score
r2 = r2_score(y_test, y_pred)

# Print the evaluation metrics
print("Mean Absolute Error (MAE):", mae)
print("R-squared (R²) score:", r2)

