import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

features_and_target_df = pd.read_excel('features_and_target.xlsx')
features_and_target_df = features_and_target_df.fillna(0)
X = features_and_target_df.drop(columns=['log_price_spread'])
y = features_and_target_df['log_price_spread']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the XGBoost model
xgb_regressor = xgb.XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
xgb_regressor.fit(X_train, y_train)

# Make predictions on the train and test sets
y_train_pred = xgb_regressor.predict(X_train)
y_test_pred = xgb_regressor.predict(X_test)

# Calculate the Mean Absolute Error (MAE)
mae_train = mean_absolute_error(y_train, y_train_pred)
mae_test = mean_absolute_error(y_test, y_test_pred)

# Calculate the R-squared (R²) score
r2_train = r2_score(y_train, y_train_pred)
r2_test = r2_score(y_test, y_test_pred)

# Print the evaluation metrics
print("Train set:")
print("Mean Absolute Error (MAE):", mae_train)
print("R-squared (R²) score:", r2_train)

print("\nTest set:")
print("Mean Absolute Error (MAE):", mae_test)
print("R-squared (R²) score:", r2_test)
