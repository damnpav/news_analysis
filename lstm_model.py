import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, LSTM
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

features_and_target_df = pd.read_excel('features_and_target.xlsx')
features_and_target_df = features_and_target_df.fillna(0)
X = features_and_target_df.drop(columns=['log_price_spread'])
y = features_and_target_df['log_price_spread']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the LSTM model
lstm_model = Sequential()
lstm_model.add(LSTM(64, input_shape=(X_train.shape[1], 1)))
lstm_model.add(Dense(1))

X_train = X_train.values.reshape(X_train.shape[0], X_train.shape[1], 1)
X_test = X_test.values.reshape(X_test.shape[0], X_test.shape[1], 1)

# Compile the model
lstm_model.compile(loss='mse', optimizer='adam')

# Train the model
history = lstm_model.fit(X_train.reshape(X_train.shape[0], X_train.shape[1], 1), y_train,
                         epochs=50, batch_size=32, verbose=1, validation_split=0.2)

# Make predictions on the train and test sets
y_train_pred = lstm_model.predict(X_train.reshape(X_train.shape[0], X_train.shape[1], 1)).flatten()
y_test_pred = lstm_model.predict(X_test.reshape(X_test.shape[0], X_test.shape[1], 1)).flatten()

# Calculate the MAE and R-squared (R²) score for the train set
train_mae = mean_absolute_error(y_train, y_train_pred)
train_r2 = r2_score(y_train, y_train_pred)

# Calculate the MAE and R-squared (R²) score for the test set
test_mae = mean_absolute_error(y_test, y_test_pred)
test_r2 = r2_score(y_test, y_test_pred)






