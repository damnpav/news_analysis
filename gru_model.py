import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, GRU
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

features_and_target_df = pd.read_excel('features_and_target.xlsx')
features_and_target_df = features_and_target_df.fillna(0)
X = features_and_target_df.drop(columns=['log_price_spread'])
y = features_and_target_df['log_price_spread']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
input_shape = (X_train.shape[1], 1)

model = Sequential()
model.add(GRU(64, input_shape=input_shape, activation='relu'))
model.add(Dense(1, activation='linear'))
model.compile(loss='mean_absolute_error', optimizer='adam')
history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.2, verbose=1)
mae = model.evaluate(X_test, y_test, verbose=0)
train_pred = model.predict(X_train)
test_pred = model.predict(X_test)
