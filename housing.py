from tensorflow import keras
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('IowaHousingPrices.csv')

squareFeet = df[['SquareFeet']].values
salePrice = df[['SalePrice']].values

model = keras.Sequential()
model.add(keras.layers.Dense(1, input_shape=(1,)))
model.compile(keras.optimizers.Adam(learning_rate=1), 'mean_squared_error')

model.fit(squareFeet, salePrice, batch_size=10, epochs=30)

#plot data
df.plot(kind='scatter', x='SquareFeet', y='SalePrice', title='Housing Prices and Square Footage of Iowa Homes')

y_pred = model.predict(squareFeet)

plt.plot(squareFeet, y_pred, color='red')