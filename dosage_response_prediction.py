import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
import numpy as np

data = {
  "x": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
  "y": [0, 0, 0, 0, 1, 1, 0, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1]
}

df = pd.DataFrame(data)
print(df.head())
print(df.shape)

x = df[['x']]
y = df['y']

model = LogisticRegression()
model.fit(x,y)

x_values = pd.DataFrame(np.linspace(x.min(), x.max(), 100), columns = ['x'])
y_values = model.predict_proba(x_values).T[1]

plt.scatter(x, y, color = 'blue')
plt.plot(x_values, y_values, color = 'red', label = 'Logistic Regression')
plt.legend()
plt.xlabel('Drug Dosage (x)')
plt.ylabel('Recovery (y)')
plt.title('Data Visualization')
plt.grid(True)
plt.show()

new_x = pd.DataFrame([[12]], columns = ['x'])
predicted_class = model.predict(new_x)
print(f"Predicted class is: {predicted_class[0]}")
