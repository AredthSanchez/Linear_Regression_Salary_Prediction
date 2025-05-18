import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error


data = {
    'X': [
        0.1,0.2,0.20584494, 0.34388521, 0.46450413, 0.58083612, 0.65051593, 0.97672114, 1.22038235, 1.39493861,
        1.5599452, 1.5601864, 1.70524124, 1.81824967, 1.8340451, 1.84854456, 1.99673782, 2.12339111,
        2.58779982, 2.9122914, 2.92144649, 3.04242243, 3.04613769, 3.11711076, 3.66361843, 3.74540119,
        4.31945019, 4.40152494, 4.56069984, 4.9517691, 5.14234438, 5.20068021, 5.24756432, 5.46710279,
        5.92414569, 5.98658484, 6.01115012, 6.07544852, 6.11852895, 6.62522284, 6.84233027, 7.08072578,
        7.31993942, 7.85175961, 8.08397348, 8.32442641, 8.66176146, 9.09320402, 9.48885537, 9.50714306,
        9.65632033, 9.69909852
    ],
    'y': [
        9,8.5,5.41070918, 7.4325107, 6.6805335, 7.72515585, 8.65181004, 7.63262829, 10.33412462, 9.44750899,
        9.72693216, 8.20203726, 11.27144757, 10.1377079, 11.35176627, 11.65997278, 10.81570356, 12.33561845,
        13.58842488, 13.80628091, 14.46001825, 15.58151547, 16.87090428, 15.08925891, 15.39392966, 16.97467038,
        18.3594581, 18.60692591, 18.32964341, 18.24564195, 20.5412222, 20.15998631, 21.38269684, 21.28148061,
        21.11763535, 22.65865117, 22.05691217, 23.28788311, 24.58576929, 24.09852383, 25.18687795, 25.99422048,
        26.8441699, 29.64469559, 29.92477188, 30.34820291, 31.12369288, 30.30676672, 34.70877214, 33.6927973,
        34.76823377, 34.18657208
    ]
}

df = pd.DataFrame(data)
print(df.shape)
print(df.head(n=3))

x = df[['X']]
y = df['y']

linear_model = LinearRegression()
linear_model.fit(x,y)
predicted_salary = linear_model.predict(x)

plt.scatter(x,y)
plt.plot(x, predicted_salary, color = 'red', label = 'Regression Line')
plt.legend()
plt.xlabel("Years of Experience")
plt.ylabel("Salary")
plt.title("Scatter Plot: Years of Experience vs Monthly Salary")
plt.grid(True)
plt.show()

predicted_value = linear_model.predict([[12]])
predicted_value = predicted_value[0].round(2)
print(predicted_value)

mse_val = mean_squared_error(y, predicted_salary)
print(f'Mean Squared Error: {mse_val}')
