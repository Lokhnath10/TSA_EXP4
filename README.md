# Ex.No:04   FIT ARMA MODEL FOR TIME SERIES
# Date: 24.09.25



### AIM:
To implement ARMA model in python.
### ALGORITHM:
1. Import necessary libraries.
2. Set up matplotlib settings for figure size.
3. Define an ARMA(1,1) process with coefficients ar1 and ma1, and generate a sample of 1000

data points using the ArmaProcess class. Plot the generated time series and set the title and x-
axis limits.

4. Display the autocorrelation and partial autocorrelation plots for the ARMA(1,1) process using
plot_acf and plot_pacf.
5. Define an ARMA(2,2) process with coefficients ar2 and ma2, and generate a sample of 10000

data points using the ArmaProcess class. Plot the generated time series and set the title and x-
axis limits.

6. Display the autocorrelation and partial autocorrelation plots for the ARMA(2,2) process using
plot_acf and plot_pacf.
### PROGRAM:
```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.arima_process import ArmaProcess
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

data = pd.read_csv('/content/Gold Price (2013-2023).csv')

data['Price'] = data['Price'].str.replace(',', '').astype(float)

X = data['Price']
plt.rcParams['figure.figsize'] = [12, 6]
plt.plot(X)
plt.title('Original Gold Price Data')
plt.show()

plt.subplot(2, 1, 1)
plot_acf(X, lags=int(len(X)/4), ax=plt.gca())
plt.title('Original Data ACF')

plt.subplot(2, 1, 2)
plot_pacf(X, lags=int(len(X)/4), ax=plt.gca())
plt.title('Original Data PACF')

plt.tight_layout()
plt.show()

arma11_model = ARIMA(X, order=(1, 0, 1)).fit()
phi1_arma11 = arma11_model.params['ar.L1']
theta1_arma11 = arma11_model.params['ma.L1']

ar1 = np.array([1, -phi1_arma11])
ma1 = np.array([1, theta1_arma11])

N = 1000
ARMA_1 = ArmaProcess(ar1, ma1).generate_sample(nsample=N)

plt.plot(ARMA_1)
plt.title('Simulated ARMA(1,1) Gold Prices')
plt.xlim([0, 500])
plt.show()

plot_acf(ARMA_1)
plt.show()
plot_pacf(ARMA_1)
plt.show()

arma22_model = ARIMA(X, order=(2, 0, 2)).fit()
phi1_arma22 = arma22_model.params['ar.L1']
phi2_arma22 = arma22_model.params['ar.L2']
theta1_arma22 = arma22_model.params['ma.L1']
theta2_arma22 = arma22_model.params['ma.L2']

ar2 = np.array([1, -phi1_arma22, -phi2_arma22])
ma2 = np.array([1, theta1_arma22, theta2_arma22])

ARMA_2 = ArmaProcess(ar2, ma2).generate_sample(nsample=N*10)

plt.plot(ARMA_2)
plt.title('Simulated ARMA(2,2) Gold Prices')
plt.xlim([0, 500])
plt.show()

plot_acf(ARMA_2)
plt.show()
plot_pacf(ARMA_2)
plt.show()
```

OUTPUT:
<img width="986" height="519" alt="image" src="https://github.com/user-attachments/assets/ec7e13c2-eb8f-4149-a5e7-3bfe310812ad" />
<img width="1177" height="276" alt="image" src="https://github.com/user-attachments/assets/611d1c2f-bda1-4103-bcf5-0b004213ebe3" />
<img width="1176" height="289" alt="image" src="https://github.com/user-attachments/assets/64dd0619-e4a1-47a2-a440-562e0f4eb17a" />
<img width="986" height="529" alt="image" src="https://github.com/user-attachments/assets/f3e4e818-c61a-4cc3-b918-4b3079fd0298" />
<img width="987" height="525" alt="image" src="https://github.com/user-attachments/assets/73b44d53-a6fe-4b7e-8e9d-134b390f67ef" />
<img width="990" height="513" alt="image" src="https://github.com/user-attachments/assets/1ec0532f-11ff-4392-bca5-0c2509a3431f" />






RESULT:
Thus, a python program is created to fir ARMA Model successfully.
