import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load your dataset
METH = pd.read_csv('methane_data.csv')

# Keep only the first 6 columns and the 14th column
METH = METH.iloc[:, list(range(6)) + [13]]
MA_METH = METH.iloc[:, 6].rolling(window=10000, center=True).median()
EXP_METH = METH.iloc[:, 6].ewm(alpha=0.03, adjust=False).mean()  # Adjust 'span' as needed
DOUBLE_EXP_METH = EXP_METH.ewm(alpha=0.03, adjust=False).mean()

# Display basic information
print(f"Dataset shape: {METH.shape}")  # (rows, columns)
print("\nFirst 5 rows:")
print(MA_METH.head(100))
print("\nData types:")
print(METH.dtypes)

# Unfiltered RH1712 plot
# plt.figure(figsize=(25, 6))
# plt.plot(METH.iloc[:, 6], label='RH1712', color='blue', linewidth=0.7)

# Moving average RH1712 plot
# plt.figure(figsize=(25, 6))
# plt.plot(METH.iloc[:, 6], label='RH1712', color='red', linewidth=0.7)
# plt.plot(MA_METH, label='MA_RH1712', color='blue', linewidth=0.7)

# Exponential moving average RH1712 plot
plt.figure(figsize=(25, 6))
plt.plot(METH.iloc[:, 6], label='RH1712', color='red', linewidth=0.7)
# plt.plot(EXP_METH, label='EXP_RH1712', color='blue', linewidth=0.7)
plt.plot(DOUBLE_EXP_METH, label='DOUBLE_EXP_RH1712', color='blue', linewidth=0.7)

# Hide x-axis values
plt.xticks([])

# Customize the plot
plt.title('RH1712 filtered by double exponential')
plt.xlabel('Time')
plt.ylabel('Relative Humidity%')
plt.show()