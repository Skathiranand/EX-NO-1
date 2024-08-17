# Exno:1
Data Cleaning Process

# AIM
To read the given data and perform data cleaning and save the cleaned data to a file.

# Explanation
Data cleaning is the process of preparing data for analysis by removing or modifying data that is incorrect ,incompleted , irrelevant , duplicated or improperly formatted. Data cleaning is not simply about erasing data ,but rather finding a way to maximize datasets accuracy without necessarily deleting the information.

# Algorithm
STEP 1: Read the given Data

STEP 2: Get the information about the data

STEP 3: Remove the null values from the data

STEP 4: Save the Clean data to the file

STEP 5: Remove outliers using IQR

STEP 6: Use zscore of to remove outliers

# Coding and Output
## Data Cleaning
```
import pandas as pd
df=pd.read_csv("/content/SAMPLEIDS.csv")
df

```
![output](./Outputs/D1.png)

```
df.isnull().sum()
```
![output](./Outputs/D2.png)

```
df.isnull().any()
```
![output](./Outputs/D3.png)

```
df.dropna()
```
![output](./Outputs/D4.png)

```
df.fillna(0)
```
![output](./Outputs/D5.png)

```
df.fillna(method = "ffill")
```
![output](./Outputs/D6.png)

```
df.fillna(method = 'bfill')
```
![output](./Outputs/D7.png)

```
df_dropped = df.dropna()
df_dropped
```
![output](./Outputs/D8.png)

```
df.fillna({'GENDER':'MALE','NAME':'SRI','ADDRESS':'POONAMALEE','M1':98,'M2':87,'M3':76,'M4':92,'TOTAL':305,'AVG':89.999999})
```
![output](./Outputs/D9.png)

## IQR(Inter Quartile Range)
```
ir=pd.read_csv('iris.csv')
ir
```
![output](./Outputs/I1.png)

```
ir.describe()
```
![output](./Outputs/I2.png)

```
import seaborn as sns
import matplotlib.pyplot as plt
sns.boxplot(x='sepal_width',data=ir)
plt.show()

```
![output](./Outputs/I3.png)

```
 c1=ir.sepal_width.quantile(0.25)
 c3=ir.sepal_width.quantile(0.75)
 iq=c3-c1
 print(c3)
```
![output](./Outputs/I4.png)

```
rid=ir[((ir.sepal_width<(c1-1.5*iq))|(ir.sepal_width>(c3+1.5*iq)))]
rid['sepal_width']
```
![output](./Outputs/I5.png)

```
delid=ir[~((ir.sepal_width<(c1-1.5*iq))|(ir.sepal_width>(c3+1.5*iq)))]
delid
```
![output](./Outputs/I6.png)

```
sns.boxplot(x='sepal_width',data=delid)
```
![output](./Outputs/I7.png)

## Z - Score
```
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import scipy.stats as stats
dataset=pd.read_csv("heights.csv")
dataset
```
![output](./Outputs/Z1.png)

```
df = pd.read_csv("heights.csv")
q1 = df['height'].quantile(0.25)
q2 = df['height'].quantile(0.5)
q3 = df['height'].quantile(0.75)
iqr = q3-q1
iqr
```
![output](./Outputs/Z2.png)

```
low = q1- 1.5*iqr
low
```
![output](./Outputs/Z3.png)

```
high = q3 + 1.5*iqr
high
```
![output](./Outputs/Z4.png)

```
df1 = df[((df['height'] >=low)& (df['height'] <=high))]
df1
```
![output](./Outputs/Z5.png)

```
z = np.abs(stats.zscore(df['height']))
z
```
![output](./Outputs/Z6.png)

```
df1 = df[z<3]
df1
```
![output](./Outputs/Z7.png)
# Result
Thus we have cleaned the data and removed the outliers by detection using IQR and Z-score method.
