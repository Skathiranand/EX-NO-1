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
df=pd.read_csv("/SAMPLEIDS.csv")
df
```
![alt text](<Screenshot 2024-08-17 105530.png>)
![alt text](<Screenshot 2024-08-17 105639.png>)

```
import pandas as pd
df=pd.read_csv("/SAMPLEIDS.csv")
df.isnull().sum()
```
![alt text](<Screenshot 2024-08-17 105746.png>)

```
import pandas as pd
df=pd.read_csv("/SAMPLEIDS.csv")
df.isnull().any()
```
![alt text](<Screenshot 2024-08-17 105759.png>)

```
import pandas as pd
df=pd.read_csv("/SAMPLEIDS.csv")
df.dropna()
```
![alt text](<Screenshot 2024-08-17 105822.png>)

```
import pandas as pd
df=pd.read_csv("/SAMPLEIDS.csv")
df.fillna(0)
```
![alt text](<Screenshot 2024-08-17 105842.png>)
![alt text](<Screenshot 2024-08-17 105849.png>)

```
import pandas as pd
df=pd.read_csv("/SAMPLEIDS.csv")
df.fillna(method = 'ffill')
```
![alt text](<Screenshot 2024-08-17 105901.png>)
![alt text](<Screenshot 2024-08-17 105908.png>)

```
import pandas as pd
df=pd.read_csv("/SAMPLEIDS.csv")
df.fillna(method = 'bfill')
```
![alt text](<Screenshot 2024-08-17 105921.png>)
![alt text](<Screenshot 2024-08-17 105928.png>)

```
import pandas as pd
df=pd.read_csv("/SAMPLEIDS.csv")
df_dropped = df.dropna()
df_dropped
```
![alt text](<Screenshot 2024-08-17 110037.png>)

```
import pandas as pd
df=pd.read_csv("/SAMPLEIDS.csv")
df.fillna({'GENDER':'MALE','NAME':'SRI','ADDRESS':'POONAMALEE','M1':98,'M2':87,'M3':76,'M4':92,'TOTAL':305,'AVG':89.999999})
```
![alt text](<Screenshot 2024-08-17 110119.png>)
![alt text](<Screenshot 2024-08-17 110126.png>)

## IQR(Inter Quartile Range)
```
import pandas as pd
ir=pd.read_csv("/iris.csv")
ir
```
![alt text](<Screenshot 2024-08-17 111512.png>)
```
import pandas as pd
ir=pd.read_csv("/iris.csv")
ir.describe()
```
![alt text](<Screenshot 2024-08-17 111522.png>)
```
import pandas as pd
ir=pd.read_csv("/iris.csv")
c1=ir.sepal_width.quantile(0.25)
c3=ir.sepal_width.quantile(0.75)
iq=c3-c1
print(c3)
```
![alt text](<Screenshot 2024-08-17 111530.png>)
```
import pandas as pd
ir=pd.read_csv("/iris.csv")
rid=ir[((ir.sepal_width<(c1-1.5*iq))|(ir.sepal_width>(c3+1.5*iq)))]
rid['sepal_width']
```
![alt text](<Screenshot 2024-08-17 111537.png>)
```
import seaborn as sns
import matplotlib.pyplot as plt
sns.boxplot(x='sepal_width',data=ir)
plt.show()
```
![alt text](<Screenshot 2024-08-17 111548.png>)
```
import pandas as pd
ir=pd.read_csv("/iris.csv")
delid=ir[~((ir.sepal_width<(c1-1.5*iq))|(ir.sepal_width>(c3+1.5*iq)))]
delid
```
![alt text](<Screenshot 2024-08-17 111556.png>)
```
import seaborn as sns
import matplotlib.pyplot as plt
sns.boxplot(x='sepal_width',data=delid)
plt.show()
```
![alt text](<Screenshot 2024-08-17 111617.png>)

## Z-Score
```
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import scipy.stats as stats
dataset=pd.read_csv("/heights.csv")
dataset
```
![alt text](<Screenshot 2024-08-17 112133.png>)
```
df = pd.read_csv("/heights.csv")
q1 = df['height'].quantile(0.25)
q2 = df['height'].quantile(0.5)
q3 = df['height'].quantile(0.75)
iqr = q3-q1
iqr
```
![alt text](<Screenshot 2024-08-17 112142.png>)
```
low = q1 - 1.5*iqr
low
```
![alt text](<Screenshot 2024-08-17 112208.png>)
```
high = q3 + 1.5*iqr
high
```
![alt text](<Screenshot 2024-08-17 112219.png>)
```
df1 = df[((df['height'] >=low)& (df['height'] <=high))]
df1
```
![alt text](<Screenshot 2024-08-17 112227.png>)
```
z = np.abs(stats.zscore(df['height']))
z
```
![alt text](<Screenshot 2024-08-17 112238.png>)
```
df1 = df[z<3]
df1
```
![alt text](<Screenshot 2024-08-17 112249.png>)

# Result
Thus we have cleaned the data and removed the outliers by detection using IQR and Z-score method.
