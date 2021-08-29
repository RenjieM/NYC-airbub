import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import seaborn as sns


airbnb = pd.read_csv('D:\Download bin\AB_NYC_2019.csv\AB_NYC_2019.csv')
airbnb.duplicated().sum()
airbnb.drop_duplicates(inplace=True)
airbnb.isnull().sum()
airbnb.drop(['id', 'name', 'host_name', 'last_review'], axis=1, inplace=True)
airbnb.fillna({'reviews_per_month': 0}, inplace=True)
airbnb.reviews_per_month.isnull().sum()
airbnb.dropna(axis=0, how='any', inplace=True)

airbnb.describe()
airbnb.columns

corr = airbnb.corr(method='kendall')
plt.figure(figsize=(15,8))
plt.title('corr')
sns.heatmap(corr, annot=True)


airbnb.head(15)

airbnb['neighbourhood_group'].unique()

plt.figure(figsize=(10,10))
sns.countplot(airbnb['neighbourhood_group'], palette='plasma')
plt.title('Neighborhood Group')


plt.figure(figsize=(10,10))
sns.countplot(airbnb['neighbourhood'], palette='plasma')
plt.title('Neighbourhood')


plt.figure(figsize=(10,10))
sns.countplot(airbnb['room_type'], palette='plasma')
plt.title('Restaurants delivering online or not')

plt.figure(figsize=(10,10))
ax = sns.boxplot(data=airbnb, x='neighbourhood_group', y='availability_365', palette='plasma')
plt.show()

plt.figure(figsize=(10,6))
sns.scatterplot(airbnb.longitude, airbnb.latitude, hue=airbnb.neighbourhood_group)


plt.figure(figsize=(10,6))
sns.scatterplot(airbnb.longitude, airbnb.latitude, hue=airbnb.neighbourhood)


plt.figure(figsize=(10,6))
sns.scatterplot(airbnb.longitude, airbnb.latitude, hue=airbnb.room_type)


plt.figure(figsize=(10, 6))
sns.scatterplot(airbnb.longitude, airbnb.latitude, hue=airbnb.availability_365)

from wordcloud import WordCloud

plt.subplots(figsize=(25,15))
wordcloud = WordCloud(background_color='white', width=1920, height=1080).generate(' '.join(airbnb.neighbourhood))
plt.imshow(wordcloud)
plt.axis('off')
plt.savefig('neighbourhood.png')

airbnb.drop(['host_id','latitude','longitude','neighbourhood','number_of_reviews','reviews_per_month'], axis=1, inplace=True)

def Encode(airbnb):
    for column in airbnb.columns[airbnb.columns.isin(['neighbourhood_group','room_type'])]:
        airbnb[column] = airbnb[column].factorize()[0]
    return airbnb

airbnb_en = Encode(airbnb.copy())

airbnb_en.head(15)

coor = airbnb_en.corr(method='kendall')
plt.figure(figsize=(18,12))
sns.heatmap(coor, annot=True)


airbnb_en.columns


from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import r2_score

x = airbnb_en.iloc[:,[0, 1, 3, 4, 5]]
y = airbnb_en.iloc[:,[2]]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=.1, random_state=353)
x_train.head()
y_train.head()

y_train.shape
x_test.shape

reg = LinearRegression()
reg.fit(x_train, y_train)
y_pred = reg.predict(x_test)
print(r2_score(y_test, y_pred))

print(reg.coef_)
print(reg.intercept_)

from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import StandardScaler

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=.1, random_state=105)
DTree = DecisionTreeRegressor(min_samples_leaf=.0001)
DTree.fit(x_train, y_train)
y_predict = DTree.predict(x_test)
print(r2_score(y_test, y_predict))



