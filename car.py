# 1. Importing Required Libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn import metrics

# 2. Data Loading & Exploration
car_data = pd.read_csv('cardata.csv')

print(car_data.head())
print(car_data.info())
print(car_data.isnull().sum())
print(car_data.describe())

# Checking categorical column distributions
print(car_data['Fuel_Type'].value_counts())
print(car_data['Seller_Type'].value_counts())
print(car_data['Transmission'].value_counts())

# 3. Data Visualization
plt.figure(figsize=(15,5))
plt.suptitle('Selling Price vs Categorical Variables')

plt.subplot(1,3,1)
sns.barplot(x='Fuel_Type', y='Selling_Price', data=car_data)
plt.subplot(1,3,2)
sns.barplot(x='Seller_Type', y='Selling_Price', data=car_data)
plt.subplot(1,3,3)
sns.barplot(x='Transmission', y='Selling_Price', data=car_data)

plt.show()

# Heatmap
plt.figure(figsize=(10,7))
sns.heatmap(car_data.corr(numeric_only=True), annot=True, cmap="coolwarm")
plt.title("Correlation between columns")
plt.show()

# Scatterplot
plt.figure(figsize=(7,5))
sns.regplot(x='Present_Price', y='Selling_Price', data=car_data)
plt.title('Present Price vs Selling Price')
plt.show()

# 4. Data Preprocessing
car_data.replace({'Fuel_Type': {'Petrol':0, 'Diesel':1, 'CNG':2}}, inplace=True)
car_data = pd.get_dummies(car_data, columns=['Seller_Type','Transmission'], drop_first=True)

# 5. Features & Target
X = car_data.drop(['Car_Name','Selling_Price'], axis=1)
y = car_data['Selling_Price']

# 6. Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 7. Feature Scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 8. Model Training
model = LinearRegression()
model.fit(X_train, y_train)

# Predictions
pred = model.predict(X_test)

# 9. Model Evaluation
print("MAE: ", metrics.mean_absolute_error(y_test, pred))
print("MSE: ", metrics.mean_squared_error(y_test, pred))
print("R2 score: ", metrics.r2_score(y_test, pred))

# Visualization of Results
plt.figure(figsize=(6,6))
sns.regplot(x=pred, y=y_test, line_kws={"color":"red"})
plt.xlabel("Predicted Price")
plt.ylabel("Actual Price")
plt.title("Actual vs Predicted Selling Price")
plt.show()
