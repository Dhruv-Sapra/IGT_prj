#Data analysis of India population
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Prediction for a range of years
print("Give a Start Year")
start_year = int(input())
print("Give a End Year")
end_year = int(input())


#import dataset
df=pd.read_csv("C:\\Users\\thesp\\Desktop\\IGT PROJECT\\CSV\\Prediction\\India_population_2023.csv")
X=df.iloc[:, :-1]
y=df.iloc[:, 1]


#spliting dataset into two part trainingset & test set
from sklearn.model_selection import train_test_split
X_train,X_test,y_train ,y_test =train_test_split(X, y, test_size=0.1 ,random_state=0)

#train the model using linear regression
from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(X_train,y_train)


#predict the testset result
y_pred=regressor.predict(X_test)

# Custom year for prediction
# custom_year = 2020
# custom_population = regressor.predict([[custom_year]])

# print("Predicted population for year", custom_year, ":", custom_population[0])



# Create an empty DataFrame to store the results
results_df = pd.DataFrame(columns=['Year', 'Population'])

# Predict population for each year and store in the DataFrame
for year in range(start_year, end_year + 1):
    predicted_population = regressor.predict([[year]])
    results_df = pd.concat([results_df, pd.DataFrame({'Year': [year], 'Population': predicted_population[0]})], ignore_index=True)
# Save the results DataFrame to a CSV file
print(results_df)
results_df.to_csv("C:\\Users\\thesp\\Desktop\\IGT PROJECT\\CSV\\Prediction\\India_pop_prediction.csv", index=False)