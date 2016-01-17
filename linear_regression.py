import numpy as numpy
from  matplotlib import pyplot
from sklearn import datasets, linear_model

# Load diabetes dataset
diabetes = datasets.load_diabetes()

# Use a single feature
data   = diabetes.data[:, numpy.newaxis, 2]
target = diabetes.target

# Split data into training and testing sets
data_training   = data[:-20]
data_testing    = data[-20:]
target_training = target[:-20]
target_testing  = target[-20:]

# Create linear regression object and train the model
regression = linear_model.LinearRegression()
regression.fit(data_training, target_training)

# Print Stats
print('Coefficients: \n', regression.coef_)
print("Residual sum of squares: %.2f"
      % numpy.mean((regression.predict(data_testing) - target_testing) ** 2))
print('Variance score: %.2f' % regression.score(data_testing, target_testing))

# Plot Outputs
pyplot.scatter(data_testing, target_testing, color='black')
pyplot.plot(data_testing, regression.predict(data_testing), color='blue', linewidth=3)
pyplot.xticks(())
pyplot.yticks(())
pyplot.show()
