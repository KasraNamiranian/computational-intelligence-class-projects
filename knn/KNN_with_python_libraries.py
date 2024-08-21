from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris

irisData = load_iris()
x = irisData.data
y = irisData.target


x_train, x_test, y_train, y_test = train_test_split(x , y, test_size=0.2, random_state=42)

k_value = 7          #Be aware that the value of K must be an odd number.
                    #It's important to select an appropriate value for k to avoid overfitting or underfitting the dataset.
                    # You can experiment with different values to find the best fit for your specific problem
knn = KNeighborsClassifier(n_neighbors = k_value)
knn.fit(x_train, y_train)


predictions = knn.predict(x_test)
print(predictions)