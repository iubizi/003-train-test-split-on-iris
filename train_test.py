####################
# load iris
####################

from sklearn.datasets import load_iris
iris = load_iris()

x = iris.data
y = iris.target

print('x.shape =', x.shape)
print('y.shape =', y.shape)
print()

# x.shape = (150, 4)
# y.shape = (150,)

####################
# train test split
####################

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.33, random_state=42, stratify=y)

print('x_train.shape =', x_train.shape)
print('y_train.shape =', y_train.shape)

print('x_test.shape =', x_test.shape)
print('y_test.shape =', y_test.shape)

'''
x_train.shape = (100, 4)
y_train.shape = (100,)
x_test.shape = (50, 4)
y_test.shape = (50,)
'''
