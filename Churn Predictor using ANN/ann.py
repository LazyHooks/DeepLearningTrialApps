#ANN

#data preprocessing 

# Import0
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Import dataset
dataset = pd.read_csv('Churn_Modelling.csv')
X = dataset.iloc[:, 3:13].values
y = dataset.iloc[:, 13].values

# Encoding the Independent Variable
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X_1= LabelEncoder()
X[:, 1] = labelencoder_X_1.fit_transform(X[:, 1])
labelencoder_X_2= LabelEncoder()
X[:, 2] = labelencoder_X_2.fit_transform(X[:, 2])
onehotencoder = OneHotEncoder(categorical_features = [1])
X = onehotencoder.fit_transform(X).toarray()
X=X[:, 1:]

# Split dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


"""
# for linear training session
#import1
import keras
from keras.models import Sequential 
from keras.layers import Dense

#initialise ANN
classifier = Sequential()

#layers
classifier.add(Dense(units=6, kernel_initializer='uniform', activation ='relu',input_dim=11))
classifier.add(Dense(units=6, kernel_initializer='uniform', activation ='relu'))

#output layer
classifier.add(Dense(units=1, kernel_initializer='uniform', activation ='sigmoid'))

#compile
classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

#train
classifier.fit(X_train, y_train, batch_size=10, nb_epoch=100)

#linear train end
"""



"""
#for k-fold cv training

#k-fold cross validation accuracy 
#import2
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from keras.models import Sequential 
from keras.layers import Dense
from keras.layers import Dropout
#build
def build_classifier():
    #initialise ANN
    classifier = Sequential()

    #layers w/ dropout
    classifier.add(Dense(units=6, kernel_initializer='uniform', activation ='relu',input_dim=11))
    classifier.add(Dropour(p=0.1))
    classifier.add(Dense(units=6, kernel_initializer='uniform', activation ='relu'))
    classifier.add(Dropour(p=0.1))

    #output layer
    classifier.add(Dense(units=1, kernel_initializer='uniform', activation ='sigmoid'))

    #compile
    classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return classifier

#init
classifier = KerasClassifier(build_fn = build_classifier, batch_size=10, epochs= 100 )



accuracies = cross_val_score(estimator = classifier, X= X_train, y= y_train, cv= 10, n_jobs = -1)
#variance
mean = accuracies.mean()
variance = accuracies.std()

#k-fold cv train end
"""

# predict
y_pred = classifier.predict(X_test)
y_pred = (y_pred>0.5)

# Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

"""
#for one new prediction
new_prediction = classifier.predict(sc.transform(np.array([[0.0,0,600,1,40,3,60000,2,1,1,50000]])))
new_prediction = (new_prediction>0.5)
"""


#for grid search cv training

#tuning [parameters]
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV
from keras.models import Sequential 
from keras.layers import Dense
from keras.layers import Dropout
#build
def build_classifier(optimizer):
    #initialise ANN
    classifier = Sequential()

    #layers w/ dropout
    classifier.add(Dense(units=6, kernel_initializer='uniform', activation ='relu',input_dim=11))
    classifier.add(Dropout(p=0.1))
    classifier.add(Dense(units=6, kernel_initializer='uniform', activation ='relu'))
    classifier.add(Dropout(p=0.1))

    #output layer
    classifier.add(Dense(units=1, kernel_initializer='uniform', activation ='sigmoid'))

    #compile
    classifier.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    return classifier

#init
classifier = KerasClassifier(build_fn = build_classifier)

#gridSearch
parameters = {'batch_size':[25,32], 'epochs':[100, 500],'optimizer':['adam', 'rmsprop']}
grid_search = GridSearchCV(estimator=classifier, param_grid=parameters, scoring = 'accuracy', cv=10)

#fit
grid_search=grid_search.fit(X_train, y_train)
best_parameters = grid_search.best_params_ 
best_accuracy =grid_search.best_score_

#gridSearchCV training end









