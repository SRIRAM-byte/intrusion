#import Libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn import tree
#import dataset
Dataset=pd.read_csv('kdd.csv')
print(Dataset.describe())

x=Dataset.iloc[:,:-1].values
x1=pd.DataFrame(x)
y=Dataset.iloc[:,10].values
y1=pd.DataFrame(y)

for i in range(800):
    if y[i]=='anom':
        y[i]=0
    else:
        y[i]=1
type(y)
type(x)
y=y.astype('int')
    
#Data Preprocessing
#Encoding categorical data
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
labelencoder_x=LabelEncoder()
for i in range(10) :
    x[:,i]=labelencoder_x.fit_transform(x[:,i])
    Y=pd.DataFrame(x[:,i])
for i in range(10) :
    onehotencoder=OneHotEncoder(categorical_features=[i])
    x=onehotencoder.fit_transform(x).toarray()
 #write in file
np.savetxt('encode_valuse.txt',x)
#Missing Data Removal
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values='NaN', strategy='most_frequent', axis=0)
imputer = imputer.fit(x[:,:])
x[:,:]=imputer.transform(x[:,:])
Missing_Data_Removed=imputer.transform(x[:,:])
#write in file
np.savetxt('Missing_values.txt',Missing_Data_Removed)
#train and test
from sklearn.model_selection import train_test_split  
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.20, random_state = 0) 
 
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
#SVM Apply
from sklearn.svm import SVC  
svclassifier = SVC(kernel='linear')  
svclassifier.fit(X_train, y_train)    
#prediction
y_pred = svclassifier.predict(X_test)  
from sklearn.metrics import classification_report, confusion_matrix  
print(confusion_matrix(y_test, y_pred))  
print(classification_report(y_test, y_pred)) 
dt = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})  
#performance anaylsis
import warnings
warnings.filterwarnings('ignore')
score = accuracy_score(y_test, y_pred)
from sklearn import metrics 
print("Accuracy:",accuracy_score(y_test,y_pred)*100)




#Decisison tree
import warnings
warnings.filterwarnings('ignore','DeprecationWarning')
import pdb
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn import tree
#import dataset
Dataset=pd.read_csv('kdd.csv')
print(Dataset.describe())
x=Dataset.iloc[:,:-1].values
x1=pd.DataFrame(x)
y=Dataset.iloc[:,41].values
k1=pd.DataFrame(y)
for i in range(999):
    if y[i]=='normal':
        y[i]=0
    else:
        y[i]=1
type(y)
type(x)
y=y.astype('int')
#Data Preprocessing
#Missing Data Removal
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values='NaN', strategy='mean', axis=0)
imputer = imputer.fit(x[:,4:41])
x[:,4:41]=imputer.transform(x[:,4:41])
Missing_Data_Removed=imputer.transform(x[:,4:41])
#Encoding categorical data
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
labelencoder_x=LabelEncoder()
x[:,1]=labelencoder_x.fit_transform(x[:,1])
Y=pd.DataFrame(x[:,1])
labelencoder_x=LabelEncoder()
x[:,2]=labelencoder_x.fit_transform(x[:,2])
Y=pd.DataFrame(x[:,2])
labelencoder_x=LabelEncoder()
x[:,3]=labelencoder_x.fit_transform(x[:,3])
Y=pd.DataFrame(x[:,3])
onehotencoder=OneHotEncoder(categorical_features=[1])
x=onehotencoder.fit_transform(x).toarray()
onehotencoder=OneHotEncoder(categorical_features=[2])
x=onehotencoder.fit_transform(x).toarray()
onehotencoder=OneHotEncoder(categorical_features=[3])
x=onehotencoder.fit_transform(x).toarray()
# Splitting the dataset into the Training set and Test set for linear
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.25, random_state = 0)
clf_gini = DecisionTreeClassifier(criterion = "gini", random_state = 100,
                               max_depth=3, min_samples_leaf=5)
clf_gini.fit(X_train, y_train)
DecisionTreeClassifier(class_weight=None, criterion='gini', max_depth=3,
            max_features=None, max_leaf_nodes=None, min_samples_leaf=5,
            min_samples_split=2, min_weight_fraction_leaf=0.0,
            presort=False, random_state=100, splitter='best')
clf_entropy = DecisionTreeClassifier(criterion = "entropy", random_state = 100,
 max_depth=3, min_samples_leaf=5)
clf_entropy.fit(X_train, y_train)
y_pred = clf_gini.predict(X_test)
print(y_pred)
print ("Accuracy is ", accuracy_score(y_test,y_pred)*100)







#naivebayes algorithm
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 
#import Libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
#import dataset
Dataset=pd.read_csv('kdd.csv')
print(Dataset.describe())
x=Dataset.iloc[:,:-1].values
x1=pd.DataFrame(x)
y=Dataset.iloc[:,41].values
k1=pd.DataFrame(y)
for i in range(999):
    if y[i]=='normal':
        y[i]=0
    else:
        y[i]=1
type(y)
type(x)
y=y.astype('int')
#Data Preprocessing
#Missing Data Removal
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values='NaN', strategy='mean', axis=0)
imputer = imputer.fit(x[:,4:41])
x[:,4:41]=imputer.transform(x[:,4:41])
Missing_Data_Removed=imputer.transform(x[:,4:41])
#write in file
np.savetxt('Missing_values.txt',Missing_Data_Removed)
#Encoding categorical data
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
labelencoder_x=LabelEncoder()
x[:,1]=labelencoder_x.fit_transform(x[:,1])
Y=pd.DataFrame(x[:,1])
labelencoder_x=LabelEncoder()
x[:,2]=labelencoder_x.fit_transform(x[:,2])
Y=pd.DataFrame(x[:,2])
labelencoder_x=LabelEncoder()
x[:,3]=labelencoder_x.fit_transform(x[:,3])
Y=pd.DataFrame(x[:,3])
onehotencoder=OneHotEncoder(categorical_features=[1])
x=onehotencoder.fit_transform(x).toarray()
onehotencoder=OneHotEncoder(categorical_features=[2])
x=onehotencoder.fit_transform(x).toarray()
onehotencoder=OneHotEncoder(categorical_features=[3])
x=onehotencoder.fit_transform(x).toarray()
#write in file
np.savetxt('encode_valuse.txt',x)
# Splitting the dataset into the Training set and Test set for linear
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.25, random_state = 0)
# Fitting Naive Bayes to the Training set
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train, y_train)
# Predicting the Test set results
y_pred = classifier.predict(X_test)
dt = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})  
#from sklearn.metrics import accuracy_score
from sklearn.metrics import accuracy_score
score = accuracy_score(y_test, y_pred)
print("Accuracy:",accuracy_score(y_test,y_pred)*100)


#graph generation
left = [54, 75, 90] 
height = [79,90,96.3] 
plt.bar(left, height, width = 10, color = ['skyblue']) 
plt.xlabel('Existing') 
plt.ylabel('Proposed') 
plt.title('Comparison Graph') 
plt.show() 
