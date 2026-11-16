#STEP 1:
import libraries
import joblib
from sklearn.dataset import load_breast-cancer
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandrdScaler
from sklearn.liner_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensembel import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import(
  f1_score,
  accuracy_score,
  classification_report,
  confusion_matrix
)
#STEP 2:Load dataset
data=load_breast_cancer()
X=pd.DataFrame(
  data.data,
  columns=data.featurn_names
)
y=pd.Series(data.target,name="target")
print("Dataset shape:",X.shape)
print("\nFirst 5 rows:")
print(X.head())
print("\nTarget values:")
print(y.value_counts())
#STEP 3:Split dataset
X_train,X_test,y_train,y_test=train_test_split(
  X,
  y,
  test_size=0.20,
  random_state=42,
  stratify=y
)
print("\nTraining samples:",X_train.shape[0])
print("\Testing sample:",X_test.shape[0])
#STEP 4:Define classification algorithms
models={
  "Logistic Regression":LogisticRegression(
    max_iter=5000
  ),
  "KNN",KNeighborsClassifier(),
  "Decision Tree":DecisionTreeClassifier(
     random_state=42
  ),
  "Random Forest":RandomForestClassifier(
    random_stae=42
  ),
  "SVM":SVC()
}
#STEP 5:Define hyperparameter grids
param_grid={
  "Logistic Regression":{
    "model__C":[0.1,1,10]
    "model__solver":["liblinear"]
  },
  "KNN":{
    "model__n_neighbores":[3,5,7,9],
    "model__weights":["uniform","distance"]
