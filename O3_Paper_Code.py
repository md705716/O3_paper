import lazypredict
from lazypredict.Supervised import LazyClassifier
from sklearn.model_selection import train_test_split
from lazypredict.Supervised import LazyRegressor
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
import numpy as np
import pandas as pd
O3 = pd.read_csv('All_cities_v3_v1.csv')
O3 = O3.iloc[: , 7:]
O3 = O3[['人口密度_人每平方公里_市辖区', '居民家庭用水量_万吨_市辖区','人均地区生产总值_元_市辖区','规模以上外商投资企业_个_市辖区','城市建设用地占市区面积比重_百分比_市辖区','全年公共汽电车客运总量_万人次_市辖区','年末实有城市道路面积_万平方米_市辖区','第三产业占地区生产总值的比重_市辖区','第三产业_卫生社会保障和社会福利业_人_市辖区','O3_mean']]
y = np.array(O3.O3_mean)
O3.drop(['O3_mean'],1,inplace = True)
X = O3.values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.10, random_state=6)
reg = LazyRegressor(verbose=0, ignore_warnings=False, custom_metric=None)
models, predictions = reg.fit(X_train, X_test, y_train, y_test)
print(models)
import xgboost as xgb
from sklearn.ensemble import HistGradientBoostingRegressor
import lightgbm as lgb
from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import BaggingRegressor
#XGBREGRESSOR
from sklearn.model_selection import GridSearchCV
parameters = {'learning_rate':[0.01,0.02,0.05,0.08,0.1,0.2,0.3,0.5,0.8,1]}
tree = xgb.XGBRegressor(random_state = 6)
clf = GridSearchCV(tree, parameters)
clf.fit(X_train,y_train)
clf.best_params_
tree = xgb.XGBRegressor(learning_rate = 0.2 ,random_state = 6)
tree.fit(X_train, y_train)
feature_importance_XGB = pd.DataFrame()
feature_importance_XGB['Feature'] = O3.columns
feature_importance_XGB['Importance'] = tree.feature_importances_
feature_importance_XGB['R2_Training'] = r2_score(y_train, tree.predict(X_train))
feature_importance_XGB['R2_Test'] = r2_score(y_test, tree.predict(X_test))
feature_importance_XGB['RMSE_Training'] = mean_squared_error(y_train, tree.predict(X_train),squared = False)
feature_importance_XGB['RMSE_Test'] = mean_squared_error(y_test, tree.predict(X_test),squared = False)
#HistGradientRegressor
parameters = {'max_iter':[10, 20, 30, 40, 50, 60 ,80 ,100, 200, 400], 'max_depth':[1,2,3,4,5,6,7,8,9,10,30,50,100],'learning_rate':[0.01,0.02,0.05,0.08,0.1,0.2,0.3,0.5,0.8,1]}
tree = HistGradientBoostingRegressor(random_state = 6)
clf = GridSearchCV(tree, parameters)
clf.fit(X_train,y_train)
clf.best_params_
tree = HistGradientBoostingRegressor(max_iter=400, max_depth=8,learning_rate=0.05,random_state = 6)
tree.fit(X_train, y_train)
from sklearn.inspection import permutation_importance
result = permutation_importance(tree, X_train, y_train, n_repeats=10, random_state=0, n_jobs=-1)
feature_importance_HGB = pd.DataFrame()
feature_importance_HGB['Feature'] = O3.columns
feature_importance_HGB['Importance'] = result.importances_mean
feature_importance_HGB['R2_Training'] = r2_score(y_train, tree.predict(X_train))
feature_importance_HGB['R2_Test'] = r2_score(y_test, tree.predict(X_test))
feature_importance_HGB['RMSE_Training'] = mean_squared_error(y_train, tree.predict(X_train),squared = False)
feature_importance_HGB['RMSE_Test'] = mean_squared_error(y_test, tree.predict(X_test),squared = False)
#EXTRATREESREGRESSOR
tree = ExtraTreesRegressor(random_state = 6)
parameters = {'n_estimators':[10, 20, 30, 40, 50, 60 ,80 ,100, 200, 400], 'max_depth':[1,2,3,4,5,6,7,8,9,10,30,50,100,200,500]}
clf = GridSearchCV(tree, parameters)
clf.fit(X_train,y_train)
clf.best_params_
tree = ExtraTreesRegressor(n_estimators = 400, max_depth = 30, random_state = 6)
tree.fit(X_train, y_train)
feature_importance_EXT = pd.DataFrame()
feature_importance_EXT['Feature'] = O3.columns
feature_importance_EXT['Importance'] = tree.feature_importances_
feature_importance_EXT['R2_Training'] = r2_score(y_train, tree.predict(X_train))
feature_importance_EXT['R2_Test'] = r2_score(y_test, tree.predict(X_test))
feature_importance_EXT['RMSE_Training'] = mean_squared_error(y_train, tree.predict(X_train),squared = False)
feature_importance_EXT['RMSE_Test'] = mean_squared_error(y_test, tree.predict(X_test),squared = False)
#RF
tree = RandomForestRegressor(random_state = 6)
parameters = {'n_estimators':[10, 20, 30, 40, 50, 60 ,80 ,100, 200, 400], 'max_depth':[1,2,3,4,5,6,7,8,9,10,30,50,100,200,500]}
clf = GridSearchCV(tree, parameters)
clf.fit(X_train,y_train)
clf.best_params_
tree = RandomForestRegressor(n_estimators = 400, max_depth = 30, random_state = 6)
tree.fit(X_train, y_train)
feature_importance_RF = pd.DataFrame()
feature_importance_RF['Feature'] = O3.columns
feature_importance_RF['Importance'] = tree.feature_importances_
feature_importance_RF['R2_Training'] = r2_score(y_train, tree.predict(X_train))
feature_importance_RF['R2_Test'] = r2_score(y_test, tree.predict(X_test))
feature_importance_RF['RMSE_Training'] = mean_squared_error(y_train, tree.predict(X_train),squared = False)
feature_importance_RF['RMSE_Test'] = mean_squared_error(y_test, tree.predict(X_test),squared = False)
#LGBMREGRESSOR
parameters = {'n_estimators':[10, 20, 30, 40, 50, 60 ,80 ,100, 200, 400], 'max_depth':[1,2,3,4,5,6,7,8,9,10,30,50,100],'learning_rate':[0.01,0.02,0.05,0.08,0.1,0.2,0.3,0.5,0.8,1]}
tree = LGBMRegressor(random_state = 6)
clf = GridSearchCV(tree, parameters)
clf.fit(X_train,y_train)
clf.best_params_
tree = LGBMRegressor(n_estimators = 200, max_depth = 30,learning_rate = 0.05, random_state = 6)
tree.fit(X_train, y_train)
feature_importance_LGB = pd.DataFrame()
feature_importance_LGB['Feature'] = O3.columns
feature_importance_LGB['Importance'] = tree.feature_importances_
feature_importance_LGB['R2_Training'] = r2_score(y_train, tree.predict(X_train))
feature_importance_LGB['R2_Test'] = r2_score(y_test, tree.predict(X_test))
feature_importance_LGB['RMSE_Training'] = mean_squared_error(y_train, tree.predict(X_train),squared = False)
feature_importance_LGB['RMSE_Test'] = mean_squared_error(y_test, tree.predict(X_test),squared = False)
#BAGGING
parameters = {'n_estimators':[10, 20, 30, 40, 50, 60 ,80 ,100, 200, 400]}
tree = BaggingRegressor(random_state = 6)
clf = GridSearchCV(tree, parameters)
clf.fit(X_train,y_train)
clf.best_params_
tree = BaggingRegressor(n_estimators = 400, random_state = 6)
tree.fit(X_train, y_train)
result = permutation_importance(tree, X_train, y_train, n_repeats=10, random_state=0, n_jobs=-1)
feature_importance_BAG = pd.DataFrame()
feature_importance_BAG['Feature'] = O3.columns
feature_importance_BAG['Importance'] = result.importances_mean
feature_importance_BAG['R2_Training'] = r2_score(y_train, tree.predict(X_train))
feature_importance_BAG['R2_Test'] = r2_score(y_test, tree.predict(X_test))
feature_importance_BAG['RMSE_Training'] = mean_squared_error(y_train, tree.predict(X_train),squared = False)
feature_importance_BAG['RMSE_Test'] = mean_squared_error(y_test, tree.predict(X_test),squared = False)
feature_importance_BAG.to_csv('Bagging.csv')
feature_importance_LGB.to_csv('LGBM.csv')
feature_importance_RF.to_csv('RF.csv')
feature_importance_EXT.to_csv('EXT.csv')
feature_importance_HGB.to_csv('HGB.csv')
feature_importance_XGB.to_csv('XGB.csv')
