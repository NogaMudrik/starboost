from sklearn import datasets
from sklearn import metrics
from sklearn import model_selection
from sklearn import tree
from os.path import dirname, abspath
import os 


d = dirname(dirname(abspath(__file__)))
#os.chdir(d+'\starboost_up')
os.chdir(d)
import starboost_up
X, y = datasets.load_breast_cancer(return_X_y=True)

X_fit, X_val, y_fit, y_val = model_selection.train_test_split(X, y, test_size=0.2, random_state=42)

def micro_f1_score(y_true, y_pred):
    return metrics.f1_score(y_true, y_pred, average='micro')

model = starboost_up.boosting.BoostingClassifier(
    loss=starboost_up.losses.LogLoss(),
    base_estimator=tree.DecisionTreeRegressor(max_depth=3),
    # #
        base_estimator_is_tree=True,
        n_estimators=30,
        init_estimator=starboost_up.init.LogOddsEstimator(), #PriorProbabilityEstimator(),
        learning_rate=0.1,
        row_sampling=0.8,
        col_sampling=0.8,
        eval_metric=micro_f1_score,
        early_stopping_rounds=5,
        random_state=42
)
#
model = model.fit(X_fit, y_fit, eval_set=(X_val, y_val))
#
y_pred = model.predict(X_val)#_proba
#
print(metrics.roc_auc_score(y_val, y_pred))
