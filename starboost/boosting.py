import abc
import collections
import copy
import itertools

import numpy as np
from scipy.special import expit as sigmoid
from sklearn import base
from sklearn import ensemble
from sklearn import preprocessing
from sklearn import tree
from sklearn import utils

from . import losses


__all__ = ['BoostingClassifier', 'BoostingRegressor']


class BaseBoosting(abc.ABC, ensemble.BaseEnsemble):
    """Implements logic common to all other boosting classes."""

    def __init__(self, loss=None, base_estimator=None, base_estimator_is_tree=False,
                 n_estimators=30, init_estimator=None, line_searcher=None, learning_rate=0.1,
                 row_sampling=1.0, col_sampling=1.0, eval_metric=None, early_stopping_rounds=None,
                 random_state=None, is_DART = False, DART_params = {}):
        self.loss = loss
        self.base_estimator = base_estimator
        self.base_estimator_is_tree = base_estimator_is_tree
        self.n_estimators = n_estimators
        self.init_estimator = init_estimator
        self.line_searcher = line_searcher
        self.learning_rate = learning_rate
        self.row_sampling = row_sampling
        self.col_sampling = col_sampling
        self.eval_metric = eval_metric
        self.early_stopping_rounds = early_stopping_rounds
        self.random_state = random_state
        self.is_DART = is_DART
        # dist_drop: can be random or by weights (inverse).
        # min_1: whether to drpo out at least one estimator
        default_DART = {'n_drop':1, 'dist_drop': 'random' , 'min_1':True, 'weights_list' : None}
        
        additional_keys = {key:val for key,val in default_DART.items() if key not in DART_params.keys()}
        DART_params = {**DART_params, **additional_keys}
        if DART_params['dist_drop'] == 'weights'  and not DART_params['weights_list'] :
            DART_params['dist_drop'] = 'random'
        self.DART_params = DART_params
        if self.is_DART:
            # drop data dictionary.
            # keys = iteration #; values = dropout trees 
            self.drop_data = {}
            # addition data - information about the tree added in each iteration.
            # keys = iteration #; values = the tree added  
            self.add_data = {}
            # store trees id and results .
            # inter_results[0] - a binary matrix of trees used; 
            # inter_results[1] 

            self.inter_results = {'estimators': {},'estimators_dropped': {}, 'gradients': {}, 'preds':{},'mutiply_factor' : np.ones((n_estimators, y.shape[1]))}
            
    @abc.abstractmethod
    def _transform_y_pred(self, y_pred):
        pass

    @property
    @abc.abstractmethod
    def _default_loss(self):
        pass

    def fit(self, X, y, eval_set=None):
        """Fit a gradient boosting procedure to a dataset.

        Args:
            X (array-like or sparse matrix of shape (n_samples, n_features)): The training input
                samples. Sparse matrices are accepted only if they are supported by the weak model.
            y (array-like of shape (n_samples,)): The training target values (strings or integers
                in classification, real numbers in regression).
            eval_set (tuple of length 2, optional, default=None): The evaluation set is a tuple
                ``(X_val, y_val)``. It has to respect the same conventions as ``X`` and ``y``.
        Returns:
            self
        """
        # Verify the input parameters
        base_estimator = self.base_estimator
        base_estimator_is_tree = self.base_estimator_is_tree
        if base_estimator is None:
            base_estimator = tree.DecisionTreeRegressor(max_depth=1, random_state=self.random_state)
            base_estimator_is_tree = True

        if not isinstance(base_estimator, base.RegressorMixin):
            raise ValueError('base_estimator must be a RegressorMixin')

        loss = self.loss or self._default_loss

        self.init_estimator_ = base.clone(self.init_estimator or loss.default_init_estimator)

        eval_metric = self.eval_metric or loss

        line_searcher = self.line_searcher
        if line_searcher is None and base_estimator_is_tree:
            line_searcher = loss.tree_line_searcher

        self._rng = utils.check_random_state(self.random_state)

        # At this point we assume the input data has been checked
        if y.ndim == 1:
            y = y.reshape(-1, 1)

        # Instantiate some training variables
        self.estimators_ = []
        self.line_searchers_ = []
        self.columns_ = []
        self.eval_scores_ = [] if eval_set else None

        # If col_sampling is lower than 1 then we only use a subset of the features
        cols = None
        if self.col_sampling < 1:
            n_cols = int(X.shape[1] * self.col_sampling)
            cols = self._rng.choice(X.shape[1], n_cols, replace=False)
        # Use init_estimator for the first fit
        if cols is None:
             self.init_estimator_.fit(X,y)
        else:
            self.init_estimator_.fit(X[:,cols], y)
        y_pred = self.init_estimator_.predict((X if cols is None else X[:, cols]))

        # We keep training weak learners until we reach n_estimators or early stopping occurs
        for esti_num in range(self.n_estimators):
            np.random.seed(esti_num)
            # Create an empty list to store drop-out information
            
            if self.is_DART:
                if self.DART_params['dist_drop'] == 'random':
                    if self.DART_params['n_drop'] < 1:
                        abs_drop = int(floor(len(self.estimators_)*self.DART_params['n_drop']))
                    else:
                        abs_drop = np.min([self.DART_params['n_drop'], len(self.estimators_)])
                    index_drop = np.random.choice(len(self.estimators_), abs_drop)
                    trees_drop = np.zeros((1,esti_num))
                    trees_drop[index_drop] = 1
                elif self.DART_params['dist_drop'] == 'weights':
                    w_list = self.DART_params['weights_list']
                    w_list = w_list / np.max(w_list)
                    random_init = np.random.rand(np.shape(w_list))
                    trees_drop = random_init >= w_list # find the trees to drop. Higher weight -> lower chance to be dropped
                    if  trees_drop.all(): # if all trees shoud be dropped
                        trees_drop[-1] = 0
                        
                    if self.DART_params['min_1'] and not trees_drop.any(): 
                        index_drop = np.random.choice( len(trees_drop), 1, False)
                        trees_drop[index_drop] = 1
                    else:
                        index_drop = np.where(trees_drop)[0]  
                    trees_drop = 1*trees_drop                                  
                else:   raise NameError('Unknown dropping type')
                self.drop_data[esti_num] = self.estimarors_[index_drop]
                # Record the trees used in the .DART_params['estimators']  matrix
                
                factor_drop = 1/(np.sum(trees_drop)+1)
                factor_return = np.sum(trees_drop)/(np.sum(trees_drop)+1)
            else:
                
                factor_drop = 1
                factor_return = 1
            # If row_sampling is lower than 1 then we're doing stochastic gradient descent
            rows = None
            if self.row_sampling < 1:
                n_rows = int(X.shape[0] * self.row_sampling)
                rows = self._rng.choice(X.shape[0], n_rows, replace=False)

            # Subset X
            X_fit = X
            if rows is not None:
                X_fit = X_fit[rows, :]
            if cols is not None:
                X_fit = X_fit[:, cols]

            if not self.is_DART:
                # Compute the gradients of the loss for the current prediction
                gradients = loss.gradient(y, self._transform_y_pred(y_pred))
            else:
                # min_drop = np.min(index_drop)
                if esti_num > 0:
                    y_preds_some = np.zeros(y.shape)
                    for esti_num, former_fitted_esti in enumerate(self.estimators_):
                        if esti_num not in trees_drop:
                            for i in former_fitted_esti_i:
                                direction = former_fitted_esti_i.predict(X if cols is None else X[:, cols])
                                multi_factor = self.inter_results['mutiply_factor'][esti_num,i]
                                y_preds_some[:, i] += self.learning_rate * direction * multi_factor
                        else:
                            for i in former_fitted_esti_i:
                                self.inter_results['mutiply_factor'][esti_num,i]*=factor_return
                    #                    store_est = self.DART_params['estimators'][:esti_num, :esti_num]
                    #                    store_est[:, trees_drop] = 1 - store_est[:, trees_drop] 
                    #                    store_est[np.isnan(store_est)] = 0
                    #                    for row_num, row_est in enumerate(store_est):
                    #                        if (row_est==0).any():
                    #                            lower_0 = np.where(row_est == 0 )[0][0]
                    #                            row_est[lower_0:] = 0
                    #                            store_est[row_num] = row_est                
                    #                        
                    #                    sum_cons = store_est.sum(1)
                    #                    argmax_sum = np.argmax(sum_cons)
                    #                    num_trees = sum_cons[argmax_sum]
                    #                    y_preds_some = self.DART_params['preds'][argmax_sum]
                    # loop for prediction until the added estimator (not including the last one)

                    # run over the trees found so far
                    #                    for comp_tree in range(argmax_sum , esti_num):
                    #                        if comp_tree not in index_drop:
                    #                            for  i in range(y_preds_some.shape[1]):
                    #                                # Estimate the descent direction using the weak learner
                    #                                former_estimator = self.estimators_[comp_tree,i]
                    #                                direction = former_estimator.predict(X if cols is None else X[:, cols])
                    #                                #gradients = loss.gradient(y, self._transform_y_pred(y_pred))
                    #                                y_preds_some[:, i] += self.learning_rate * direction
                    y_pred = y_preds_some
                #y_pred = self.init_estimator_.predict(X if cols is None else X[:, cols])
                gradients = loss.gradient(y, self._transform_y_pred(y_pred))
                                    
                                

            """
            We will train one weak model per column in y
            """
            estimators = [] # It is for specific estimator (for all cols of y)
            line_searchers = []
            # running over the different cols of y
            for i, gradient in enumerate(gradients.T):

                # Fit a weak learner to the negative gradient of the loss function
                estimator = base.clone(base_estimator)
                estimator = estimator.fit(X_fit, -gradient if rows is None else -gradient[rows])
                estimators.append(estimator)

                # Estimate the descent direction using the weak learner
                direction = estimator.predict(X if cols is None else X[:, cols])

                # Apply line search if a line searcher has been provided
                if line_searcher:
                    ls = copy.copy(line_searcher)
                    ls = ls.fit(y[:, i], y_pred[:, i], gradient, direction)
                    line_searchers.append(ls)
                    direction = ls.update(direction)

                # Move the predictions along the estimated descent direction
                #if not self.is_DART: # if is_DARt -> update y_pred after droupout
                y_pred[:, i] += self.learning_rate * direction * factor_drop
                #else:
                #    set_take = self.inter_results['estimators']
                #    y_pred_prev = self.inter_results['preds'][:,i]
                    
                

            # Store the estimator and the step
            self.estimators_.append(estimators)
            if line_searchers:
                self.line_searchers_.append(line_searchers)
            self.columns_.append(cols)

            # We're now at the end of a round so we can evaluate the model on the validation set
            if not eval_set:
                continue
            X_val, y_val = eval_set
            self.eval_scores_.append(eval_metric(y_val, self.predict(X_val)))

            # Check for early stopping
            if self.early_stopping_rounds and len(self.eval_scores_) > self.early_stopping_rounds:
                if self.eval_scores_[-1] > self.eval_scores_[-(self.early_stopping_rounds + 1)]:
                    break

            """
            Store the last estimator in memory
            """
            if self.is_DART:
                self.inter_results['estimators'][esti_num] = [model.get_booster().get_dump() for model in estimators]
                self.inter_results['preds'][esti_num] = y_pred
                self.inter_results['gradients'][esti_num] = gradients
                self.add_data[esti_num] = estimators
                self.inter_results['estimators_dropped'][esti_num] = [[model.get_booster().get_dump() for model in estimators_spec] for estimators_spec in self.estimarors_[index_drop]]
        
        return self
model.get_booster().get_dump()
    def iter_predict(self, X, include_init=False):
        """Returns the predictions for ``X`` at every stage of the boosting procedure.

        Args:
            X (array-like or sparse matrix of shape (n_samples, n_features): The input samples.
                Sparse matrices are accepted only if they are supported by the weak model.
            include_init (bool, default=False): If ``True`` then the prediction from
                ``init_estimator`` will also be returned.
        Returns:
            iterator of arrays of shape (n_samples,) containing the predicted values at each stage
        """
        utils.validation.check_is_fitted(self, 'init_estimator_')
        X = utils.check_array(X, accept_sparse=['csr', 'csc'], dtype=None, force_all_finite=False)

        y_pred = self.init_estimator_.predict(X)

        # The user decides if the initial prediction should be included or not
        if include_init:
            yield y_pred

        for estimators, line_searchers, cols in itertools.zip_longest(self.estimators_,
                                                                      self.line_searchers_,
                                                                      self.columns_):

            for i, (estimator, line_searcher) in enumerate(itertools.zip_longest(estimators,
                                                                                 line_searchers or [])):

                # If we used column sampling then we have to make sure the columns of X are arranged
                # in the correct order
                if cols is None:
                    direction = estimator.predict(X)
                else:
                    direction = estimator.predict(X[:, cols])

                if line_searcher:
                    direction = line_searcher.update(direction)

                y_pred[:, i] += self.learning_rate * direction

            yield y_pred

    def predict(self, X):
        """Returns the predictions for ``X``.

        Under the hood this method simply goes through the outputs of ``iter_predict`` and returns
        the final one.

        Arguments:
            X (array-like or sparse matrix of shape (n_samples, n_features)): The input samples.
                Sparse matrices are accepted only if they are supported by the weak model.

        Returns:
            array of shape (n_samples,) containing the predicted values.
        """
        return collections.deque(self.iter_predict(X), maxlen=1).pop()


class BoostingRegressor(BaseBoosting, base.RegressorMixin):
    """Gradient boosting for regression.

    Arguments:
        loss (starboost.losses.Loss, default=starboost.loss.L2Loss)
            The loss function that will be optimized. At every stage a weak learner will be fit to
            the negative gradient of the loss. The provided value must be a class that at the very
            least implements a ``__call__`` method and a ``gradient`` method.
        base_estimator (sklearn.base.RegressorMixin, default=None): The weak learner. This must be
            a regression model. If `None` then a decision stump will be used.
        base_estimator_is_tree (bool, default=False): Indicates if the provided ``base_estimator``
            is a tree model or not. Various boosting optimizations specific to trees can be made to
            improve the overall performance.
        n_estimators (int, default=30): The maximum number of weak learners to train. The final
            number of trained weak learners will be lower than ``n_estimators`` if early stopping
            happens.
        init_estimator (sklearn.base.BaseEstimator, default=None): The estimator used to make the
            initial guess. If ``None`` then the ``init_estimator`` property from the ``loss`` will
            be used.
        line_searcher (starboost.line_searchers.LineSearcher, default=None): A line searcher which
            can be used to find the optimal step size during gradient descent. If you've set
            ``base_estimator_is_tree`` to ``True`` and are using one of StarBoost's losses then an optimal
            line searcher will be used, meaning you safely set this field to ``None``.
        learning_rate (float, default=0.1): The learning rate shrinks the contribution of each tree.
            Specifically the descent direction estimated by each weak learner will be multiplied by
            ``learning_rate``. There is a trade-off between learning_rate and ``n_estimators``.
        row_sampling (float, default=1.0): The ratio of rows to sample at each stage.
        col_sampling (float, default=1.0): The ratio of columns to sample at each stage.
        eval_metric (function, default=None): The evaluation metric used to check for early
            stopping. If ``None`` it will default to ``loss``.
        random_state (int, RandomState instance or None, default=None): If int, ``random_state`` is
            the seed used by the random number generator; if ``RandomState`` instance,
            ``random_state`` is the random number generator; if ``None``, the random number
            generator is the ``RandomState`` instance used by ``np.random``.
    """

    def _transform_y_pred(self, y_pred):
        return y_pred

    @property
    def _default_loss(self):
        return losses.L2Loss()

    def fit(self, X, y, eval_set=None):
        X, y = utils.check_X_y(X, y, ['csr', 'csc'], dtype=None, force_all_finite=False)
        return super().fit(X=X, y=y, eval_set=eval_set)

    def iter_predict(self, X, include_init=False):
        for y_pred in super().iter_predict(X, include_init=include_init):
            yield y_pred[:, 0]


def softmax(x):
    """Can be replaced once scipy 1.3 is released, although numeric stability should be checked."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=1)[:, None]


class BoostingClassifier(BaseBoosting, base.ClassifierMixin):
    """Gradient boosting for regression.

    Arguments:
        loss (starboost.losses.Loss, default=starboost.loss.L2Loss)
            The loss function that will be optimized. At every stage a weak learner will be fit to
            the negative gradient of the loss. The provided value must be a class that at the very
            least implements a ``__call__`` method and a ``gradient`` method.
        base_estimator (sklearn.base.RegressorMixin, default=None): The weak learner. This must be
            a regression model, even though the task is classification. If `None` then a decision
            stump will be used.
        base_estimator_is_tree (bool, default=False): Indicates if the provided ``base_estimator``
            is a tree model or not. Various boosting optimizations specific to trees can be made to
            improve the overall performance.
        n_estimators (int, default=30): The maximum number of weak learners to train. The final
            number of trained weak learners will be lower than ``n_estimators`` if early stopping
            happens.
        init_estimator (sklearn.base.BaseEstimator, default=None): The estimator used to make the
            initial guess. If ``None`` then the ``init_estimator`` property from the ``loss`` will
            be used.
        line_searcher (starboost.line_searchers.LineSearcher, default=None): A line searcher which
            can be used to find the optimal step size during gradient descent. If you've set
            ``base_estimator_is_tree`` to ``True`` and are using one of StarBoost's losses then an optimal
            line searcher will be used, meaning you safely set this field to ``None``.
        learning_rate (float, default=0.1): The learning rate shrinks the contribution of each tree.
            Specifically the descent direction estimated by each weak learner will be multiplied by
            ``learning_rate``. There is a trade-off between learning_rate and ``n_estimators``.
        row_sampling (float, default=1.0): The ratio of rows to sample at each stage.
        col_sampling (float, default=1.0): The ratio of columns to sample at each stage.
        eval_metric (function, default=None): The evaluation metric used to check for early
            stopping. If ``None`` it will default to ``loss``.
        random_state (int, RandomState instance or None, default=None): If int, ``random_state`` is
            the seed used by the random number generator; if ``RandomState`` instance,
            ``random_state`` is the random number generator; if ``None``, the random number
            generator is the ``RandomState`` instance used by ``np.random``.
    """

    def _transform_y_pred(self, y_pred):
        if len(self.classes_) > 2:
            return softmax(y_pred)
        return sigmoid(y_pred)

    @property
    def _default_loss(self):
        return losses.LogLoss()

    def fit(self, X, y, eval_set=None):
        # Check the inputs
        X, y = utils.check_X_y(
            X, y, ['csr', 'csc'], dtype=None, force_all_finite=False,
            multi_output=True
        )
        utils.multiclass.check_classification_targets(y)
        # We first encode the labels into integers starting from 0
        self.encoder_ = preprocessing.LabelEncoder().fit(y)
        y = self.encoder_.transform(y)
        self.classes_ = self.encoder_.classes_
        # Next we one-hot encode the labels
        self.binarizer_ = preprocessing.LabelBinarizer(sparse_output=False).fit(y)
        y = self.binarizer_.transform(y)
        return super().fit(X=X, y=y, eval_set=eval_set)

    def iter_predict_proba(self, X, include_init=False):
        """Returns the predicted probabilities for ``X`` at every stage of the boosting procedure.

        Arguments:
            X (array-like or sparse matrix of shape (n_samples, n_features)): The input samples.
                Sparse matrices are accepted only if they are supported by the weak model.
            include_init (bool, default=False): If ``True`` then the prediction from
                ``init_estimator`` will also be returned.

        Returns:
            iterator of arrays of shape (n_samples, n_classes) containing the predicted
            probabilities at each stage
        """
        utils.validation.check_is_fitted(self, 'init_estimator_')
        X = utils.check_array(X, accept_sparse=['csr', 'csc'], dtype=None, force_all_finite=False)

        probas = np.empty(shape=(len(X), len(self.classes_)), dtype=np.float64)

        for y_pred in super().iter_predict(X, include_init=include_init):
            if len(self.classes_) == 2:
                probas[:, 1] = sigmoid(y_pred[:, 0])
                probas[:, 0] = 1. - probas[:, 1]
            else:
                probas[:] = softmax(y_pred)
            yield probas

    def iter_predict(self, X, include_init=False):
        """Returns the predicted classes for ``X`` at every stage of the boosting procedure.

        Arguments:
            X (array-like or sparse matrix of shape (n_samples, n_features)): The input samples.
                Sparse matrices are accepted only if they are supported by the weak model.
            include_init (bool, default=False): If ``True`` then the prediction from
                ``init_estimator`` will also be returned.

        Returns:
            iterator of arrays of shape (n_samples, n_classes) containing the predicted classes at
            each stage.
        """
        for probas in self.iter_predict_proba(X, include_init=include_init):
            yield self.encoder_.inverse_transform(np.argmax(probas, axis=1))

    def predict_proba(self, X):
        """Returns the predicted probabilities for ``X``.

        Arguments:
            X (array-like or sparse matrix of shape (n_samples, n_features)): The input samples.
                Sparse matrices are accepted only if they are supported by the weak model.

        Returns:
            array of shape (n_samples, n_classes) containing the predicted probabilities.
        """
        return collections.deque(self.iter_predict_proba(X), maxlen=1).pop()
"""
to do:
    add weights to drop trees 

"""