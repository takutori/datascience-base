from abc import ABC, abstractmethod

import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import optuna.integration.lightgbm as lgb_tuner
from matplotlib.figure import Figure
from sklearn.metrics import auc, roc_curve
from sklearn.model_selection import train_test_split


class AdValBaseModel(ABC):
    def __init__(
        self,
        val_size: float = 0.2,
        test_size: float = 0.2,
    ):
        self._val_size = val_size
        self._test_size = test_size

    @abstractmethod
    def _create_dataset(self, train_X: np.ndarray, test_X: np.ndarray):
        raise NotImplementedError

    @abstractmethod
    def _fit(self, lgb_train, lgb_val):
        raise NotImplementedError

    @abstractmethod
    def _predict(self, model, data: np.ndarray):
        raise NotImplementedError

    @abstractmethod
    def run(self, train_X: np.ndarray, test_X: np.ndarray):
        raise NotImplementedError


class AdValLightGBMTuner(AdValBaseModel):
    """学習データセットとテストデータセットを特徴量のみから判断する"""

    def __init__(
        self,
        val_size: float = 0.2,
        test_size: float = 0.2,
        verbose_eval=False,
        show_progress_bar=False,  # プログレスバーの非表示
        num_boost_round=100,
        early_stopping_rounds=50,
        random_state: int = 42,
    ):
        """_summary_

        Parameters
        ----------
        val_size : float, optional
            _description_, by default 0.2
        test_size : float, optional
            _description_, by default 0.2
        verbose_eval : bool, optional
            _description_, by default False
        show_progress_bar : bool, optional
            _description_, by default False
        early_stopping_rounds : int, optional
            _description_, by default 50
        random_state : int, optional
            _description_, by default 42
        """
        super().__init__(val_size, test_size)
        self._val_size = val_size
        self._test_size = test_size
        self._verbose_eval = verbose_eval
        self._show_progress_bar = show_progress_bar
        self._num_boost_round = num_boost_round
        self._early_stopping_rounds = early_stopping_rounds
        self._random_state = random_state

    def _create_dataset(self, train_X: np.ndarray, test_X: np.ndarray):
        """
        学習データとテストデータを受けとり、それぞれに0, 1のラベルを付与したのち、合わせたデータセットを作成する。
        その後、このデータセットを通常通り、訓練、検証、テストの三つに分割し、lgb用のデータセットにして出力する。


        Parameters
        ----------
        train_X : np.ndarray
            _description_
        test_X : np.ndarray
            _description_

        Returns
        -------
        _type_
            _description_
        """
        label = np.array([0] * train_X.shape[0] + [1] * test_X.shape[0])
        dataset = np.concatenate([train_X, test_X], axis=0)

        adval_train_X, adval_test_X, adval_train_y, adval_test_y = train_test_split(
            dataset,
            label,
            test_size=self._test_size,
            random_state=self._random_state,
            shuffle=True,
            stratify=label,
        )

        adval_train_X, adval_val_X, adval_train_y, adval_val_y = train_test_split(
            adval_train_X,
            adval_train_y,
            test_size=self._test_size / (1 - self._test_size),
            random_state=self._random_state,
            shuffle=True,
            stratify=adval_train_y,
        )

        return {
            "train": (adval_train_X, adval_train_y),
            "val": (adval_val_X, adval_val_y),
            "test": (adval_test_X, adval_test_y),
        }

    def _fit(self, lgb_train, lgb_val):
        # 固定するparamsは先に指定
        params = {
            "objective": "binary",
            "metric": "binary_logloss",
            "verbose_eval": self._verbose_eval,
            "early_stopping_rounds": self._early_stopping_rounds,
            "boosting": "gbdt",
        }

        opt = lgb_tuner.train(
            params,
            lgb_train,
            valid_sets=lgb_val,
            num_boost_round=self._num_boost_round,
        )

        model = lgb.train(
            opt.params,
            lgb_train,
            valid_sets=lgb_val,
            num_boost_round=self._num_boost_round,
        )

        return model

    def _predict(self, model, data: np.ndarray):
        return model.predict(data)

    def run(self, train_X: np.ndarray, test_X: np.ndarray):
        dataset_dict = self._create_dataset(train_X=train_X, test_X=test_X)

        adval_lgb_train = lgb.Dataset(dataset_dict["train"][0], dataset_dict["train"][1])
        adval_lgb_val = lgb.Dataset(dataset_dict["val"][0], dataset_dict["val"][1])

        model = self._fit(lgb_train=adval_lgb_train, lgb_val=adval_lgb_val)

        return dataset_dict["test"][1], self._predict(model, dataset_dict["test"][0])


class AdversarialValidation:
    def __init__(self, model_cls):
        self._model_cls = model_cls

    def run(self, train_X, test_X):
        y_true, y_predict = self._model_cls.run(train_X, test_X)

        return y_true, y_predict


class Vizalize:
    def __init__(self, y_true, y_predict):
        self._y_true = y_true
        self._y_predict = y_predict

    def roc_curve(self) -> tuple[Figure, float]:
        fpr, tpr, _ = roc_curve(self._y_true, self._y_predict)
        roc_auc = auc(fpr, tpr)

        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(1, 1, 1)
        ax.plot(fpr, tpr)

        ax.set_xlabel("fpt", fontsize=18)
        ax.set_ylabel("tpr", fontsize=18)

        ticks_list = [round(x, 1) for x in np.arange(0, 1.1, 0.1)]
        ax.set_xticks(ticks_list)
        ax.set_xticklabels(map(str, ticks_list))
        ax.set_yticks(ticks_list)
        ax.set_yticklabels(map(str, ticks_list))

        ax.tick_params(labelsize=12)
        ax.grid(linestyle="--")
        ax.set_title(
            f"Adversarial Validation ROC-CURVE AUC: {100*round(roc_auc, 4)}%",
            fontsize=16,
        )

        return fig, roc_auc
