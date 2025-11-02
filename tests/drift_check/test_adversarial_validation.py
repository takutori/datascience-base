import numpy as np

from dsbase.drift_check.adversarial_validation import AdValLightGBMTuner


class TestAdValLightGBMTuner:
    def test_create_dataset(self):
        train_X = np.random.randn(100, 5)
        test_X = np.random.randn(30, 5)

        adval_lightgbm_tuner = AdValLightGBMTuner(val_size=0.2, test_size=0.2)

        dataset_dict = adval_lightgbm_tuner._create_dataset(train_X, test_X)
        train_num = dataset_dict["train"][0].shape[0]
        val_num = dataset_dict["val"][0].shape[0]
        test_num = dataset_dict["test"][0].shape[0]
        assert val_num / (train_num + val_num + test_num) == 0.2
        assert test_num / (train_num + val_num + test_num) == 0.2

    def test_run(self):
        train_X = np.random.randn(10000, 5)
        test_X = np.random.randn(2000, 5)

        adval_lightgbm_tuner = AdValLightGBMTuner(val_size=0.2, test_size=0.2, show_progress_bar=True)

        y_true, y_pred = adval_lightgbm_tuner.run(train_X=train_X, test_X=test_X)

        assert len(y_true) == len(y_pred)
