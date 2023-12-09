"""Models for active testing."""

import logging
from omegaconf import OmegaConf, DictConfig
import numpy as np
from scipy.stats import special_ortho_group

from ase.loss import RMSELoss, AccuracyLoss, CrossEntropyLoss


class BaseModel:
    """Base class for models."""
    def __init__(self, cfg):
        # Set task_type and global_std if not present.
        self.cfg = OmegaConf.merge(
            dict(cfg),
            DictConfig(dict(task_type=cfg.get('task_type', 'regression')))
        )

    def fit(self, x, y):
        raise NotImplementedError

    def predict(self, x, **kwargs):
        raise NotImplementedError

    def performance(self, x, y, task_type):
        pred = self.predict(x)

        if task_type == 'regression':
            loss = RMSELoss()(pred, y)
            logging.info(f'RMSE: {loss}')
            return loss
        elif task_type == 'classification':
            loss = [
                AccuracyLoss()(pred, y).mean(),
                CrossEntropyLoss()(pred, y).mean()]
            logging.info(f'Accuracy: {loss[0]*100}%.')
            logging.info(f'CrossEntropy: {loss[1]}.')
            return loss
        else:
            raise ValueError


class SKLearnModel(BaseModel):
    """SKLearn derived models."""
    def __init__(self, cfg):
        super().__init__(cfg)
        self.is_fit = False
        if cfg.type== 'GradientBoostingClassifier':
            from sklearn.ensemble import GradientBoostingClassifier
            self.model = GradientBoostingClassifier()
        elif cfg.type == 'RandomForestClassifier':
            from sklearn.ensemble import RandomForestClassifier
            self.model = RandomForestClassifier()
        elif cfg.type == 'DecisionTreeClassifier':
            from sklearn.tree import DecisionTreeClassifier
            self.model = DecisionTreeClassifier()
        elif cfg.type == 'LogisticRegression':
            from sklearn.linear_model import LogisticRegression
            self.model = LogisticRegression()
        else:
            raise ValueError(f'Unknown model type {cfg.type}.')

        # from sklearn.calibration import CalibratedClassifierCV
        # self.model = CalibratedClassifierCV(
        #     self.model,
        #     method='isotonic',
        #     # method='sigmoid',
        #     cv=5
        # )


    def fit(self, x, y, **kwargs):
        if x.ndim == 1:
            x = x[..., np.newaxis] # Sklearn expects x to be NxD
        self.model = self.model.fit(x, y, **kwargs)
        self.is_fit = True


    def test(self, x, y, loss_function: callable):
        y_pred = self.model.predict(x)
        return loss_function(y_pred, y)


    def predict(self, x, idxs=None, *args, **kwargs):
        predict_proba = True
        if idxs is not None:
            x = x[idxs]
        return self.predict_sk(x, predict_proba=predict_proba, **kwargs)


    def predict_sk(self, x, predict_proba, **kwargs):
        if predict_proba:
            y = self.model.predict_proba(x, **kwargs)
        else:
            y = self.model.predict(x, **kwargs)
        return y
