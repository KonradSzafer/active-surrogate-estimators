"""Main active testing loop."""
import os
import sys
import socket
import logging
import hydra
import warnings

import torch
import numpy as np
import matplotlib.pyplot as plt

from ase.experiment import Experiment
from ase.utils import maps
from ase.utils.utils import add_val_idxs_to_cfg
from ase.utils.data import to_json
from ase.hoover import Hoover
from ase.models import make_efficient
from omegaconf import OmegaConf


@hydra.main(config_path='conf', config_name='config')
def main(cfg):
    """Run main experiment loop.

    Repeat active testing across multiple data splits and acquisition
    functions for all risk estimators.
    """

    rng = cfg.experiment.random_seed
    if rng == -1:
        rng = np.random.randint(0, 1000)

    if rng is not False:
        np.random.seed(rng)
        torch.torch.manual_seed(rng)

    dcc = cfg.dataset.get('creation_args', dict())
    if dcc.get('dim_normalise_mean', False):
        dim = dcc.dim
        dcc.f_mean = float(dcc.f_mean / np.sqrt(dim))
        dcc.p_mean = float(dcc.p_mean / np.sqrt(dim))
        logging.info(
            f'Updating means in dataset cfg: {cfg.dataset.creation_args}')

    stats = dict(
        dir=os.getcwd(),
        host=socket.gethostname(),
        job_id=os.getenv("SLURM_JOB_ID", None),
        random_state=rng)
    STATS_STATUS = False

    logging.info(
        f'Logging to {stats["dir"]} on {stats["host"]} '
        f'for id={cfg.get("id", -1)}')

    logging.info(f'Slurm job: {stats["job_id"]}.')
    logging.info(f'Setting random seed to {rng}.')
    logging.info(f'Uniform clip val is {cfg.acquisition.uniform_clip}.')

    hoover = Hoover(cfg.hoover)
    model = None

    experiments = {} # run id -> experiment
    for run in range(cfg.experiment.n_runs):
        logging.info(f'Run: {run}')

        dataset = maps.dataset[cfg.dataset.name](
            cfg.dataset, model_cfg=cfg.model)

        # Train model on training data.
        if (not cfg.model.get('keep_constant', False)) or (model is None):
            model = maps.model[cfg.model.name](cfg.model)
            model.fit(dataset.X_train, dataset.Y_train)

            loss_fun = maps.loss[cfg.experiment.loss]()
            train_loss = loss_fun(dataset.Y_train_prob, dataset.Y_train)
            test_loss = loss_fun(dataset.Y_test_prob, dataset.Y_test)
            print(f'Train loss: {train_loss}, Test loss: {test_loss}')

            if not STATS_STATUS:
                STATS_STATUS = True
                stats['loss'] = train_loss
                to_json(stats, 'stats.json')

        if run < cfg.experiment.save_data_until:
            hoover.add_data(run, dataset.export())

        for acq_dict in cfg.acquisition_functions:
            # Slightly unclean, but could not figure out how to make
            # this work with Hydra otherwise
            acquisition = list(acq_dict.keys())[0]
            acq_cfg_name = list(acq_dict.values())[0]

            if cfg.experiment.debug:
                logging.info(f'\t Acquisition: {acquisition}')

            # Reset selected test_indices.
            dataset.restart(acquisition)

            if (n := acq_cfg_name) is not None:
                acq_config = cfg['acquisition_configs'][n]
            else:
                acq_config = None

            experiment = Experiment(run, cfg, dataset, model, acquisition, acq_config)

            testset_size = len(dataset.Y_test)
            for i in range(testset_size):
                logging.info(f'\t Acquisition: {acquisition} – \t Step {i}/{testset_size}')
                experiment.step(i)

            experiments[run] = experiment
            # import matplotlib.pyplot as plt
            # for k, v in experiment.estimated_risk.items():
            #     plt.plot(v, label=k)
            # plt.title(f'Active Testing - dataset ID: {dataset.dataset_id}, run: {run}')
            # plt.legend()
            # plt.show()

            if (n := acq_cfg_name) is not None: # Add config to name for logging
                acquisition = f'{acquisition}_{n}'

    logging.info('Completed all runs.')

    # average risk accross runs
    average_risks = {k: np.zeros(len(v)) for k, v in experiments[0].estimated_risk.items()}
    for ex in experiments.values():
        for k, v in ex.estimated_risk.items():
            average_risks[k] += np.array(v)

    for k, v in average_risks.items():
        plt.plot(v, label=k)
    plt.title(f'Active Testing - dataset ID: {dataset.dataset_id}')
    plt.legend()
    plt.show()


def check_valid(model, dataset):
    """For classification with small number of points and unstratified."""
    if hasattr(model.model, 'n_classes_'):
        if (nc := model.model.n_classes_) != dataset.cfg.n_classes:
            warnings.warn(
                f'Not all classes present in train data. '
                f'Skipping run.')
            return False
    return True


if __name__ == '__main__':
    os.environ['HYDRA_FULL_ERROR'] = '1'

    BASE_DIR = os.getenv('BASE_DIR', default='.')
    RAND = np.random.randint(10000)

    print(
        f"Env variable BASE_DIR: {BASE_DIR}")
    sys.argv.append(f'+BASE_DIR={BASE_DIR}')
    sys.argv.append(f'+RAND={RAND}')

    OmegaConf.register_new_resolver('BASE_DIR', lambda: BASE_DIR)
    OmegaConf.register_new_resolver('RAND', lambda: RAND)

    main()
