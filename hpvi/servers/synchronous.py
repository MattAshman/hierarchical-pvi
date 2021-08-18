import logging
import time

from tqdm.auto import tqdm
from hpvi.servers import HPVIServer, MFHPVIServer, HPVIServerBayesianHypers

logger = logging.getLogger(__name__)


class HPVISynchronousServer(HPVIServer):
    def get_default_config(self):
        return {
            **super().get_default_config(),
            "init_q_always": False,
        }

    def tick(self):
        if self.should_stop():
            return False

        logger.debug("Getting client updates.")
        for i, client in tqdm(enumerate(self.clients), leave=False):
            if client.can_update():
                logger.debug(f"On client {i + 1} of {len(self.clients)}.")

                # Get q(θ_k).
                q_i = self.compute_marginal(client_idx=i)

                if self.iterations == 0 or self.config["init_q_always"]:
                    # First iteration. Pass q_init(θ) to client.
                    _, _ = client.fit(q_i, self.init_q)
                else:
                    _, _ = client.fit(q_i)

        # Single communication per iteration.
        self.communications += 1

        logger.debug("Received client updates. Updating global posterior.")

        # Update global posterior.
        self.q_glob = self.compute_marginal(glob=True)
        self.q = self.compute_marginal(loc=True)

        logger.debug(f"Iteration {self.iterations} complete.\n")
        self.iterations += 1

        # Update hyperparameters.
        if (
            self.config["train_model"]
            and self.iterations % self.config["model_update_freq"] == 0
        ):
            self.update_hyperparameters()

        # Log progress.
        self.evaluate_performance()
        self.log["communications"].append(self.communications)

    def should_stop(self):
        return self.iterations > self.config["max_iterations"] - 1


class MFHPVISynchronousServer(MFHPVIServer):
    def get_default_config(self):
        return {
            **super().get_default_config(),
            "init_q_always": False,
        }

    def tick(self):
        if self.should_stop():
            return False

        if self.t0 is None:
            self.t0 = time.time()
            self.pc0 = time.perf_counter()
            self.pt0 = time.process_time()

        logger.debug("Getting client updates.")
        for i, client in tqdm(enumerate(self.clients), leave=False):
            if client.can_update():
                logger.debug(f"On client {i + 1} of {len(self.clients)}.")

                if self.iterations == 0 or self.config["init_q_always"]:
                    # First iteration. Pass q_init(θ) to client.
                    _, _ = client.fit(self.qphi, self.init_q)
                else:
                    _, _ = client.fit(self.qphi)

        # Single communication per iteration.
        self.communications += 1

        logger.debug("Received client updates. Updating global posterior.")

        # Update global posterior.
        self.compute_marginals()

        logger.debug(f"Iteration {self.iterations} complete.\n")
        self.iterations += 1

        # Update hyperparameters.
        if (
            self.config["train_model"]
            and self.iterations % self.config["model_update_freq"] == 0
        ):
            self.update_hyperparameters()

        # Log progress.
        self.evaluate_performance()
        self.log["communications"].append(self.communications)

    def should_stop(self):
        return self.iterations > self.config["max_iterations"] - 1


class HPVISynchronousServerBayesianHypers(HPVIServerBayesianHypers):
    def get_default_config(self):
        return {
            **super().get_default_config(),
            "init_q_always": False,
        }

    def tick(self):

        if self.should_stop():
            return False

        if self.t0 is None:
            self.t0 = time.time()
            self.pc0 = time.perf_counter()
            self.pt0 = time.process_time()

        logger.debug("Getting client updates.")
        for i, client in tqdm(enumerate(self.clients), leave=False):
            if client.can_update():
                logger.debug(f"On client {i + 1} of {len(self.clients)}.")

                if self.iterations == 0 or self.config["init_q_always"]:
                    # First iteration. Pass q_init to client.
                    _, _, _, _, _ = client.fit(
                        self.qphi, self.qa, self.init_q, self.init_q
                    )
                else:
                    _, _, _, _, _ = client.fit(self.qphi, self.qa)

        # Single communication per iteration.
        self.communications += 1

        logger.debug("Received client updates. Updating global posteriors.")

        # Update global posterior.
        self.compute_marginals()

        logger.debug(f"Iteration {self.iterations} complte.\n")
        self.iterations += 1

        # Log progress.
        self.evaluate_performance()
        self.log["communications"].append(self.communications)

    def should_stop(self):
        return self.iterations > self.config["max_iterations"] - 1
