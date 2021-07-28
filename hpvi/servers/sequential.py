import logging

from tqdm.auto import tqdm
from hpvi.servers import HPVIServer

logger = logging.getLogger(__name__)


class HPVISequentialServer(HPVIServer):

    def get_default_config(self):
        return {
            **super().get_default_config(),
            "init_q_always": False,
            "init_q_to_all": False,
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
                if (not self.config["init_q_to_all"] and self.communications
                    == 0) or \
                    (self.config["init_q_to_all"] and self.iterations == 0) \
                        or self.config["init_q_always"]:
                    # First iteratio. Pass q_init(θ) to client.
                    _, _ = client.fit(q_i, self.init_q)
                else:
                    _, _ = client.fit(q_i)

                logger.debug(
                    "Received client updates. Updating global posterior.")

                # Update global posterior.
                self.q_glob = self.compute_marginal(glob=True)
                self.q = self.compute_marginal(loc=True)

                self.communications += 1

                # Evaluate performance after every posterior update for first
                # iteration.
                if self.iterations == 0:
                    self.evaluate_performance()
                    self.log["communications"].append(self.communications)

        logger.debug(f"Iteration {self.iterations} complete.\n")
        self.iterations += 1

        # Update hyperparameters.
        if self.config["train_model"] and \
                self.iterations % self.config["model_update_freq"] == 0:
            self.update_hyperparameters()

        # Log progress.
        self.evaluate_performance()
        self.log["communications"].append(self.communications)

    def should_stop(self):
        return self.iterations > self.config["max_iterations"] - 1
