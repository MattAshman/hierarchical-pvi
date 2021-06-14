import logging

from tqdm.auto import tqdm
from hierarchical_pvi.servers import HPVIServer

logger = logging.getLogger(__name__)


class HPVISynchronousServer(HPVIServer):
    def tick(self):
        if self.should_stop():
            return False

        logger.debug("Getting client updates.")
        for i, client in tqdm(enumerate(self.clients), leave=False):
            if client.can_update():
                logger.debug(f"On client {i + 1} of {len(self.clients)}.")

                # Get q(θ_k).
                q = self.compute_marginal(i)

                if self.iterations == 0:
                    # First iteration. Pass q_init(θ) to client.
                    _, _ = client.fit(q, self.init_q)
                else:
                    _, _ = client.fit(q)

        # Single communication per iteration.
        self.communications += 1

        logger.debug("Received client updates. Updating global posterior.")

        # Update global posterior.
        self.q = self.compute_marginal()

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