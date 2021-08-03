import logging
import time
import numpy as np
import ray

from tqdm.auto import tqdm
from pvi.servers import update_client
from hpvi.servers import HPVIServer


class HPVIAsynchronousRayFactory(HPVIServer):
    """
    Acts as both the server and clients to enable scalable distributed
    learning
    """

    def get_default_config(self):
        return {
            **super().get_default_config(),
            "init_q_always": False,
            "ray_options": {
                "num_cpus": 0.1,
                "num_gpus": 0.0,
            },
        }

    def tick(self):

        if self.t0 is None:
            self.t0 = time.time()
            self.pc0 = time.perf_counter()
            self.pt0 = time.process_time()

        working_clients = []
        for i, client in enumerate(self.clients):
            # Get q(θ_k).
            q_i = self.compute_marginal(client_idx=i)

            working_clients.append(
                update_client.options(**self.config["ray_options"]).remote(
                    client, q_i, self.init_q, i
                )
            )

        while not self.should_stop():
            ready_clients, _ = ray.wait(list(working_clients))
            client_id = ready_clients[0]
            client_idx = working_clients.index(client_id)

            # Apply change in factors.
            self.clients[client_idx] = ray.get(client_id)[0]

            # Update global posterior.
            # TODO: Can we make this a remote function too?
            self.q_glob = self.compute_marginal(glob=True)
            self.q = self.compute_marginal(loc=True)

            # Get client training again.
            q_i = self.compute_marginal(client_idx=i)
            if self.config["init_q_always"]:
                working_clients[client_idx] = update_client.options(
                    **self.config["ray_options"]
                ).remote(
                    self.clients[client_idx], q_i, self.init_q, client_idx=client_idx
                )
            else:
                working_clients[client_idx] = update_client.options(
                    **self.config["ray_options"]
                ).remote(self.clients[client_idx], q_i, client_idx=client_idx)

            self.communications += 1
            if self.communications % len(self.clients) == 0:
                # Evaluate current posterior.
                self.evaluate_performance()
                self.log["communications"].append(self.communications)

                metrics = self.log["performance_metrics"][-1]
                print("Communications: {}.".format(self.communications))
                print(
                    "Test mll: {:.3f}. Test acc: {:.3f}.".format(
                        metrics["val_mll"], metrics["val_acc"]
                    )
                )
                print(
                    "Train mll: {:.3f}. Train acc: {:.3f}.\n".format(
                        metrics["train_mll"], metrics["train_acc"]
                    )
                )

    def should_stop(self):
        com_test = self.communications > self.config["max_communications"] - 1

        if len(self.log["performance_metrics"]) > 0:
            perf_test = self.log["performance_metrics"][-1]["val_mll"] < -10
        else:
            perf_test = False

        return com_test or perf_test


class HPVISynchronousRayFactory(HPVIServer):
    """
    Acts as both the server and clients to enable scalable distributed
    learning
    """

    def get_default_config(self):
        return {
            **super().get_default_config(),
            "init_q_always": False,
            "ray_options": {
                "num_cpus": 0.1,
                "num_gpus": 0,
            },
        }

    def tick(self):

        if self.t0 is None:
            self.t0 = time.time()
            self.pc0 = time.perf_counter()
            self.pt0 = time.process_time()

        while not self.should_stop():
            # Pass current q to clients.
            if self.iterations == 0 or self.config["init_q_always"]:
                working_clients = [
                    update_client.options(**self.config["ray_options"]).remote(
                        client, self.compute_marginal(client_idx=i), self.init_q, i
                    )
                    for i, client in enumerate(self.clients)
                ]
            else:
                working_clients = [
                    update_client.options(**self.config["ray_options"]).remote(
                        client, self.compute_marginal(client_idx=i), client_idx=i
                    )
                    for i, client in enumerate(self.clients)
                ]

            # Update stored clients.
            for i, working_client in enumerate(working_clients):
                self.clients[i] = ray.get(working_client)[0]

            # Update global posterior.
            self.q_glob = self.compute_marginal(glob=True)
            self.q = self.compute_marginal(loc=True)

            self.communications += 1
            self.iterations += 1

            # Evaluate current posterior.
            self.evaluate_performance()
            self.log["communications"].append(self.communications)

            metrics = self.log["performance_metrics"][-1]
            print(f"Communications: {self.communications}.")
            print("\nServer performance:")
            print(
                "Test mll: {:.3f}. Test acc: {:.3f}.".format(
                    metrics["q_server"]["val_data"]["mll"],
                    metrics["q_server"]["val_data"]["acc"],
                )
            )
            for k in range(len(self.clients)):
                print(
                    "Client {}. Test mll: {:.3f}. Test acc: {:.3f}.".format(
                        k,
                        metrics["q_server"][f"client_{k}_val_data"]["mll"],
                        metrics["q_server"][f"client_{k}_val_data"]["acc"],
                    )
                )

            for k in range(len(self.clients)):
                print(f"\nClient {k} performance:")
                print(
                    "Test mll: {:.3f}. Test acc: {:.3f}.".format(
                        metrics[f"q_{k}_server"]["val_data"]["mll"],
                        metrics[f"q_{k}_server"]["val_data"]["acc"],
                    )
                )

                print(
                    "Client {}. Test mll: {:.3f}. Test acc: {:.3f}.".format(
                        k,
                        metrics[f"q_{k}_server"][f"client_{k}_val_data"]["mll"],
                        metrics[f"q_{k}_server"][f"client_{k}_val_data"]["acc"],
                    )
                )

    def should_stop(self):
        iter_test = self.iterations > self.config["max_iterations"] - 1

        if len(self.log["performance_metrics"]) > 0:
            perf_test = (
                self.log["performance_metrics"][-1]["q_server"]["val_data"]["mll"] < -10
            )
        else:
            perf_test = False

        return iter_test or perf_test
