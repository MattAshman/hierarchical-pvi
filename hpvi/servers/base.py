import torch

from pvi.servers.base import Server


class HPVIServer(Server):
    """
    An base class for hierarchical PVI.
    """

    def __init__(self, param_model, clients_val_data=None, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.param_model = param_model
        self.clients_val_data = clients_val_data

        # Update current q_glob and q_loc based on initial client.ts.
        self.q_glob = self.compute_marginal(glob=True)
        self.q = self.compute_marginal(loc=True)

    def evaluate_performance(self, default_metrics=None):
        metrics = {
            "communications": self.communications,
            "iterations": self.iterations,
        }

        if default_metrics is not None:
            metrics = {**default_metrics, **metrics}

        if self.config["performance_metrics"] is not None:
            train_metrics = self.config["performance_metrics"](
                self, self.data, q=self.q
            )
            metrics["q_server"] = {}
            metrics["q_server"]["train_data"] = train_metrics

            if self.val_data is not None:
                # First get q_server performance on val data.
                val_metrics = self.config["performance_metrics"](
                    self, self.val_data, q=self.q
                )

                metrics["q_server"]["val_data"] = val_metrics

                # Now get each client's q_loc performance.
                for i, client in enumerate(self.clients):
                    q_i_server = self.compute_marginal(client_idx=i)
                    val_metrics = self.config["performance_metrics"](
                        self, self.val_data, q=q_i_server
                    )
                    metrics["q_i_server"]["val_data"] = val_metrics

                    q_i_client = client.q
                    val_metrics = self.config["performance_metrics"](
                        self, self.val_data, q=q_i_client
                    )
                    metrics["q_i_client"]["val_data"] = val_metrics

            if self.clients_val_data is not None:
                # Get performance on local validation data.
                for i, (client, val_data) in enumerate(
                    zip(self.clients, self.clients_val_data)
                ):
                    val_metrics = self.config["performance_metrics"](
                        self, val_data, q=self.q
                    )
                    metrics["q_server"][f"client_{i}_val_data"] = val_metrics

                    q_i_server = self.compute_marginal(client_idx=i)
                    val_metrics = self.config["performance_metrics"](
                        self, val_data, q=q_i_server
                    )
                    metrics["q_i_server"][f"client_{i}_val_data"] = val_metrics

                    q_i_client = client.q
                    val_metrics = self.config["performance_metrics"](
                        self, val_data, q=q_i_client
                    )
                    metrics["q_i_client"][f"client_{i}_val_data"] = val_metrics

        if self.config["track_q"]:
            # Store current q(ɸ) natural parameters.
            metrics["npq_phi"] = {
                k: v.detach().cpu() for k, v in self.q_glob.nat_params.items()
            }
            metrics["npq_theta"] = {
                k: v.detach().cpu() for k, v in self.q.nat_params.items()
            }

        self.log["performance_metrics"].append(metrics)

    def compute_marginal(self, glob=False, loc=False, client_idx=None):
        """
        Computes the marginal distibution over local parameters θ_k or global
        parameters ɸ.
        """
        # TODO: currently assumes mean-field Gaussian priors.
        if client_idx is not None:
            # q(θ_k) = t_k(θ_k) ∫ p(ɸ)p(θ_k|ɸ) Π_m [∫ t_m(θ_m)p(θ_m|ɸ)dθ_m]dɸ.

            # t(θ_k).
            t_nps = self.clients[client_idx].t.nat_params

            # Useful to employ a matrix inversion identity here.
            if torch.is_nonzero(sum(t_nps["np2"])):
                t_var = -0.5 / t_nps["np2"]
                t_loc = -0.5 * t_nps["np1"] / t_nps["np2"]

                # t_k(ɸ) = ∫t(θ_k)p(θ_k | ɸ) dθ_k.
                tphi_var = t_var + self.param_model.outputsigma ** 2
                tphi_loc = t_loc
                tphi_np1 = tphi_loc * tphi_var ** -1
                tphi_np2 = -0.5 * tphi_var ** (-1)

            else:
                tphi_np1 = torch.zeros_like(t_nps["np1"])
                tphi_np2 = torch.zeros_like(t_nps["np2"])

            # q_{\k}(ɸ) = p(ɸ) Π_{m≠k} ∫ p(θ_m|ɸ)t_m(θ_m) dθ_m.
            # TODO: assumes q(ɸ) has been computed correctly.
            qphi_nps = self.q_glob.nat_params
            qphi_cav_np1 = qphi_nps["np1"] - tphi_np1
            qphi_cav_np2 = qphi_nps["np2"] - tphi_np2
            qphi_cav_var = -0.5 / qphi_cav_np2
            qphi_cav_loc = -0.5 * qphi_cav_np1 / qphi_cav_np2

            # q_{\k}(θ_k) = ∫ p(θ_k|ɸ)q_{\k}(ɸ) dɸ.
            q_cav_var = qphi_cav_var + self.param_model.outputsigma ** 2
            q_cav_loc = qphi_cav_loc
            q_cav_np1 = q_cav_loc * q_cav_var ** -1
            q_cav_np2 = -0.5 * q_cav_var ** (-1)

            q_np1 = q_cav_np1 + t_nps["np1"]
            q_np2 = q_cav_np2 + t_nps["np2"]
            q_nps = {"np1": q_np1, "np2": q_np2}
            q = self.q.create_new(nat_params=q_nps, is_trainable=False)

            return q

        if glob or loc:
            # q(ɸ) = p(ɸ) Π_m ∫ p(θ_m|ɸ)t_m(θ_m) dθ_m.
            p_nps = self.p.nat_params
            q_np1 = p_nps["np1"]
            q_np2 = p_nps["np2"]

            for client in self.clients:
                # t(θ_m).
                t_nps = client.t.nat_params

                if torch.is_nonzero(sum(t_nps["np2"])):
                    t_var = -0.5 / t_nps["np2"]
                    t_loc = -0.5 * t_nps["np1"] / t_nps["np2"]

                    # t_m(ɸ) = ∫t(θ_m)p(θ_m|ɸ) dθ_m.
                    tphi_var = t_var + self.param_model.outputsigma ** 2
                    tphi_loc = t_loc
                    tphi_np1 = tphi_loc * tphi_var ** -1
                    tphi_np2 = -0.5 * tphi_var ** (-1)

                else:
                    tphi_np1 = torch.zeros_like(t_nps["np1"])
                    tphi_np2 = torch.zeros_like(t_nps["np2"])

                q_np1 += tphi_np1
                q_np2 += tphi_np2

            if glob:
                q_nps = {"np1": q_np1, "np2": q_np2}
            else:
                q_var = -0.5 / q_np2
                q_loc = -0.5 * q_np1 / q_np2

                # q(θ) = ∫p(θ|ɸ)q(ɸ)dɸ.
                q_var = q_var + self.param_model.outputsigma ** 2
                q_loc = q_loc
                q_np1 = q_loc * q_var ** -1
                q_np2 = -0.5 * q_var ** -1
                q_nps = {"np1": q_np1, "np2": q_np2}

            q = self.q.create_new(nat_params=q_nps, is_trainable=False)
            return q

        else:
            return ValueError("Must specifiy either glob, loc or client_idx")
