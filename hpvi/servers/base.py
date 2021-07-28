import torch

from pvi.servers.base import Server


class HPVIServer(Server):
    """
    An base class for hierarchical PVI.
    """
    def __init__(self, data_model, param_model, p, clients, config=None,
                 init_q=None, data=None, val_data=None):
        super().__init__(data_model, p, clients, config, init_q, val_data)

        self.param_model = param_model
        self.data_model = self.model

        # Update current q_glob and q_loc based on initial client.ts.
        self.q_glob = self.compute_marginal(glob=True)

        # self.q_loc is self.q.
        self.q = self.compute_marginal(loc=True)

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

        elif glob or loc:
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
