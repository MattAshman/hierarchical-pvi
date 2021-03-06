import logging
import torch
import numpy as np

from collections import defaultdict
from torch.utils.data import TensorDataset, DataLoader
from tqdm.auto import tqdm
from pvi.clients import Client

logger = logging.getLogger(__name__)


# =============================================================================
# Client class
# =============================================================================


class HPVIClient(Client):
    """
    The same as Client, excepts retains a self.loc_val_data.
    """

    def __init__(self, loc_val_data=None, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Local validation dataset.
        self.loc_val_data = loc_val_data

    def evaluate_performance(self, default_metrics=None):
        metrics = super().evaluate_performance(default_metrics)

        if self.config["performance_metrics"] is not None:
            if self.loc_val_data is not None:
                loc_val_metrics = self.config["performance_metrics"](
                    self, self.loc_val_data
                )
                for k, v in loc_val_metrics.items():
                    metrics["loc_val_" + k] = v

        return metrics


class HPVIClientBayesianHypers(Client):
    """
    HPVI client with Bayesian treatment of model hyperparameters.

    Learns a contribution t(θ), g(ɸ), l(𝛼) to an approximation to the posterior

    q(ɸ, θ, 𝛼) = p(𝛼)p(ɸ)∏ t(θ_k)g(ɸ)l(𝛼)

    of the hierarchical model p(𝛼)p(ɸ)∏p(θ_k | ɸ, 𝛼) p(y_k | θ_k).
    """

    def __init__(
        self,
        data,
        model,
        param_model,
        tphi=None,
        qtheta=None,
        ta=None,
        config=None,
        val_data=None,
        loc_val_data=None,
    ):
        super().__init__(data, model, tphi, config, val_data)

        self.param_model = param_model

        # q(θ_k).
        self.q = qtheta

        # t(𝛼).
        self.ta = ta

    def get_default_config(self):
        return {
            **super().get_default_config(),
            "num_elbo_hyper_samples": 1,
        }

    def fit(self, qphi, qa, init_qphi=None, init_qtheta=None):
        return self.update_q(qphi, qa, init_qphi, init_qtheta)

    def update_q(self, qphi, qa, init_qphi=None, init_qtheta=None):
        """
        Computes a refined approximate posterior and the associated
        approximating likelihood term.
        """

        # Pass a trainable copy to optimise.
        qphi, self.q, qa, self.t, self.ta = self.gradient_based_update(
            pphi=qphi, pa=qa, init_qphi=init_qphi, init_qtheta=init_qtheta
        )

        return qphi, self.q, qa, self.t, self.ta

    def gradient_based_update(self, pphi, pa, init_qphi=None, init_qtheta=None):
        # Cannot update during optimisation.
        self._can_update = False

        # Copy the approximate posterior, make non-trainable.
        qphi_old = pphi.non_trainable_copy()
        qphi_cav = pphi.non_trainable_copy()
        qa_old = pa.non_trainable_copy()
        qa_cav = pa.non_trainable_copy()

        if self.t is not None:
            qphi_cav = qphi_cav.replace_factor(self.t, None)

        if self.ta is not None:
            qa_cav = qa_cav.replace_factor(self.ta, None)

        if init_qphi is not None:
            qphi = init_qphi.trainable_copy()
        else:
            # Initialise to prior.
            qphi = pphi.trainable_copy()

        qa = pa.trainable_copy()

        if init_qtheta is not None:
            qtheta = init_qtheta.trainable_copy()
        else:
            if self.q is None:
                # Initialise to q(ɸ).
                qtheta = qphi.trainable_copy()
            else:
                # Initialise to previous q(θ).
                qtheta = self.q.trainable_copy()

        qphi_parameters = list(qphi.parameters())
        qa_parameters = list(qa.parameters())
        qtheta_parameters = list(qtheta.parameters())
        if self.config["train_model"]:
            parameters = [
                {"params": qphi_parameters},
                {"params": qa_parameters},
                {"params": qtheta_parameters},
                {
                    "params": self.model.parameters(),
                    **self.config["model_optimiser_params"],
                },
            ]
        else:
            parameters = [
                {"params": qphi_parameters},
                {"params": qa_parameters},
                {"params": qtheta_parameters},
            ]

        # Reset optimiser.
        logging.info("Resetting optimiser")
        optimiser = getattr(torch.optim, self.config["optimiser"])(
            parameters, **self.config["optimiser_params"]
        )
        lr_scheduler = getattr(torch.optim.lr_scheduler, self.config["lr_scheduler"])(
            optimiser, **self.config["lr_scheduler_params"]
        )

        # Set up data
        x = self.data["x"]
        y = self.data["y"]

        tensor_dataset = TensorDataset(x, y)
        loader = DataLoader(
            tensor_dataset, batch_size=self.config["batch_size"], shuffle=True
        )

        if self.config["device"] == "cuda":
            loader.pin_memory = True

        # Dict for logging optimisation progress.
        training_metrics = defaultdict(list)

        # Dict for logging performance progress.
        performance_metrics = defaultdict(list)

        # Reset early stopping.
        self.config["early_stopping"](
            scores=None,
            model=[
                qphi.non_trainable_copy(),
                qa.non_trainable_copy(),
                qtheta.non_trainable_copy(),
            ],
        )

        # Gradient-based optimisation loop -- loop over epochs.
        epoch_iter = tqdm(
            range(self.config["epochs"]),
            desc="Epoch",
            leave=True,
            disable=(not self.config["verbose"]),
        )
        # for i in range(self.config["epochs"]):
        for i in epoch_iter:
            epoch = defaultdict(lambda: 0.0)

            # Loop over batches in current epoch
            for (x_batch, y_batch) in iter(loader):
                x_batch = x_batch.to(self.config["device"])
                y_batch = y_batch.to(self.config["device"])

                optimiser.zero_grad()

                batch = {
                    "x": x_batch,
                    "y": y_batch,
                }

                # Compute the KL divergences.
                klphi = qphi.kl_divergence(qphi_cav).sum() / len(x)
                kla = sum(qa.kl_divergence(qa_cav).values()) / len(x)

                # Now to estimate E_{q(ɸ)q(𝛼)}[KL(q(θ)||p(θ|ɸ,𝛼))].
                kltheta = 0
                for _ in range(self.config["num_elbo_hyper_samples"]):
                    # Sample and set hyperparameters of parameter model.
                    alpha = qa.rsample()
                    self.param_model.hyperparameters = alpha
                    # phi = qphi.rsample()

                    # sigma = self.param_model.outputsigma
                    # ptheta_phi = qtheta.non_trainable_copy()
                    # ptheta_phi.std_params = {
                    #    "loc": phi,
                    #    "scale": torch.ones_like(phi) * sigma,
                    # }
                    # kltheta += qtheta.kl_divergence(ptheta_phi).sum()
                    # Compute E_{q(ɸ)}[KL(q(θ)||p(θ|ɸ,𝛼))] in closed-form.
                    # KL(q(θ) | p(θ | ɸ))
                    #   = (η_q - η_p) ^ T E_q[f(θ)] - A(η_q) + A(η_p).
                    # E_ɸ[KL(q(θ) | p(θ))]
                    #   = (η_q - E_ɸ[η_p]) ^ T E_q[f(θ)] - A(η_q) + E_ɸ[A(η_p)].
                    # TODO: assumes Gaussian distributions.
                    sigma = self.param_model.outputsigma
                    npq = torch.cat(list(qtheta.nat_params.values()))
                    mq = torch.cat(list(qtheta.mean_params.values()))
                    log_aq = qtheta.log_a()
                    enpp = torch.cat(
                        [
                            qphi.std_params["loc"] / sigma ** 2,
                            torch.ones_like(qphi.std_params["loc"])
                            * (-1 / (2 * sigma ** 2)),
                        ]
                    )
                    elog_ap = (
                        qphi.std_params["loc"] ** 2 + qphi.std_params["scale"] ** 2
                    ) / (2 * sigma ** 2)
                    elog_ap += torch.log(sigma)
                    elog_ap = elog_ap.sum() - 0.5 * np.log(np.pi) * len(
                        qphi.std_params["loc"]
                    )
                    kltheta += (npq - enpp).dot(mq) - log_aq + elog_ap

                kltheta /= len(x) * self.config["num_elbo_hyper_samples"]

                # Sample θ from q(θ) and compute p(y | θ, x) for each θ
                ll = self.model.expected_log_likelihood(
                    batch, qtheta, self.config["num_elbo_samples"]
                ).sum()
                ll /= len(x_batch)

                # Compute E_q[log t(ɸ)] and E_q[log t(𝛼)].
                # logt = self.t.eqlogt(qphi, self.config["num_elbo_samples"])
                # logt /= len(x)
                # logta = self.ta.eqlogt(qa, self.config["num_elbo_samples"])
                # logta /= len(x)
                logt = torch.tensor(0.0).to(self.config["device"])
                logta = torch.tensor(0.0).to(self.config["device"])

                loss = klphi + kla + kltheta - ll
                loss.backward()
                optimiser.step()

                # Keep track of quantities for current batch.
                epoch["elbo"] += -loss.item() / len(loader)
                epoch["klphi"] += klphi.item() / len(loader)
                epoch["kla"] += kla.item() / len(loader)
                epoch["kltheta"] += kltheta.item() / len(loader)
                epoch["ll"] += ll.item() / len(loader)
                epoch["logt"] += logt.item() / len(loader)
                epoch["logta"] += logta.item() / len(loader)

            epoch_iter.set_postfix(
                elbo=epoch["elbo"],
                klphi=epoch["klphi"],
                kla=epoch["kla"],
                kltheta=epoch["kltheta"],
                ll=epoch["ll"],
            )

            # Log progress for current epoch.
            training_metrics["elbo"].append(epoch["elbo"])
            training_metrics["klphi"].append(epoch["klphi"])
            training_metrics["kla"].append(epoch["kla"])
            training_metrics["kltheta"].append(epoch["kltheta"])
            training_metrics["ll"].append(epoch["ll"])

            if self.t is not None:
                training_metrics["logt"].append(epoch["logt"])
                training_metrics["logta"].append(epoch["logta"])

            # Check whether to stop early.
            stop_early = self.config["early_stopping"](
                scores=training_metrics,
                model=[
                    qphi.non_trainable_copy(),
                    qa.non_trainable_copy(),
                    qtheta.non_trainable_copy(),
                ],
            )

            if (
                (i > 0 and i % self.config["print_epochs"] == 0)
                or i == (self.config["epochs"] - 1)
                or stop_early
            ):
                # Update local posterior before evaluating performance.
                self.q = qtheta.non_trainable_copy()

                metrics = self.evaluate_performance(
                    {
                        "epochs": i,
                        "elbo": epoch["elbo"],
                        "klphi": epoch["klphi"],
                        "kla": epoch["kla"],
                        "kltheta": epoch["kltheta"],
                        "ll": epoch["ll"],
                    }
                )

                # Report performance.
                report = ""
                report += f"epochs: {metrics['epochs']} "
                report += f"elbo: {metrics['elbo']:.3f} "
                report += f"ll: {metrics['ll']:.3f} "
                report += f"klphi: {metrics['klphi']:.3f} "
                report += f"kla: {metrics['kla']:.3f} "
                report += f"kltheta: {metrics['kltheta']:.3f} \n"
                for k, v in metrics.items():
                    performance_metrics[k].append(v)
                    if "mll" in k or "acc" in k:
                        report += f"{k}: {v:.3f} "

                tqdm.write(report)

            # Update learning rate.
            lr_scheduler.step()

            # Check whether to stop early.
            if stop_early:
                break

        # Log the training curves for this update.
        self.log["training_curves"].append(training_metrics)
        self.log["performance_curves"].append(performance_metrics)

        # Create non-trainable copy to send back to server.
        if self.config["early_stopping"].stash_model:
            qphi_new, qa_new, qtheta_new = self.config["early_stopping"].best_model
        else:
            qphi_new = qphi.non_trainable_copy()
            qa_new = qa.non_trainable_copy()
            qtheta_new = qtheta.non_trainable_copy()

        # Finished optimisation, can now update.
        self._can_update = True

        if self.t is not None:
            # Compute new local contribution from old distributions
            t_new = self.t.compute_refined_factor(
                qphi_new,
                qphi_old,
                damping=self.config["damping_factor"],
                valid_dist=self.config["valid_factors"],
                update_log_coeff=self.config["update_log_coeff"],
            )

            ta_new = self.ta.compute_refined_factor(
                qa_new,
                qa_old,
                damping=self.config["damping_factor"],
                valid_dist=self.config["valid_factors"],
                update_log_coeff=self.config["update_log_coeff"],
            )

            return qphi_new, qtheta_new, qa_new, t_new, ta_new

        else:
            return qphi_new, qtheta_new, qa_new, None, None


class MFHPVIClient(Client):
    """
    Learns a contribution t(ɸ) g(θ) to an approximation to the posterior,

    q(ɸ)q(θ) = p(ɸ) Π t(ɸ) g(θ)

    of the hierarchical model p(ɸ) Π p(θ|ɸ) p(y | θ).
    """

    def __init__(
        self,
        data,
        model,
        param_model,
        tphi=None,
        qtheta=None,
        config=None,
        val_data=None,
        loc_val_data=None,
    ):

        super().__init__(data, model, tphi, config, val_data)

        self.param_model = param_model

        # q(θ_k).
        self.q = qtheta

    def update_q(self, q, init_q=None):
        """
        Computes a refined approximate posterior and the associated
        approximating likelihood term.
        """

        # Pass a trainable copy to optimise.
        qphi, self.q, self.t = self.gradient_based_update(p=q, init_q=init_q)

        # Only return new q and approximate likelihood term. Server doesn't
        # need to know about local model.
        return qphi, self.t

    def gradient_based_update(self, p, init_q=None):
        # Cannot update during optimisation.
        self._can_update = False

        # Copy the approximate posterior, make non-trainable.
        qphi_old = p.non_trainable_copy()
        qphi_cav = p.non_trainable_copy()

        if self.q is None:
            # Set equal to prior for q(ɸ).
            qtheta_old = p.non_trainable_copy()
        else:
            qtheta_old = self.q.non_trainable_copy()

        if self.t is not None:
            # TODO: check if valid distribution.
            qphi_cav = qphi_cav.replace_factor(self.t, None)

        if init_q is not None:
            qphi = init_q.trainable_copy()
            qtheta = init_q.trainable_copy()
        else:
            # Initialise to prior.
            qphi = p.trainable_copy()

            if self.q is None:
                qtheta = qphi.trainable_copy()
            else:
                qtheta = self.q.trainable_copy()

        # TODO: currently assumes Gaussian distribution.
        # Parameters are those of q(ɸ), q(θ) and self.model.
        qphi_parameters = list(qphi.parameters())
        qtheta_parameters = list(qtheta.parameters())
        if self.config["train_model"]:
            parameters = [
                {"params": qphi_parameters},
                {"params": qtheta_parameters},
                {
                    "params": self.model.parameters(),
                    **self.config["model_optimiser_params"],
                },
            ]
        else:
            parameters = [{"params": qphi_parameters}, {"params": qtheta_parameters}]

        # Reset optimiser.
        logging.info("Resetting optimiser")
        optimiser = getattr(torch.optim, self.config["optimiser"])(
            parameters, **self.config["optimiser_params"]
        )
        lr_scheduler = getattr(torch.optim.lr_scheduler, self.config["lr_scheduler"])(
            optimiser, **self.config["lr_scheduler_params"]
        )

        # Set up data
        x = self.data["x"]
        y = self.data["y"]

        tensor_dataset = TensorDataset(x, y)
        loader = DataLoader(
            tensor_dataset, batch_size=self.config["batch_size"], shuffle=True
        )

        if self.config["device"] == "cuda":
            loader.pin_memory = True

        # Dict for logging optimisation progress.
        training_metrics = defaultdict(list)

        # Dict for logging performance progress.
        performance_metrics = defaultdict(list)

        # Reset early stopping.
        self.config["early_stopping"](
            scores=None, model=[qphi.non_trainable_copy(), qtheta.non_trainable_copy()]
        )

        # Gradient-based optimisation loop -- loop over epochs.
        epoch_iter = tqdm(
            range(self.config["epochs"]),
            desc="Epoch",
            leave=True,
            disable=(not self.config["verbose"]),
        )
        # for i in range(self.config["epochs"]):
        for i in epoch_iter:
            epoch = defaultdict(lambda: 0.0)

            # Loop over batches in current epoch
            for (x_batch, y_batch) in iter(loader):
                x_batch = x_batch.to(self.config["device"])
                y_batch = y_batch.to(self.config["device"])

                optimiser.zero_grad()

                batch = {
                    "x": x_batch,
                    "y": y_batch,
                }

                # Compute the KL divergence between q(ɸ) and q_cav(ɸ), ignoring
                # A(η_cav).
                klphi = qphi.kl_divergence(qphi_cav, calc_log_ap=False).sum() / len(x)

                # Estimate KL divergence E_{q(ɸ)}[KL(q(θ) || p(θ | ɸ))].
                # phis = q.rsample((self.config["num_elbo_samples"],))
                # kl_loc = torch.tensor(0.)
                # for phi in phis:
                #     p_loc = self.param_model.likelihood_forward(
                #         phi, theta=None)
                #     kl_loc += torch.distributions.kl_divergence(
                #         qtheta.distribution, p_loc).sum() / len(x)
                #
                # kl_loc /= self.config["num_elbo_samples"]

                # TODO: Closed-form solution for Gaussians.
                # KL(q(θ) | p(θ | ɸ))
                #   = (η_q - η_p) ^ T E_q[f(θ)] - A(η_q) + A(η_p).
                # E_ɸ[KL(q(θ) | p(θ))]
                #   = (η_q - E_ɸ[η_p]) ^ T E_q[f(θ)] - A(η_q) + E_ɸ[A(η_p)].
                sigma = self.param_model.outputsigma
                npq = torch.cat([np for np in qtheta.nat_params.values()])
                mq = torch.cat([mp for mp in qtheta.mean_params.values()])
                log_aq = qtheta.log_a()
                enpp = torch.cat(
                    [
                        qphi.std_params["loc"] / sigma ** 2,
                        torch.ones_like(qphi.std_params["loc"])
                        * (-1 / (2 * sigma ** 2)),
                    ]
                )
                elog_ap = (
                    qphi.std_params["loc"] ** 2 + qphi.std_params["scale"] ** 2
                ) / (2 * sigma ** 2)
                elog_ap += torch.log(sigma)
                elog_ap = elog_ap.sum() - 0.5 * np.log(np.pi) * len(
                    qphi.std_params["loc"]
                )
                kltheta = (npq - enpp).dot(mq) - log_aq + elog_ap
                kltheta /= len(x)

                # Sample θ from q(θ) and compute p(y | θ, x) for each θ
                ll = self.model.expected_log_likelihood(
                    batch, qtheta, self.config["num_elbo_samples"]
                ).sum()
                ll /= len(x_batch)

                # Compute E_q[log t(θ)].
                # logt = self.t.eqlogt(q, self.config["num_elbo_samples"])
                # logt /= len(x)
                logt = torch.tensor(0.0).to(self.config["device"])

                loss = klphi + kltheta - ll
                loss.backward()
                optimiser.step()

                # Keep track of quantities for current batch.
                epoch["elbo"] += -loss.item() / len(loader)
                epoch["klphi"] += klphi.item() / len(loader)
                epoch["kltheta"] += kltheta.item() / len(loader)
                epoch["ll"] += ll.item() / len(loader)
                epoch["logt"] += logt.item() / len(loader)

            epoch_iter.set_postfix(
                elbo=epoch["elbo"],
                klphi=epoch["klphi"],
                kltheta=epoch["kltheta"],
                ll=epoch["ll"],
            )

            # Log progress for current epoch.
            training_metrics["elbo"].append(epoch["elbo"])
            training_metrics["klphi"].append(epoch["klphi"])
            training_metrics["kltheta"].append(epoch["kltheta"])
            training_metrics["ll"].append(epoch["ll"])

            if self.t is not None:
                training_metrics["logt"].append(epoch["logt"])

            # Check whether to stop early.
            stop_early = self.config["early_stopping"](
                scores=training_metrics,
                model=[qphi.non_trainable_copy(), qtheta.non_trainable_copy()],
            )

            if (
                (i > 0 and i % self.config["print_epochs"] == 0)
                or i == (self.config["epochs"] - 1)
                or stop_early
            ):
                # Update global posterior before evaluating performance.
                self.q = qtheta.non_trainable_copy()

                metrics = self.evaluate_performance(
                    {
                        "epochs": i,
                        "elbo": epoch["elbo"],
                        "klphi": epoch["klphi"],
                        "kltheta": epoch["kltheta"],
                        "ll": epoch["ll"],
                    }
                )

                # Report performance.
                report = ""
                report += f"epochs: {metrics['epochs']} "
                report += f"elbo: {metrics['elbo']:.3f} "
                report += f"ll: {metrics['ll']:.3f} "
                report += f"klphi: {metrics['klphi']:.3f} "
                report += f"kltheta: {metrics['kltheta']:.3f} \n"
                for k, v in metrics.items():
                    performance_metrics[k].append(v)
                    if "mll" in k or "acc" in k:
                        report += f"{k}: {v:.3f} "

                tqdm.write(report)

            # Update learning rate.
            lr_scheduler.step()

            # Check whether to stop early.
            if stop_early:
                break

        # Log the training curves for this update.
        self.log["training_curves"].append(training_metrics)
        self.log["performance_curves"].append(performance_metrics)

        # Create non-trainable copy to send back to server.
        if self.config["early_stopping"].stash_model:
            qphi_new, qtheta_new = self.config["early_stopping"].best_model
        else:
            qphi_new = qphi.non_trainable_copy()
            qtheta_new = qtheta.non_trainable_copy()

        # Finished optimisation, can now update.
        self._can_update = True

        if self.t is not None:
            # Compute new local contribution from old distributions
            t_new = self.t.compute_refined_factor(
                qphi_new,
                qphi_old,
                damping=self.config["damping_factor"],
                valid_dist=self.config["valid_factors"],
                update_log_coeff=self.config["update_log_coeff"],
            )

            # Apply damping to qtheta_new.
            qtheta_new_nps = {
                k: v * self.config["damping_factor"]
                + (1 - self.config["damping_factor"]) * qtheta_old.nat_params[k]
                for k, v in qtheta_new.nat_params.items()
            }
            qtheta_new = qtheta_new.create_new(
                nat_params=qtheta_new_nps, is_trainable=False
            )

            return qphi_new, qtheta_new, t_new

        else:
            return qphi_new, qtheta_new, None


class HPVIClientIndependent(Client):
    """
    Rather than learn a contribution to an approximation to the posterior,

    q(ɸ)q(θ) = p(ɸ) Π t(ɸ) g(θ)

    of the hierarchical model p(ɸ) Π p(θ|ɸ) p(y | θ), this client learns a
    contribution of an approximation to the posterior,

    q(ɸ) = p(ɸ) Π t(ɸ)

    of the non-hierarchical model p(ɸ) Π p(y | ɸ), and then learns an
    approximation, q(θ), to the posterior of the local model

    p(θ, y) =  [ ∫ q_cav(ɸ) p(θ|ɸ) dɸ ] p(y | θ)

    where q_cav(ɸ) excludes the clients factor (as to not count the data twice)
    and the mixture ∫ q_cav(ɸ) p(θ|ɸ) dɸ is used as the prior.
    """

    def __init__(
        self,
        data,
        model,
        param_model,
        t=None,
        qtheta=None,
        config=None,
        val_data=None,
    ):

        super().__init__(data, model, t, config, val_data)

        self.param_model = param_model
        self.model = self.model

        # q(θ_k).
        self.q = qtheta

    def update_q(self, q, init_q=None):
        """
        Computes a refined approximate posterior and the associated
        approximating likelihood term, and then updates the local approximate
        posterior.
        """

        # Find new local approximate posterior.
        qtheta = self.local_gradient_based_update(q_glob=q)

        # Find new global contribution.
        q_new, self.t = self.gradient_based_update(p=q, init_q=init_q)

        self.q = qtheta

        # Only return new q and approximate likelihood term. Server doesn't
        # need to know about local model.
        return q_new, self.t

    def local_gradient_based_update(self, q_glob):
        # Cannot update during optimisation.
        self._can_update = False

        # Compute cavity.
        q_cav = q_glob.non_trainable_copy()
        q_cav.nat_params = {
            k: v - self.t.nat_params[k] for k, v in q_cav.nat_params.items()
        }

        # In the case of Gaussians, can compute p(θ) = ∫ q(ɸ) p(θ | ɸ) dɸ in
        # closed-form.
        # TODO: obviously this isn't always the case.
        p_loc_std_params = {
            "loc": q_cav.std_params["loc"],
            "scale": (
                q_cav.std_params["scale"] ** 2 + self.param_model.outputsigma ** 2
            )
            ** 0.5,
        }

        # TODO: which to use?
        p_loc = q_cav.create_new(std_params=p_loc_std_params, is_trainable=False)
        # p_loc = q_glob.non_trainable_copy()

        if self.q is None:
            # Initialise to q(ɸ).
            qtheta = q_glob.trainable_copy()
        else:
            qtheta = self.q.trainable_copy()

        # Parameters are those of q(θ).
        parameters = qtheta.parameters()

        # Reset optimiser.
        logging.info("Resetting optimiser")
        optimiser = getattr(torch.optim, self.config["optimiser"])(
            parameters, **self.config["optimiser_params"]
        )
        lr_scheduler = getattr(torch.optim.lr_scheduler, self.config["lr_scheduler"])(
            optimiser, **self.config["lr_scheduler_params"]
        )

        # Set up data
        x = self.data["x"]
        y = self.data["y"]

        tensor_dataset = TensorDataset(x, y)
        loader = DataLoader(
            tensor_dataset, batch_size=self.config["batch_size"], shuffle=True
        )

        if self.config["device"] == "cuda":
            loader.pin_memory = True

        # Dict for logging optimisation progress.
        training_metrics = defaultdict(list)

        # Dict for logging performance progress.
        performance_metrics = defaultdict(list)

        # Reset early stopping.
        self.config["early_stopping"](None, qtheta.non_trainable_copy())

        # Gradient-based optimisation loop -- loop over epochs.
        epoch_iter = tqdm(
            range(self.config["epochs"]),
            desc="Epoch",
            leave=True,
            disable=(not self.config["verbose"]),
        )

        # for i in range(self.config["epochs"]):
        for i in epoch_iter:
            epoch = defaultdict(lambda: 0.0)

            # Loop over batches in current epoch
            for (x_batch, y_batch) in iter(loader):
                x_batch = x_batch.to(self.config["device"])
                y_batch = y_batch.to(self.config["device"])

                optimiser.zero_grad()

                batch = {
                    "x": x_batch,
                    "y": y_batch,
                }

                # Compute the KL divergence between q(θ) and p(θ), ignoring
                # A(η_0).
                kl = qtheta.kl_divergence(p_loc, calc_log_ap=False).sum()
                kl /= len(x)

                # Sample θ from q(θ) and compute p(y | θ) for each θ
                ll = self.model.expected_log_likelihood(
                    batch, qtheta, self.config["num_elbo_samples"]
                ).sum()
                ll /= len(x_batch)

                loss = kl - ll
                loss.backward()
                optimiser.step()

                # Keep track of quantities for current batch.
                epoch["elbo"] += -loss.item() / len(loader)
                epoch["kl"] += kl.item() / len(loader)
                epoch["ll"] += ll.item() / len(loader)

            epoch_iter.set_postfix(elbo=epoch["elbo"], kl=epoch["kl"], ll=epoch["ll"])

            # Log progress for current epoch.
            training_metrics["elbo"].append(epoch["elbo"])
            training_metrics["kl"].append(epoch["kl"])
            training_metrics["ll"].append(epoch["ll"])

            # Check whether to stop early.
            stop_early = self.config["early_stopping"](
                training_metrics, qtheta.non_trainable_copy()
            )

            if (
                (i > 0 and i % self.config["print_epochs"] == 0)
                or i == (self.config["epochs"] - 1)
                or stop_early
            ):
                # Update global posterior before evaluating performance.
                self.q = qtheta.non_trainable_copy()

                metrics = self.evaluate_performance(
                    {
                        "epochs": i,
                        "elbo": epoch["elbo"],
                        "kl": epoch["kl"],
                        "ll": epoch["ll"],
                    }
                )

                # Report performance.
                report = ""
                report += f"epochs: {metrics['epochs']} "
                report += f"elbo: {metrics['elbo']:.3f} "
                report += f"ll: {metrics['ll']:.3f} "
                report += f"kl: {metrics['kl']:.3f} \n"
                for k, v in metrics.items():
                    performance_metrics[k].append(v)
                    if "mll" in k or "acc" in k:
                        report += f"{k}: {v:.3f} "

                tqdm.write(report)

            # Update learning rate.
            lr_scheduler.step()

            # Check whether to stop early.
            if stop_early:
                break

        # Log the training curves for this update.
        self.log["local_training_curves"].append(training_metrics)
        self.log["local_performance_curves"].append(performance_metrics)

        # Create non-trainable copy to send back to server.
        if self.config["early_stopping"].stash_model:
            qtheta_new = self.config["early_stopping"].best_model
        else:
            qtheta_new = qtheta.non_trainable_copy()

        # Finished optimisation, can now update.
        self._can_update = True

        return qtheta_new
