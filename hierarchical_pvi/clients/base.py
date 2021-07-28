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


class HPVIClientJoint(Client):
    """
    Learns a contribution t(ɸ) g(θ) to an approximation to the posterior,

    q(ɸ)q(θ) = p(ɸ) Π t(ɸ) g(θ)

    of the hierarchical model p(ɸ) Π p(θ|ɸ) p(y | θ).
    """
    def __init__(self, data, data_model, param_model, t=None, q_loc=None,
                 config=None, val_data=None):

        super().__init__(data, data_model, t, config, val_data)

        self.param_model = param_model
        self.data_model = self.model

        # q(θ_k).
        self.q = q_loc

    def update_q(self, q, init_q=None):
        """
        Computes a refined approximate posterior and the associated
        approximating likelihood term.
        """

        # Pass a trainable copy to optimise.
        q_glob, self.t, self.q = self.gradient_based_update(
            p=q, init_q=init_q)

        # Only return new q and approximate likelihood term. Server doesn't
        # need to know about local model.
        return q_glob, self.t

    def gradient_based_update(self, p, init_q=None):
        # Cannot update during optimisation.
        self._can_update = False

        # Copy the approximate posterior, make non-trainable.
        q_old = p.non_trainable_copy()
        q_cav = p.non_trainable_copy()

        if self.t is not None:
            # TODO: check if valid distribution.
            q_cav.nat_params = {k: v - self.t.nat_params[k]
                                for k, v in q_cav.nat_params.items()}

        if init_q is not None:
            q = init_q.trainable_copy()
        else:
            # Initialise to prior.
            q = p.trainable_copy()

        if self.q is None:
            # Initialise to q(ɸ).
            q_loc = p.trainable_copy()
        else:
            q_loc = self.q.trainable_copy()

        # TODO: currently assumes Gaussian distribution.
        # Parameters are those of q(ɸ), q(θ) and self.model.
        q_parameters = list(q.parameters())
        q_loc_parameters = list(q_loc.parameters())
        if self.config["train_model"]:
            parameters = [
                {"params": q_parameters},
                {"params": q_loc_parameters},
                {"params": self.data_model.parameters(),
                 **self.config["model_optimiser_params"]}
            ]
        else:
            parameters = [
                {"params": q_parameters},
                {"params": q_loc_parameters}
            ]

        # Reset optimiser.
        logging.info("Resetting optimiser")
        optimiser = getattr(torch.optim, self.config["optimiser"])(
            parameters, **self.config["optimiser_params"])
        lr_scheduler = getattr(torch.optim.lr_scheduler,
                               self.config["lr_scheduler"])(
            optimiser, **self.config["lr_scheduler_params"])

        # Set up data
        x = self.data["x"]
        y = self.data["y"]

        tensor_dataset = TensorDataset(x, y)
        loader = DataLoader(tensor_dataset,
                            batch_size=self.config["batch_size"],
                            shuffle=True)

        if self.config["device"] == "cuda":
            loader.pin_memory = True

        # Dict for logging optimisation progress.
        training_metrics = defaultdict(list)

        # Dict for logging performance progress.
        performance_metrics = defaultdict(list)

        # Reset early stopping.
        self.config["early_stopping"](
            scores=None,
            model=[q.non_trainable_copy(), q_loc.non_trainable_copy()])

        # Gradient-based optimisation loop -- loop over epochs.
        epoch_iter = tqdm(range(self.config["epochs"]), desc="Epoch",
                          leave=True, disable=(not self.config["verbose"]))
        # for i in range(self.config["epochs"]):
        for i in epoch_iter:
            epoch = defaultdict(lambda: 0.)

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
                kl = q.kl_divergence(q_cav, calc_log_ap=False).sum() / len(x)

                # Estimate KL divergence E_{q(ɸ)}[KL(q(θ) || p(θ | ɸ))].
                # phis = q.rsample((self.config["num_elbo_samples"],))
                # kl_loc = torch.tensor(0.)
                # for phi in phis:
                #     p_loc = self.param_model.likelihood_forward(
                #         phi, theta=None)
                #     kl_loc += torch.distributions.kl_divergence(
                #         q_loc.distribution, p_loc).sum() / len(x)
                #
                # kl_loc /= self.config["num_elbo_samples"]

                # TODO: Closed-form solution for Gaussians.
                # KL(q(θ) | p(θ | ɸ))
                #   = (η_q - η_p) ^ T E_q[f(θ)] - A(η_q) + A(η_p).
                # E_ɸ[KL(q(θ) | p(θ))]
                #   = (η_q - E_ɸ[η_p]) ^ T E_q[f(θ)] - A(η_q) + E_ɸ[A(η_p)].
                sigma = self.param_model.outputsigma
                npq = torch.cat([np for np in q_loc.nat_params.values()])
                mq = torch.cat([mp for mp in q_loc.mean_params.values()])
                log_aq = q_loc.log_a()
                enpp = torch.cat([q.std_params["loc"] / sigma ** 2,
                                 torch.ones_like(q.std_params["loc"])
                                 * (- 1 / (2 * sigma ** 2))])
                elog_ap = (q.std_params["loc"] ** 2
                           + q.std_params["scale"] ** 2) / (2 * sigma ** 2)
                elog_ap += torch.log(sigma)
                elog_ap = (elog_ap.sum()
                           - 0.5 * np.log(np.pi) * len(q.std_params["loc"]))
                kl_loc = (npq - enpp).dot(mq) - log_aq + elog_ap
                kl_loc /= len(x)

                # Sample θ from q(θ) and compute p(y | θ, x) for each θ
                ll = self.data_model.expected_log_likelihood(
                    batch, q_loc, self.config["num_elbo_samples"]).sum()
                ll /= len(x_batch)

                # Compute E_q[log t(θ)].
                # logt = self.t.eqlogt(q, self.config["num_elbo_samples"])
                # logt /= len(x)
                logt = torch.tensor(0.).to(self.config["device"])

                loss = kl + kl_loc - ll
                loss.backward()
                optimiser.step()

                # Keep track of quantities for current batch.
                epoch["elbo"] += -loss.item() / len(loader)
                epoch["kl"] += kl.item() / len(loader)
                epoch["kl_loc"] += kl_loc.item() / len(loader)
                epoch["ll"] += ll.item() / len(loader)
                epoch["logt"] += logt.item() / len(loader)

            epoch_iter.set_postfix(elbo=epoch["elbo"], kl=epoch["kl"],
                                   kl_loc=epoch["kl_loc"], ll=epoch["ll"])

            # Log progress for current epoch.
            training_metrics["elbo"].append(epoch["elbo"])
            training_metrics["kl"].append(epoch["kl"])
            training_metrics["kl_loc"].append(epoch["kl_loc"])
            training_metrics["ll"].append(epoch["ll"])

            if self.t is not None:
                training_metrics["logt"].append(epoch["logt"])

            # Check whether to stop early.
            stop_early = self.config["early_stopping"](
                scores=training_metrics,
                model=[q.non_trainable_copy(),
                       q_loc.non_trainable_copy()])

            if (i > 0 and i % self.config["print_epochs"] == 0) \
                    or i == (self.config["epochs"] - 1) or stop_early:
                # Update global posterior before evaluating performance.
                self.q = q_loc.non_trainable_copy()

                metrics = self.evaluate_performance({
                    "epochs": i,
                    "elbo": epoch["elbo"],
                    "kl": epoch["kl"],
                    "kl_loc": epoch["kl_loc"],
                    "ll": epoch["ll"],
                })

                # Report performance.
                report = ""
                for k, v in metrics.items():
                    report += f"{k}: {v:.3f} "
                    performance_metrics[k].append(v)

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
            q_new, q_loc_new = self.config["early_stopping"].best_model
        else:
            q_new = q.non_trainable_copy()
            q_loc_new = q_loc.non_trainable_copy()

        # Finished optimisation, can now update.
        self._can_update = True

        if self.t is not None:
            # Compute new local contribution from old distributions
            t_new = self.t.compute_refined_factor(
                q_new, q_old, damping=self.config["damping_factor"],
                valid_dist=self.config["valid_factors"],
                update_log_coeff=self.config["update_log_coeff"])

            return q_new, t_new, q_loc_new

        else:
            return q_new, None, q_loc_new


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
    def __init__(self, data, data_model, param_model, t=None, q_loc=None,
                 config=None, val_data=None):

        super().__init__(data, data_model, t, config, val_data)

        self.param_model = param_model
        self.data_model = self.model

        # q(θ_k).
        self.q = q_loc

    def update_q(self, q, init_q=None):
        """
        Computes a refined approximate posterior and the associated
        approximating likelihood term, and then updates the local approximate
        posterior.
        """

        # Find new local approximate posterior.
        q_loc = self.local_gradient_based_update(q_glob=q)

        # Find new global contribution.
        q_new, self.t = self.gradient_based_update(
            p=q, init_q=init_q)

        self.q = q_loc

        # Only return new q and approximate likelihood term. Server doesn't
        # need to know about local model.
        return q_new, self.t

    def local_gradient_based_update(self, q_glob):
        # Cannot update during optimisation.
        self._can_update = False

        # Compute cavity.
        q_cav = q_glob.non_trainable_copy()
        q_cav.nat_params = {k: v - self.t.nat_params[k]
                            for k, v in q_cav.nat_params.items()}

        # In the case of Gaussians, can compute p(θ) = ∫ q(ɸ) p(θ | ɸ) dɸ in
        # closed-form.
        # TODO: obviously this isn't always the case.
        p_loc_std_params = {
            "loc": q_cav.std_params["loc"],
            "scale": (
                q_cav.std_params["scale"] ** 2
                + self.param_model.outputsigma ** 2) ** 0.5
        }

        # TODO: which to use?
        p_loc = q_cav.create_new(std_params=p_loc_std_params,
                                 is_trainable=False)
        # p_loc = q_glob.non_trainable_copy()

        if self.q is None:
            # Initialise to q(ɸ).
            q_loc = q_glob.trainable_copy()
        else:
            q_loc = self.q.trainable_copy()

        # Parameters are those of q(θ).
        parameters = q_loc.parameters()

        # Reset optimiser.
        logging.info("Resetting optimiser")
        optimiser = getattr(torch.optim, self.config["optimiser"])(
            parameters, **self.config["optimiser_params"])
        lr_scheduler = getattr(torch.optim.lr_scheduler,
                               self.config["lr_scheduler"])(
            optimiser, **self.config["lr_scheduler_params"])

        # Set up data
        x = self.data["x"]
        y = self.data["y"]

        tensor_dataset = TensorDataset(x, y)
        loader = DataLoader(tensor_dataset,
                            batch_size=self.config["batch_size"],
                            shuffle=True)

        if self.config["device"] == "cuda":
            loader.pin_memory = True

        # Dict for logging optimisation progress.
        training_metrics = defaultdict(list)

        # Dict for logging performance progress.
        performance_metrics = defaultdict(list)

        # Reset early stopping.
        self.config["early_stopping"](None, q_loc.non_trainable_copy())

        # Gradient-based optimisation loop -- loop over epochs.
        epoch_iter = tqdm(range(self.config["epochs"]), desc="Epoch",
                          leave=True, disable=(not self.config["verbose"]))

        # for i in range(self.config["epochs"]):
        for i in epoch_iter:
            epoch = defaultdict(lambda: 0.)

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
                kl = q_loc.kl_divergence(p_loc, calc_log_ap=False).sum()
                kl /= len(x)

                # Sample θ from q(θ) and compute p(y | θ) for each θ
                ll = self.data_model.expected_log_likelihood(
                    batch, q_loc, self.config["num_elbo_samples"]).sum()
                ll /= len(x_batch)

                loss = kl - ll
                loss.backward()
                optimiser.step()

                # Keep track of quantities for current batch.
                epoch["elbo"] += -loss.item() / len(loader)
                epoch["kl"] += kl.item() / len(loader)
                epoch["ll"] += ll.item() / len(loader)

            epoch_iter.set_postfix(elbo=epoch["elbo"], kl=epoch["kl"],
                                   ll=epoch["ll"])

            # Log progress for current epoch.
            training_metrics["elbo"].append(epoch["elbo"])
            training_metrics["kl"].append(epoch["kl"])
            training_metrics["ll"].append(epoch["ll"])

            # Check whether to stop early.
            stop_early = self.config["early_stopping"](
                training_metrics, q_loc.non_trainable_copy())

            if (i > 0 and i % self.config["print_epochs"] == 0) \
                    or i == (self.config["epochs"] - 1) or stop_early:
                # Update global posterior before evaluating performance.
                self.q = q_loc.non_trainable_copy()

                metrics = self.evaluate_performance({
                    "epochs": i,
                    "elbo": epoch["elbo"],
                    "kl": epoch["kl"],
                    "ll": epoch["ll"],
                })

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
            q_loc_new = self.config["early_stopping"].best_model
        else:
            q_loc_new = q_loc.non_trainable_copy()

        # Finished optimisation, can now update.
        self._can_update = True

        return q_loc_new
