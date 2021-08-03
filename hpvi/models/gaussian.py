import torch

from torch import distributions, nn
from pvi.models import Model


class GaussianNoise(Model, nn.Module):
    """
    Gaussian noise model.
    """

    conjugate_family = None

    def __init__(self, train_sigma=True, **kwargs):
        Model.__init__(self, **kwargs)
        nn.Module.__init__(self)

        self.train_sigma = train_sigma
        if self.train_sigma:
            self.register_parameter("log_outputsigma", nn.Parameter(
                torch.as_tensor(self.hyperparameters["outputsigma"]).log(),
                requires_grad=True))
        else:
            self.register_buffer("log_outputsigma", torch.as_tensor(
                self.hyperparameters["outputsigma"]).log())

        # Set ε after model is constructed.
        self.hyperparameters = self.hyperparameters

    def get_default_nat_params(self):
        """
        :return: A default set of natural parameters for the prior.
        """
        return {}

    def get_default_config(self):
        """
        :return: A default set of config for the model.
        """
        return {}

    @property
    def hyperparameters(self):
        return self._hyperparameters

    @hyperparameters.setter
    def hyperparameters(self, hyperparameters):
        self._hyperparameters = {**self._hyperparameters, **hyperparameters}

        if hasattr(self, "log_outputsigma"):
            if self.train_sigma:
                self.log_outputsigma.data = \
                    self.hyperparameters["outputsigma"].log()
            else:
                self.log_outputsigma = \
                    self.hyperparameters["outputsigma"].log()

    def get_default_hyperparameters(self):
        """
        :return: A default set of ε for the model.
        """
        return {
            "outputsigma": torch.tensor(1.),
        }

    def forward(self, x, q, **kwargs):
        """
        Returns the (approximate) predictive posterior.
        :param x: The input locations to make predictions at.
        :param q: The approximate posterior distribution q(θ).
        :return: ∫ p(y | θ, x) q(θ) dθ.
        """
        raise NotImplementedError

    def likelihood_forward(self, x, theta, **kwargs):
        """
        Returns the model's likelihood p(y | θ, x).
        :param x: The input locations to make predictions at.
        :param theta: The latent variables of the model.
        :return: p(y | θ, x)
        """
        return distributions.Normal(x, self.outputsigma)

    def conjugate_update(self, data, q, t=None):
        """
        If the likelihood is conjugate with q(θ), performs a conjugate update.
        :param data: The data to compute the conjugate update with.
        :param q: The current global posterior q(θ).
        :param t: The local factor t(θ).
        :return: The updated posterior, p(θ | data).
        """
        raise NotImplementedError

    @property
    def outputsigma(self):
        return self.log_outputsigma.exp()
