import logging
import time

import pytest
import torch

from npe_pf.npe_pf import TabPFN_Based_NPE_PF
from npe_pf.support_posterior import PosteriorSupport

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@pytest.mark.parametrize(
    "num_proposal_samples, sampling_method",
    [(10000, "rejection"), (200, "sir")],
)
def test_posterior_support(
    num_proposal_samples,
    sampling_method,
):
    # some constants
    num_simulations = 10000
    sampling_batch_size = 10000
    oversample_sir = 100
    num_samples_to_estimate_support = 20000
    allowed_false_negatives = 0.001

    # Define prior and simulator
    prior = torch.distributions.MultivariateNormal(torch.zeros(2), torch.eye(2))

    def simulator(thetas):
        return thetas + torch.randn_like(thetas)

    obs = torch.zeros(2)

    # Generate training data
    theta_train = prior.sample((num_simulations,))
    x_train = simulator(theta_train)

    # Initialize and train model
    posterior = TabPFN_Based_NPE_PF(
        prior=prior,
        filter_type="standardized_euclidean_filtering",
    )
    posterior.append_simulations(theta_train, x_train)

    posterior_support = PosteriorSupport(
        prior,
        posterior,
        obs,
        num_samples_to_estimate_support=num_samples_to_estimate_support,
        allowed_false_negatives=allowed_false_negatives,
        sampling_method=sampling_method,
        oversample_sir=oversample_sir,
        log_prob_kwargs={"mode": "ratio_based"},
    )

    start = time.perf_counter()

    proposal_samples = posterior_support.sample(
        (num_proposal_samples,),
        show_progress_bars=False,
        sampling_batch_size=sampling_batch_size,
    )
    logger.info(f"Time to sample from proposal: {time.perf_counter() - start}s")

    assert proposal_samples.shape == (num_proposal_samples, 2)
    assert not torch.isnan(proposal_samples).any()
    assert not torch.isinf(proposal_samples).any()
