import logging
from typing import Callable, Mapping

import torch
from sbi import inference as inference
from torch.distributions import Distribution

from npe_pf.npe_pf import TabPFN_Based_NPE_PF
from npe_pf.support_posterior import PosteriorSupport

log = logging.getLogger(__name__)


def run_tsnpe_pf(
    simulator: Callable,
    prior: Distribution,
    observation: torch.Tensor,
    num_simulations: int = 10_000,
    num_rounds: int = 10,
    proposal_batch_size: int = 1000,
    simulation_batch_size: int = 1000,
    num_samples_to_estimate_support: int = 10_000,
    allowed_false_negatives: float = 0.0001,
    context_size: int = 10_000,
    log_prob_mode: str = "ratio_based",
    sampling_method: str = "rejection",
    max_iter_rejection: int = 1000,
    oversample_sir: int = 100,
    filtering: str = "no_filtering",
    regressor_init_kwargs: Mapping = {},
    classifier_init_kwargs: Mapping = {},
):
    """Runs TSNPE-PF with the given parameters.
    Args:
        simulator: Simulator function, which takes theta and returns x
        prior: Prior distribution
        observation: Observation tensor
        num_simulations: Number of simulations per round
        num_rounds: Number of rounds
        proposal_batch_size: Batch size for proposal sampling
        simulation_batch_size: Batch size for simulator
        num_samples_to_estimate_support: Number of samples to estimate support
        allowed_false_negatives: Allowed false negatives, epsilon
        context_size: Context size for filtering
        log_prob_mode: Log probability mode, ["ratio_based", "autoregressive"]
        sampling_method: Sampling method, ["rejection", "sir"]
        max_iter_rejection: Maximum iterations for rejection sampling
        oversample_sir: Oversampling factor for SIR
        filtering: Filtering method, ["no_filtering", "latest_filtering", "random_filtering", "standardized_euclidean"]
        regressor_init_kwargs: Keyword arguments for regressor initialization
        classifier_init_kwargs: Keyword arguments for classifier initialization
    Returns:
        posterior: `TabPFN_Based_NPE_PF` object
    """

    if num_rounds == 1:
        log.info(f"Running NPE_PF")
        num_simulations_per_round = num_simulations
    else:
        log.info(f"Running TSNPE_PF")
        num_simulations_per_round = num_simulations // num_rounds

    log.info(f"Number of simulations per round: {num_simulations_per_round}")

    if simulation_batch_size > num_simulations_per_round:
        simulation_batch_size = num_simulations_per_round
        log.warning("Reduced simulation_batch_size to num_simulation_per_round")

    tabpfn_posterior = TabPFN_Based_NPE_PF(
        prior=prior,
        regressor_init_kwargs=regressor_init_kwargs,
        classifier_init_kwargs=classifier_init_kwargs,
        filter_type=filtering,
        filter_context_size=context_size,
    )
    proposal = prior

    theta_per_round = []
    x_per_round = []
    for round_num in range(num_rounds):
        log.info(f"Round {round_num + 1}/{num_rounds}")
        # TODO the proposal in here never gets the batch size for proposal.sample, currently default 10k
        # Same for other sampling arguments like progress bar etc.
        # The progress bar of the simulator is hard coded.
        log.info("Drawing from proposal and simulating!")
        theta, x = inference.simulate_for_sbi(
            simulator,
            proposal,
            num_simulations=num_simulations_per_round,
            simulation_batch_size=simulation_batch_size,
        )

        theta_per_round.append(theta)  # append to the end
        x_per_round.append(x)

        theta_cat = torch.cat(theta_per_round, dim=0)
        x_cat = torch.cat(x_per_round, dim=0)

        log.info("Appending simulations and initializing restricted proposal!")
        posterior = tabpfn_posterior.append_simulations(theta_cat, x_cat)

        if round_num == num_rounds - 1:
            break

        posterior_support = PosteriorSupport(
            prior,
            posterior,
            obs=observation,
            num_samples_to_estimate_support=num_samples_to_estimate_support,
            batch_size_for_estimate_support=proposal_batch_size,
            allowed_false_negatives=allowed_false_negatives,
            sampling_method=sampling_method,
            max_iter_rejection=max_iter_rejection,
            oversample_sir=oversample_sir,
            log_prob_kwargs={"mode": log_prob_mode},
        )
        proposal = posterior_support

    return posterior
