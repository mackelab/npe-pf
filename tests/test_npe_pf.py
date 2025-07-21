import logging
import time

import pytest
import torch

from npe_pf.npe_pf import (
    TabPFN_Based_NPE_PF,
    NPE_PF_Core,
    TabPFN_Based_Uncond_Estimator,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@pytest.mark.parametrize(
    "n_samples,n_test,n_posterior,feature_dim,obs_dim",
    [
        pytest.param(10, 1, 100, 2, 2, marks=pytest.mark.fast),
        pytest.param(10, 1, 100, 1, 3, marks=pytest.mark.fast),
        pytest.param(10, 1, 100, 3, 1, marks=pytest.mark.fast),
        pytest.param(100, 1, 1000, 4, 10),
        pytest.param(50, 1, 500, 3, 5),
        pytest.param(200, 1, 2000, 5, 15),
        pytest.param(
            100,
            2,
            1000,
            4,
            10,
            marks=[
                pytest.mark.xfail(reason="Multiple test samples not supported"),
                pytest.mark.fast,
            ],
        ),
    ],
)
def test_sampling_and_log_prob_base(
    n_samples,
    n_test,
    n_posterior,
    feature_dim,
    obs_dim,
):
    # training data
    prior = torch.distributions.Normal(
        torch.zeros(feature_dim), torch.ones(feature_dim)
    )
    theta = prior.sample((n_samples,))
    w = torch.randn(obs_dim, feature_dim)
    x = theta @ w.T + torch.randn(n_samples, obs_dim) * 0.1 + 1.0
    # test data
    theta_test = torch.randn(n_test, feature_dim)
    x_test = theta_test @ w.T + torch.randn(n_test, obs_dim) * 0.1 + 1.0

    inference_model = NPE_PF_Core(prior=prior)
    inference_model.append_simulations(theta, x)
    # sample
    theta_posterior = inference_model.sample(
        sample_shape=torch.Size([n_posterior, 1]),
        x=x_test,
        max_sampling_batch_size=10_000,
    )
    # log prob
    log_prob = inference_model.log_prob(theta_posterior, x_test)

    # Add assertions to verify results
    assert log_prob.shape == torch.Size([n_posterior])
    assert not torch.isnan(log_prob).any()
    assert not torch.isinf(log_prob).any()


@pytest.mark.parametrize(
    "n_samples,n_context,n_test,n_posterior,feature_dim,obs_dim,filter",
    [
        pytest.param(
            100_000,
            10,
            1,
            100,
            2,
            2,
            "random_filtering",
            marks=pytest.mark.fast,
        ),
        pytest.param(
            10,
            10_000,
            1,
            100,
            1,
            3,
            "standardized_euclidean_filtering",
            marks=pytest.mark.fast,
        ),
        pytest.param(
            100_000,
            10,
            1,
            100,
            3,
            1,
            "standardized_euclidean_filtering",
            marks=pytest.mark.fast,
        ),
    ],
)
def test_sampling_and_log_prob_NPE_PF(
    n_samples,
    n_context,
    n_test,
    n_posterior,
    feature_dim,
    obs_dim,
    filter,
):
    # training data
    prior = torch.distributions.Normal(
        torch.zeros(feature_dim), torch.ones(feature_dim)
    )
    theta = prior.sample((n_samples,))
    w = torch.randn(obs_dim, feature_dim)
    x = theta @ w.T + torch.randn(n_samples, obs_dim) * 0.1 + 1.0
    # test data
    theta_test = torch.randn(n_test, feature_dim)
    x_test = theta_test @ w.T + torch.randn(n_test, obs_dim) * 0.1 + 1.0

    inference_model = TabPFN_Based_NPE_PF(
        prior=prior,
        filter_type=filter,
        filter_context_size=n_context,
    )
    inference_model.append_simulations(theta, x)
    # sample
    theta_posterior = inference_model.sample(
        sample_shape=torch.Size([n_posterior, 1]),
        x=x_test,
        max_sampling_batch_size=10_000,
    )
    # log prob
    log_prob = inference_model.log_prob(theta_posterior, x_test)

    # Add assertions to verify results
    assert log_prob.shape == torch.Size([n_posterior])
    assert not torch.isnan(log_prob).any()
    assert not torch.isinf(log_prob).any()


@pytest.mark.parametrize(
    "n_train, n_context, grid_size, num_posterior_samples",
    [
        pytest.param(10_000, 10, 10, 5, marks=pytest.mark.fast),
        pytest.param(10_000, 10, 10, 2, marks=pytest.mark.fast),
        pytest.param(10_000, 2, 10, 5, marks=pytest.mark.fast),
        pytest.param(2000, 10_000, 100, 5000),
        pytest.param(10_000, 10_000, 100, 5000),
        pytest.param(1_000_000, 10_000, 100, 5000),
    ],
)
def test_ratio_based_log_prob(n_train, n_context, grid_size, num_posterior_samples):
    # Define prior and simulator
    prior = torch.distributions.Normal(torch.zeros(2), torch.ones(2))

    def simulator(thetas):
        return thetas + torch.randn_like(thetas)

    obs = torch.zeros(2)
    diff_obs = torch.ones(2)

    # Generate training data
    theta_train = prior.sample((n_train,))
    x_train = simulator(theta_train)

    # Initialize and train model
    inference_model = TabPFN_Based_NPE_PF(
        prior=prior,
        filter_type="standardized_euclidean_filtering",
        filter_context_size=n_context,
    )
    inference_model.append_simulations(theta_train, x_train)

    # Create grid for evaluation
    x = torch.linspace(-1, 1, grid_size)
    y = torch.linspace(-1, 1, grid_size)
    X, Y = torch.meshgrid(x, y, indexing="ij")
    grid_points = torch.stack([X.flatten(), Y.flatten()], dim=1)

    # Calculate log probabilities
    start = time.perf_counter()
    log_probs = inference_model.log_prob(
        theta=grid_points,
        x=obs,
        mode="ratio_based",
        **{"num_posterior_samples": num_posterior_samples},
    )
    logger.info(f"Time for first ratio_based log prob: {time.perf_counter() - start}s")

    start = time.perf_counter()
    log_probs_two = inference_model.log_prob(
        theta=grid_points,
        x=obs,
        mode="ratio_based",
        num_posterior_samples=num_posterior_samples,
    )
    logger.info(f"Time for second ratio_based log prob: {time.perf_counter() - start}s")

    # Change training data
    inference_model.append_simulations(
        theta_train[: n_train // 2], x_train[: n_train // 2]
    )

    start = time.perf_counter()
    log_probs_three = inference_model.log_prob(
        theta=grid_points,
        x=obs,
        mode="ratio_based",
        **{"num_posterior_samples": num_posterior_samples},
    )
    logger.info(f"Time for third ratio_based log prob: {time.perf_counter() - start}s")

    start = time.perf_counter()
    log_probs_four = inference_model.log_prob(
        theta=grid_points,
        x=obs,
        mode="ratio_based",
        num_posterior_samples=num_posterior_samples,
    )
    logger.info(f"Time for fourth ratio_based log prob: {time.perf_counter() - start}s")

    # Change obersvation
    start = time.perf_counter()
    log_probs_five = inference_model.log_prob(
        theta=grid_points,
        x=diff_obs,
        mode="ratio_based",
        **{"num_posterior_samples": num_posterior_samples},
    )
    logger.info(f"Time for fifth ratio_based log prob: {time.perf_counter() - start}s")

    start = time.perf_counter()
    log_probs_six = inference_model.log_prob(
        theta=grid_points,
        x=diff_obs,
        mode="ratio_based",
        **{"num_posterior_samples": num_posterior_samples},
    )
    logger.info(f"Time for sixth ratio_based log prob: {time.perf_counter() - start}s")

    # Add assertions to verify results
    assert log_probs.shape == (grid_size * grid_size,)
    assert not torch.isnan(log_probs).any()
    assert not torch.isinf(log_probs).any()

    assert log_probs_two.shape == (grid_size * grid_size,)
    assert not torch.isnan(log_probs_two).any()
    assert not torch.isinf(log_probs_two).any()

    assert log_probs_three.shape == (grid_size * grid_size,)
    assert not torch.isnan(log_probs_three).any()
    assert not torch.isinf(log_probs_three).any()

    assert log_probs_four.shape == (grid_size * grid_size,)
    assert not torch.isnan(log_probs_four).any()
    assert not torch.isinf(log_probs_four).any()

    assert log_probs_five.shape == (grid_size * grid_size,)
    assert not torch.isnan(log_probs_five).any()
    assert not torch.isinf(log_probs_five).any()

    assert log_probs_six.shape == (grid_size * grid_size,)
    assert not torch.isnan(log_probs_six).any()
    assert not torch.isinf(log_probs_six).any()


# These test can fail if one gets very unlucky with the random seed
# and there is a cluster with only a single sample
# TODO seeding
@pytest.mark.parametrize(
    "n_samples,n_posterior,feature_dim,num_clusters",
    [
        pytest.param(10, 100, 1, 1, marks=pytest.mark.fast),
        pytest.param(10, 100, 3, 1, marks=pytest.mark.fast),
        pytest.param(1000, 1000, 4, 5),
        pytest.param(
            50, 500, 3, 50, marks=pytest.mark.xfail(reason="Too small clusters")
        ),
        pytest.param(2000, 12000, 5, 1),
        pytest.param(2000, 12000, 5, 3),
    ],
)
def test_sampling_and_log_prob_unconditional_TabPFN(
    n_samples,
    n_posterior,
    feature_dim,
    num_clusters,
):
    # training data
    prior = torch.distributions.Normal(
        torch.zeros(feature_dim), torch.ones(feature_dim)
    )
    theta = prior.sample((n_samples,))

    inference_model = TabPFN_Based_Uncond_Estimator(num_clusters=num_clusters)
    inference_model.append_simulations(theta)
    # sample
    theta_posterior = inference_model.sample(
        sample_shape=torch.Size([n_posterior, 1]),
        max_sampling_batch_size=10_000,
    )
    # log prob
    log_prob = inference_model.log_prob(theta_posterior)

    # Add assertions to verify results
    assert log_prob.shape == torch.Size([n_posterior])
    assert not torch.isnan(log_prob).any()
    assert not torch.isinf(log_prob).any()
