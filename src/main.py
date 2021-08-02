from environment import ContextualEnvironment
from policies import KLUCBSegmentPolicy, RandomPolicy, ExploreThenCommitSegmentPolicy, EpsilonGreedySegmentPolicy, TSPolicy, TSSegmentPolicy, LinearTSPolicy
import argparse
import json
import logging
import numpy as np
import pandas as pd
import time

# List of implemented policies
def set_policies(policies_name, user_segment, user_features, n_playlists):
    # Please see section 3.3 of RecSys paper for a description of policies
    POLICIES_SETTINGS = {
        # All policies from [Bendada et al., 2020]
        'random' : RandomPolicy(n_playlists),
        'etc-seg-explore' : ExploreThenCommitSegmentPolicy(user_segment, n_playlists, min_n = 100, cascade_model = True),
        'etc-seg-exploit' : ExploreThenCommitSegmentPolicy(user_segment, n_playlists, min_n = 20, cascade_model = True),
        'epsilon-greedy-explore' : EpsilonGreedySegmentPolicy(user_segment, n_playlists, epsilon = 0.1, cascade_model = True),
        'epsilon-greedy-exploit' : EpsilonGreedySegmentPolicy(user_segment, n_playlists, epsilon = 0.01, cascade_model = True),
        'kl-ucb-seg' : KLUCBSegmentPolicy(user_segment, n_playlists, cascade_model = True),
        'ts-seg-naive' : TSSegmentPolicy(user_segment, n_playlists, alpha_zero = 1, beta_zero = 1, cascade_model = True),
        'ts-seg-pessimistic' : TSSegmentPolicy(user_segment, n_playlists, alpha_zero = 1, beta_zero = 99, cascade_model = True),
        'ts-lin-naive' : LinearTSPolicy(user_features, n_playlists, bias = 0.0, cascade_model = True),
        'ts-lin-pessimistic' : LinearTSPolicy(user_features, n_playlists, bias = -5.0, cascade_model = True, l2_reg = 1),
        # Versions of epsilon-greedy-explore and ts-seg-pessimistic WITHOUT cascade model
        'epsilon-greedy-explore-no-cascade' : EpsilonGreedySegmentPolicy(user_segment, n_playlists, epsilon = 0.1, cascade_model = False),
        'ts-seg-pessimistic-no-cascade' : TSSegmentPolicy(user_segment, n_playlists, alpha_zero = 1, beta_zero = 99, cascade_model = False),

        # Extensions
        # Multi-Arm Thomspon Sampling
        'ts-mab-pessimistic' : TSPolicy(n_playlists, alpha_zero = 1, beta_zero = 99, cascade_model = True),        
        
        # Algorithms in the manuscript for RecSys 2021:
        'ts-lin-pessimistic-reg-2000' : LinearTSPolicy(user_features, n_playlists, bias = -5., cascade_model = True, l2_reg = 2000),
        'ts-lin-pessimistic-reg-2000-shuffle-6' : LinearTSPolicy(user_features, n_playlists, bias = -5., cascade_model = True, l2_reg = 2000, shuffle_K = 6),
        'ts-lin-pessimistic-reg-2000-personalised-shuffle-99': LinearTSPolicy(user_features, n_playlists, bias = -5., cascade_model = True, l2_reg = 2000, shuffle_K = -1, epsilon=0.010),
        'ts-lin-pessimistic-reg-2000-personalised-shuffle-975': LinearTSPolicy(user_features, n_playlists, bias = -5., cascade_model = True, l2_reg = 2000, shuffle_K = -1, epsilon=0.025),
        'ts-lin-pessimistic-reg-2000-personalised-shuffle-95': LinearTSPolicy(user_features, n_playlists, bias = -5., cascade_model = True, l2_reg = 2000, shuffle_K = -1, epsilon=0.050),
        'ts-lin-pessimistic-reg-2000-personalised-shuffle-90': LinearTSPolicy(user_features, n_playlists, bias = -5., cascade_model = True, l2_reg = 2000, shuffle_K = -1, epsilon=0.1),
    }

    return [POLICIES_SETTINGS[name] for name in policies_name]


if __name__ == "__main__":

    # Arguments

    parser = argparse.ArgumentParser()
    parser.add_argument("--users_path", type = str, default = "data/user_features.csv", required = False,
                        help = "Path to user features file")
    parser.add_argument("--playlists_path", type = str, default = "data/playlist_features.csv", required = False,
                        help = "Path to playlist features file")
    parser.add_argument("--output_path", type = str, default = "results.json", required = False,
                        help = "Path to json file to save regret values")
    parser.add_argument("--policies", type = str, default = "random,ts-seg-naive", required = False,
                        help = "Bandit algorithms to evaluate, separated by commas")
    parser.add_argument("--n_recos", type = int, default = 12, required = False,
                        help = "Number of slots L in the carousel i.e. number of recommendations to provide")
    parser.add_argument("--l_init", type = int, default = 1, required = False,
                        help = "Number of slots L_init initially visible in the carousel")
    parser.add_argument("--n_users_per_round", type = int, default = 20000, required = False,
                        help = "Number of users randomly selected (with replacement) per round")
    parser.add_argument("--n_rounds", type = int, default = 100, required = False,
                        help = "Number of simulated rounds")
    parser.add_argument("--print_every", type = int, default = 10, required = False,
                        help = "Print cumulative regrets every 'print_every' round")
    parser.add_argument("--gamma", type = float, default = 0.9, required = False,
                        help = "Patience parameter for the cascade user model")

    args = parser.parse_args()

    logging.basicConfig(level = logging.INFO)
    logger = logging.getLogger(__name__)

    if args.l_init > args.n_recos:
        raise ValueError('l_init is larger than n_recos')


    # Data Loading and Preprocessing steps

    logger.info("LOADING DATA")
    logger.info("Loading playlist data")
    playlists_df = pd.read_csv(args.playlists_path)

    logger.info("Loading user data\n \n")
    users_df = pd.read_csv(args.users_path)

    n_users = len(users_df)
    n_playlists = len(playlists_df)
    n_recos = args.n_recos
    print_every = args.print_every

    user_features = np.array(users_df.drop(["segment"], axis = 1))
    user_features = np.concatenate([user_features, np.ones((n_users,1))], axis = 1)
    playlist_features = np.array(playlists_df)

    user_segment = np.array(users_df.segment)

    logger.info("SETTING UP SIMULATION ENVIRONMENT")
    logger.info("for %d users, %d playlists, %d recommendations per carousel \n \n" % (n_users, n_playlists, n_recos))

    cont_env = ContextualEnvironment(user_features, playlist_features, user_segment, n_recos, gamma=args.gamma)

    logger.info("SETTING UP POLICIES")
    logger.info("Policies to evaluate: %s \n \n" % (args.policies))

    policies_name = args.policies.split(",")
    policies = set_policies(policies_name, user_segment, user_features, n_playlists)
    n_policies = len(policies)
    n_users_per_round = args.n_users_per_round
    n_rounds = args.n_rounds
    overall_rewards = np.zeros((n_policies, n_rounds))
    overall_expected_clicks = np.zeros((n_policies, n_rounds))
    overall_hellinger = np.zeros((n_policies, n_rounds))
    overall_kullback_leibler = np.zeros((n_policies, n_rounds))
    overall_total_variation = np.zeros((n_policies, n_rounds))

    overall_optimal_reward = np.zeros(n_rounds)

    # Simulations for Top-n_recos carousel-based playlist recommendations
    logger.info("STARTING SIMULATIONS")
    logger.info("for %d rounds, with %d users per round (randomly drawn with replacement)\n \n" % (n_rounds, n_users_per_round))
    start_time = time.time()
    for i in range(n_rounds):
        # Select batch of n_users_per_round users
        user_ids = np.random.choice(range(n_users), n_users_per_round)
        overall_optimal_reward[i] = np.take(cont_env.optimal_reward, user_ids).sum()
        # Iterate over all policies
        for j in range(n_policies):
            print(f'---------- {policies_name[j]} ----------')
            # Compute n_recos recommendations
            recos = policies[j].recommend_to_users_batch(user_ids, args.n_recos, args.l_init)
            # Compute rewards, Expected Exposure Disparity (EE-D), Expected Exposure Relevance (EE-R), Expected Clicks and Expected Exposure
            rewards, E_clicks_user, H, KL, TV = cont_env.simulate_batch_users_reward(batch_user_ids=user_ids, batch_recos=recos)

            # Update policy based on rewards
            policies[j].update_policy(user_ids, recos, rewards, args.l_init)
            overall_rewards[j,i] = rewards.sum()
            # overall_exposure_relevance[j,i] = exposure_relevance.sum()
            overall_expected_clicks[j,i] = E_clicks_user.mean()
            overall_hellinger[j,i] = H
            overall_kullback_leibler[j,i] = KL
            overall_total_variation[j,i] = TV
        # Print info
        if i == 0 or (i+1) % print_every == 0 or i+1 == n_rounds:
            logger.info("Round: %d/%d. Elapsed time: %f sec." % (i+1, n_rounds, time.time() - start_time))
            logger.info("Cumulative regrets: \n%s \n" % "\n".join(["	%s : %s" % (policies_name[j], str(np.sum(overall_optimal_reward - overall_rewards[j]))) for j in range(n_policies)]))
            logger.info("Optimal reward : {0}".format(overall_optimal_reward[i] / n_users_per_round))
            logger.info("Expected clicks: \n%s \n" % "\n".join(["	%s : %s" % (policies_name[j], str(overall_expected_clicks[j,i])) for j in range(n_policies)]))
            logger.info("Hellinger: \n%s \n" % "\n".join(["	%s : %s" % (policies_name[j], str(overall_hellinger[j,i])) for j in range(n_policies)]))
            logger.info("KL: \n%s \n" % "\n".join(["	%s : %s" % (policies_name[j], str(overall_kullback_leibler[j,i])) for j in range(n_policies)]))
            logger.info("TV: \n%s \n" % "\n".join(["	%s : %s" % (policies_name[j], str(overall_total_variation[j,i])) for j in range(n_policies)]))

    # Save results
    logger.info("Saving output in %s" % args.output_path)
    cumulative_regrets = {policies_name[j] : list(np.cumsum(overall_optimal_reward - overall_rewards[j])) for j in range(n_policies)}
    with open(args.output_path, 'w') as fp:
        json.dump(cumulative_regrets, fp)

    logger.info("Saving expected clicks in %s" % args.output_path)
    expected_clicks = {policies_name[j] : list(overall_expected_clicks[j]) for j in range(n_policies)}
    with open('expected_clicks_' + args.output_path, 'w') as fp:
        json.dump(expected_clicks, fp)

    logger.info("Saving Hellinger distances in %s" % args.output_path)
    expected_clicks = {policies_name[j] : list(overall_hellinger[j]) for j in range(n_policies)}
    with open('hellinger_' + args.output_path, 'w') as fp:
        json.dump(expected_clicks, fp)

    logger.info("Saving KL-Divergences in %s" % args.output_path)
    expected_clicks = {policies_name[j] : list(overall_kullback_leibler[j]) for j in range(n_policies)}
    with open('kullback_leibler_' + args.output_path, 'w') as fp:
        json.dump(expected_clicks, fp)

    logger.info("Saving Total Variation in %s" % args.output_path)
    expected_clicks = {policies_name[j] : list(overall_total_variation[j]) for j in range(n_policies)}
    with open('total_variation_' + args.output_path, 'w') as fp:
        json.dump(expected_clicks, fp)
