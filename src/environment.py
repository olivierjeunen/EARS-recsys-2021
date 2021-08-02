from functools import reduce
from scipy.sparse import csr_matrix
from scipy.special import expit
import numpy as np

import json

# This class uses user and playlist features datasets to simulate users responses to a list of recommendations
class ContextualEnvironment():
    def __init__(self, user_features, playlist_features, user_segment, n_recos, gamma = 1.0): # Originally, Gamma = 1.0
        self.user_features = user_features
        self.playlist_features = playlist_features
        self.user_segment = user_segment
        self.n_recos = n_recos
        self.gamma = gamma # Patience for ERR -- P(E = 1 | rank = 1) = \gamma

        # Compute and store the true matrix storing P(R = 1 | u, i)
        self.P_relevant = expit(self.user_features.dot(self.playlist_features.T))

        # Compute optimal recommendations for every user (with respect to P(R = 1))
        self.optimal_recos = np.argsort(-self.P_relevant)[:, :self.n_recos]

        # Compute expected number of clicks given optimal recommendations (with respect to P(R = 1))
        self.optimal_reward, self.optimal_H, self.optimal_KL, self.optimal_TV = self.compute_expected_clicks(np.arange(self.P_relevant.shape[0]), self.optimal_recos, dump=True)

        print('Optimal E_clicks: \t', self.optimal_reward.mean())

    # Computes expected reward for each user given their recommendations
    def compute_expected_clicks(self, users, recommendations, dump=False):
        # If the batch of users is not equal to all users -- first filter the correct rows
        # P(R = 1) for sampled users and all items
        P_relevant_recos = np.take(self.P_relevant, users, axis = 0) if len(users) < self.P_relevant.shape[0] else self.P_relevant

        # Extract only columns for recommended items from P(R = 1)
        # P(R = 1) for sampled users and recommended items
        P_relevant_recos_items = np.take_along_axis(P_relevant_recos, recommendations, axis=1)
   
        # Compute P(C = 1) = P(R = 1 | P(E = 1)) P(E = 1)
        # P_click = P_relevant_recos_items * P_exposed_patience

        # Now compute the exposure probability conditioned on previously shown items
        # P(E = 1 | k) = \Prod_{j < k} (1 - P(C = 1 | j))
        P_exposure = np.cumprod((1 - P_relevant_recos_items), axis=-1)[:,:-1]
        # P(E = 1 | k = 1) = 1
        P_exposure = np.hstack((np.ones(len(users)).reshape(-1,1), P_exposure))

        # Discount with Gamma^{k} at rank k
        P_exposure = np.multiply(np.cumprod(self.gamma*np.ones_like(P_exposure[0,:])), P_exposure)

        indptr = np.arange(len(users) + 1) * recommendations.shape[1]
        indices = recommendations.flatten()
        data = P_exposure.flatten()
        full_P_exposure = csr_matrix((data, indices, indptr), shape = (len(users), self.playlist_features.shape[0]))

        # Multiply with P_click to get P(C = 1 | rank = k, \sigma)
        # In this formulation -- \sigma is the full ranking
        # P_click *= P_exposure
        P_click = np.multiply(P_relevant_recos_items, P_exposure)

        # P(A=a|R=1) for all items
        # P(A=a|C=1) for all items, i.e. explicitly set exposure to zero
        normalized_P_C = P_click/P_click.sum(axis=1, keepdims=True)
        indptr = np.arange(len(users) + 1) * recommendations.shape[1]
        indices = recommendations.flatten()
        data = normalized_P_C.flatten()
        normalized_full_P_click = csr_matrix((data, indices, indptr), shape = (len(users), self.playlist_features.shape[0]))
        
        normalized_P_R = P_relevant_recos/P_relevant_recos.sum(axis=1, keepdims=True)

        # Compute Hellinger distance
        H = (np.linalg.norm(np.sqrt(normalized_full_P_click)-np.sqrt(normalized_P_R), axis=1, ord=2) / np.sqrt(2)) # .mean()
        H_std = np.std(H)
        H = np.mean(H)
        # Compute divergence between distributions for every user and take the average
        print('\tMicro Hellinger(Relevance||Clicks):', H)

        ## Only look at the top-K recommendations
        normalized_P_R = np.take_along_axis(normalized_P_R, self.optimal_recos[users,:], axis=1)
        normalized_P_R /= normalized_P_R.sum(axis=1, keepdims=True)

        normalized_full_P_click = np.take_along_axis(normalized_full_P_click, self.optimal_recos[users,:], axis=1)
        normalized_full_P_click /= normalized_full_P_click.sum(axis=1)
        np.nan_to_num(normalized_full_P_click, copy=False)

        # Compute divergence between average distributions
        R = normalized_P_R.mean(axis=0).ravel()
        C = normalized_full_P_click.mean(axis=0).A1 

        print('R:', R)
        print('C:', C)

        H = (np.linalg.norm(np.sqrt(R)-np.sqrt(C), ord=2) / np.sqrt(2))
        print('\tMacro Hellinger(Relevance||Clicks):', H)

        KL = np.sum(R*np.log(R/C))
        print('\tMacro D-KL(Relevance||Clicks):', KL) 

        TV = np.linalg.norm(R-C,ord=1) / 2.0
        print('\tMacro TV(Relevance||Clicks):', TV) 

        # Expected number of clicks per user
        E_clicks_user = P_click.sum(axis = 1)

        print('\tVariance E[C|U]:', E_clicks_user.var())

        return E_clicks_user, H, KL, TV
    
    # Given a list of users and their respective list of recos (each of size self.n_recos), computes
    # corresponding simulated reward and Expected Exposure Disparity (EE-D)
    def simulate_batch_users_reward(self, batch_user_ids, batch_recos):
        
        # First, compute probability of streaming each reco and draw rewards accordingly
        n_users = len(batch_user_ids)
        n = batch_recos.shape[1]

        # P(E = 1 | rank = k) irrespective of previous rewards
        # i.e. \gamma^{k} 
        P_exposed_patience = np.cumprod(np.ones(n) * self.gamma)

        # Extract only rows for sampled users from P(R = 1)
        P_relevant_recos = np.take(self.P_relevant, batch_user_ids, axis = 0)

        # Extract only columns for recommended items from P(R = 1)
        P_relevant_recos = np.take_along_axis(P_relevant_recos, batch_recos, axis=1)

        # Compute P(C = 1) = P(R = 1 | E = 1) P(E = 1)
        P_click = P_relevant_recos * P_exposed_patience

        # Simulate rewards
        rewards = np.zeros((n_users, n))
        i = 0
        rewards_uncascaded = np.random.binomial(1, P_click) # drawing rewards from probabilities
        positive_rewards = set()

        # Then, for each user, positive rewards after the first one are set to 0 (and playlists as "unseen" subsequently)
        # to imitate a cascading browsing behavior
        # (nonetheless, users can be drawn several times in the batch of a same round ; therefore, each user
        # can have several positive rewards - i.e. stream several playlists - in a same round, consistently with
        # the multiple-plays framework from the paper)
        nz = rewards_uncascaded.nonzero()
        for i in range(len(nz[0])):
            if nz[0][i] not in positive_rewards:
                rewards[nz[0][i]][nz[1][i]] = 1
                positive_rewards.add(nz[0][i])

        # Compute Expected number of clicks
        E_clicks_user, H, KL, TV = self.compute_expected_clicks(batch_user_ids, batch_recos)

        return rewards, E_clicks_user, H, KL, TV 
