"""
SYNAPSE - Self-Organizing Hierarchical Pattern Recognition Engine
Author: Mehdi Abdu Mohammed
Version: 2.0.0
"""

import numpy as np
import random
import time


class Synapse:
    """
    SYNAPSE: Self-Organizing Hierarchical Pattern Recognition Engine

    Unified mathematical framework:
    - Robbins-Monro Stochastic Approximation
    - Manifold Regularization (Graph Laplacian)
    - Adaptive Resonance Theory
    - Dirichlet Process Node Creation
    - Birth-Death Edge Dynamics
    - MDL-based Pruning
    - Johnson-Lindenstrauss Random Projection
    """

    VERSION = "2.0.0"
    AUTHOR = "Mehdi Abdu Mohammed"

    def __init__(self, dim=2, tau0=1.3, gamma=0.1, beta=0.05, delta=0.01,
                 alpha_dp=0.1, learning_rate=0.1, context_dim=10,
                 usage_decay=0.98, node_cap=None):
        """
        Initialize SYNAPSE.

        Parameters:
            dim         : Input dimensionality
            tau0        : Base resonance threshold
            gamma       : Manifold regularization strength
            beta        : Edge birth rate
            delta       : Edge decay rate
            alpha_dp    : Dirichlet process concentration
            learning_rate: Robbins-Monro step size
            context_dim : Random projection dimension (Johnson-Lindenstrauss)
            usage_decay : Usage decay factor (0-1), lower = faster forgetting
            node_cap    : Maximum number of nodes (None = unlimited)
        """
        self.dim = dim
        self.tau0 = tau0
        self.gamma = gamma
        self.beta = beta
        self.delta = delta
        self.alpha_dp = alpha_dp
        self.eta = learning_rate
        self.context_dim = context_dim
        self.usage_decay = usage_decay
        self.node_cap = node_cap

        self.metadata = {
            'version': self.VERSION,
            'author': self.AUTHOR,
            'created': time.time(),
            'params': {
                'dim': dim, 'tau0': tau0, 'gamma': gamma,
                'beta': beta, 'delta': delta, 'alpha_dp': alpha_dp,
                'learning_rate': learning_rate, 'usage_decay': usage_decay,
                'node_cap': node_cap
            }
        }

        # System state
        self.nodes = []             # Prototype vectors w_i
        self.contexts = []          # Context embeddings c_i
        self.edges = {}             # Edge weights e_ij
        self.usage = []             # Usage counters
        self.node_ages = []         # Node ages
        self.local_variances = []   # Local variance sigma_i
        self.t = 0                  # Time step

        # Random projection matrix (Johnson-Lindenstrauss)
        self.R = np.random.randn(dim, context_dim) / np.sqrt(context_dim)

    # -------------------------------------------------------------------------
    # Core mathematical methods
    # -------------------------------------------------------------------------

    def _validate_input(self, x):
        """Validate and normalize input."""
        x = np.asarray(x).flatten()
        if x.shape[0] != self.dim:
            if x.shape[0] > self.dim:
                x = x[:self.dim]
            else:
                x = np.pad(x, (0, self.dim - x.shape[0]))
        return x.astype(np.float32)

    def distance(self, x, w):
        """Euclidean distance (L2 norm)."""
        try:
            return float(np.linalg.norm(x - w))
        except:
            return float('inf')

    def resonance_threshold(self, node_idx):
        """
        Adaptive resonance threshold.
        tau_i = tau_0 * (1 + lambda * sigma_i / mu_i)^{-1}
        """
        if len(self.nodes) == 0 or node_idx >= len(self.nodes):
            return self.tau0

        usage_i = self.usage[node_idx] if node_idx < len(self.usage) else 1.0
        confidence = min(1.0, usage_i / 50)
        lambda_factor = 0.5 * (1 - confidence)
        sigma_i = self.local_variances[node_idx] if node_idx < len(self.local_variances) else 1.0
        adaptive_factor = 1.0 / (1.0 + lambda_factor * sigma_i)
        return self.tau0 * adaptive_factor

    def context_projection(self, x):
        """Random projection for context learning (Johnson-Lindenstrauss)."""
        return np.dot(x, self.R)

    def _ensure_edge_dict(self, idx):
        if idx not in self.edges:
            self.edges[idx] = {}

    def edge_birth_death(self, i, j, co_activated):
        """
        Edge dynamics: de_ij/dt = beta * CoAct_ij * (1-e_ij) - delta * e_ij
        Stationary: e_ij* = beta*CoAct / (beta*CoAct + delta)
        """
        self._ensure_edge_dict(i)
        self._ensure_edge_dict(j)

        current_edge = self.edges[i].get(j, 0)

        if co_activated:
            new_edge = current_edge + self.beta * (1 - current_edge)
        else:
            new_edge = current_edge * (1 - self.delta)

        stationary = (self.beta * int(co_activated)) / (self.beta * int(co_activated) + self.delta + 1e-10)
        new_edge = 0.5 * new_edge + 0.5 * stationary

        self.edges[i][j] = new_edge
        self.edges[j][i] = new_edge

        if new_edge < 0.01:
            self.edges[i].pop(j, None)
            self.edges[j].pop(i, None)

    def dirichlet_process_node_creation(self, x):
        """
        Dirichlet process: P(new) = alpha_dp / (n + alpha_dp)
        """
        n = len(self.nodes)
        if n == 0:
            return True

        if self.node_cap and n >= self.node_cap:
            return False

        p_new = self.alpha_dp / (n + self.alpha_dp)
        distances = [self.distance(x, w) for w in self.nodes]
        thresholds = [self.resonance_threshold(i) for i in range(n)]
        no_resonance = all(d > t for d, t in zip(distances, thresholds))

        return (random.random() < p_new) or no_resonance

    def stochastic_approximation_update(self, node_idx, x, neighbors):
        """
        Robbins-Monro: theta_{t+1} = theta_t + eta_t * H(theta_t, x_t)
        H = (x - w_i) - gamma * sum_j e_ij * (w_i - w_j)
        """
        w = self.nodes[node_idx].copy()

        # Laplacian regularization term
        laplacian_term = np.zeros_like(w)
        if neighbors:
            for j in neighbors:
                if j < len(self.nodes):
                    edge_weight = self.edges.get(node_idx, {}).get(j, 0)
                    laplacian_term += edge_weight * (w - self.nodes[j])

        h_theta = (x - w) - self.gamma * laplacian_term
        eta_t = self.eta / (1 + 0.01 * self.t)
        self.nodes[node_idx] += eta_t * h_theta

        self.usage[node_idx] += 1.0
        self.node_ages[node_idx] += 1

        if node_idx < len(self.local_variances):
            error = float(np.linalg.norm(x - self.nodes[node_idx]) ** 2)
            self.local_variances[node_idx] = (
                0.9 * self.local_variances[node_idx] + 0.1 * error
            )

    def _add_node(self, x):
        """Add a new node at position x."""
        self.nodes.append(x.copy())
        self.contexts.append(self.context_projection(x))
        self.usage.append(1.0)
        self.node_ages.append(0)
        self.local_variances.append(1.0)
        new_idx = len(self.nodes) - 1
        self._ensure_edge_dict(new_idx)
        return new_idx

    def mdl_prune(self, min_usage=5.0, cost_threshold=1000.0):
        """
        MDL-based pruning.
        Minimizes: L = log(n)*dim + variance*usage
        """
        if len(self.nodes) == 0:
            return

        keep_indices = []
        for i in range(len(self.nodes)):
            param_cost = np.log(max(len(self.nodes), 1)) * self.dim
            data_cost = self.local_variances[i] * self.usage[i] if i < len(self.usage) else 0
            total_cost = param_cost + data_cost

            if self.usage[i] >= min_usage and total_cost < cost_threshold:
                keep_indices.append(i)

        if not keep_indices:
            most_used = int(np.argmax(self.usage)) if self.usage else 0
            keep_indices = [most_used]

        if len(keep_indices) < len(self.nodes):
            self.nodes = [self.nodes[i] for i in keep_indices]
            self.contexts = [self.contexts[i] for i in keep_indices]
            self.usage = [self.usage[i] for i in keep_indices]
            self.node_ages = [self.node_ages[i] for i in keep_indices]
            self.local_variances = [self.local_variances[i] for i in keep_indices]

            new_edges = {}
            idx_map = {old: new for new, old in enumerate(keep_indices)}
            for old_i in keep_indices:
                new_i = idx_map[old_i]
                new_edges[new_i] = {}
                for old_j, weight in self.edges.get(old_i, {}).items():
                    if old_j in idx_map:
                        new_edges[new_i][idx_map[old_j]] = weight
            self.edges = new_edges

    # -------------------------------------------------------------------------
    # Main step
    # -------------------------------------------------------------------------

    def step(self, x):
        """
        Process one data point. Core SYNAPSE update.
        Returns index of winning/created node.
        """
        x = self._validate_input(x)
        self.t += 1

        # Decay usage
        for i in range(len(self.usage)):
            self.usage[i] *= self.usage_decay

        # Dirichlet process node creation
        if self.dirichlet_process_node_creation(x):
            new_idx = self._add_node(x)
            if len(self.nodes) > 1:
                distances = [self.distance(x, w) for w in self.nodes[:-1]]
                nearest = np.argsort(distances)[:3]
                for j in nearest:
                    self.edge_birth_death(new_idx, j, co_activated=True)
            return new_idx

        # Find resonant nodes
        distances = [self.distance(x, w) for w in self.nodes]
        thresholds = [self.resonance_threshold(i) for i in range(len(self.nodes))]
        resonant_nodes = [i for i, (d, t) in enumerate(zip(distances, thresholds)) if d <= t]

        if not resonant_nodes:
            new_idx = self._add_node(x)
            return new_idx

        # Select winner
        winner_idx = resonant_nodes[np.argmin([distances[i] for i in resonant_nodes])]
        neighbors = list(self.edges.get(winner_idx, {}).keys())

        # Stochastic approximation with manifold regularization
        self.stochastic_approximation_update(winner_idx, x, neighbors)

        # Update context
        self.contexts[winner_idx] = (
            0.9 * self.contexts[winner_idx] + 0.1 * self.context_projection(x)
        )

        # Edge dynamics
        for j in neighbors:
            if j < len(self.nodes):
                self.edge_birth_death(winner_idx, j, co_activated=True)

        # Periodic MDL pruning
        if self.t % 200 == 0 and len(self.nodes) > 3:
            self.mdl_prune()

        return winner_idx

    def get_state(self):
        """Return current system state as a dictionary."""
        return {
            'nodes': [node.tolist() for node in self.nodes],
            'node_count': len(self.nodes),
            'edges': {str(k): v for k, v in self.edges.items()},
            'usage': list(self.usage),
            'step': self.t,
            'version': self.VERSION,
            'author': self.AUTHOR
        }

    def reset(self):
        """Reset the model to initial state."""
        self.nodes = []
        self.contexts = []
        self.edges = {}
        self.usage = []
        self.node_ages = []
        self.local_variances = []
        self.t = 0
