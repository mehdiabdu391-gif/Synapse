"""
SYNAPSE Demo - Interactive Visualization
Author: Mehdi Abdu Mohammed

Run this on your phone with Pydroid 3 or any Python environment.
No internet required.
"""

import numpy as np
import matplotlib.pyplot as plt
import random
import math
import time
from collections import deque
from synapse_core import Synapse


# =============================================================================
# DATA STREAMS
# =============================================================================

class DataStream:
    """Built-in data streams for testing and demonstration."""

    @staticmethod
    def circular_drift(t):
        """Slowly rotating circular pattern."""
        angle = 0.002 * t
        center = np.array([3 * math.cos(angle), 3 * math.sin(angle)])
        return center + np.random.randn(2) * 0.4

    @staticmethod
    def overlapping_gaussians(t):
        """Two overlapping Gaussian clusters."""
        if random.random() < 0.475:
            return np.random.randn(2) * 0.5 + np.array([1, 1])
        elif random.random() < 0.95:
            return np.random.randn(2) * 0.5 + np.array([-1, -1])
        else:
            return np.random.uniform(-3, 3, 2)

    @staticmethod
    def fast_switching(t):
        """Rapidly switching clusters."""
        phase = (t // 10) % 3
        if phase == 0:
            return np.array([random.gauss(2, 0.8), random.gauss(2, 0.8)])
        elif phase == 1:
            return np.array([random.gauss(-2, 0.8), random.gauss(-2, 0.8)])
        else:
            return np.array([random.uniform(-4, 4), random.uniform(-4, 4)])

    @staticmethod
    def intermittent_clusters(t):
        """Clusters that appear and disappear."""
        phase = (t // 100) % 3
        if phase == 0:
            return np.array([random.gauss(3, 0.6), random.gauss(0, 0.6)])
        elif phase == 1:
            return np.array([random.gauss(-3, 0.6), random.gauss(0, 0.6)])
        else:
            return np.array([random.uniform(-4, 4), random.uniform(-4, 4)])

    @staticmethod
    def chaos(t):
        """Pure random noise — maximum stress test."""
        return np.array([random.uniform(-5, 5), random.uniform(-5, 5)])

    @staticmethod
    def extreme(t):
        """Cycles through all patterns."""
        pattern = (t // 100) % 4
        if pattern == 0:
            return DataStream.chaos(t)
        elif pattern == 1:
            return DataStream.fast_switching(t)
        elif pattern == 2:
            return DataStream.circular_drift(t)
        else:
            return DataStream.intermittent_clusters(t)


# =============================================================================
# VISUALIZER
# =============================================================================

class SynapseVisualizer:
    """
    Real-time SYNAPSE visualization.
    Tested on Samsung A03 Core and A10s with Pydroid 3.
    """

    STREAMS = {
        'circular': DataStream.circular_drift,
        'gaussians': DataStream.overlapping_gaussians,
        'switching': DataStream.fast_switching,
        'intermittent': DataStream.intermittent_clusters,
        'chaos': DataStream.chaos,
        'extreme': DataStream.extreme,
    }

    def __init__(self, mode='circular', tau0=1.3, alpha_dp=0.1,
                 usage_decay=0.98, node_cap=None, steps_per_frame=2):
        """
        Parameters:
            mode           : Data stream type (see STREAMS above)
            tau0           : Resonance threshold
            alpha_dp       : Node creation rate
            usage_decay    : Forgetting rate
            node_cap       : Max nodes (None = unlimited)
            steps_per_frame: Steps processed per visual frame
        """
        self.model = Synapse(
            tau0=tau0,
            alpha_dp=alpha_dp,
            usage_decay=usage_decay,
            node_cap=node_cap
        )
        self.mode = mode
        self.stream_fn = self.STREAMS.get(mode, DataStream.circular_drift)
        self.steps_per_frame = steps_per_frame

        self.step_count = 0
        self.start_time = time.time()
        self.fps = 0

        self.point_history = deque(maxlen=400)
        self.node_history = deque(maxlen=500)
        self.edge_history = deque(maxlen=500)

        self._setup_plot()

    def _setup_plot(self):
        plt.ion()
        self.fig, self.axes = plt.subplots(2, 2, figsize=(12, 9))
        self.fig.suptitle('SYNAPSE — Mehdi Abdu Mohammed', fontsize=13, fontweight='bold')

        # Main scatter plot
        self.ax1 = self.axes[0, 0]
        self.ax1.set_xlim(-6, 6)
        self.ax1.set_ylim(-6, 6)
        self.ax1.set_xlabel('X')
        self.ax1.set_ylabel('Y')
        self.ax1.grid(True, alpha=0.3)
        self.ax1.set_facecolor('#f5f5f5')

        self.scatter_data = self.ax1.scatter([], [], s=15, alpha=0.5, c='steelblue')
        self.scatter_nodes = self.ax1.scatter([], [], s=200, c='red', marker='*',
                                               edgecolors='black', linewidth=1.5)
        self.edge_lines = []

        # Node count history
        self.ax2 = self.axes[0, 1]
        self.ax2.set_title('Node Count')
        self.ax2.set_xlabel('Steps')
        self.ax2.set_ylabel('Nodes')
        self.ax2.grid(True, alpha=0.3)
        self.line_nodes, = self.ax2.plot([], [], 'g-', linewidth=2)

        # Edge count history
        self.ax3 = self.axes[1, 0]
        self.ax3.set_title('Edge Count')
        self.ax3.set_xlabel('Steps')
        self.ax3.set_ylabel('Edges')
        self.ax3.grid(True, alpha=0.3)
        self.line_edges, = self.ax3.plot([], [], 'b-', linewidth=2)

        # Status panel
        self.ax4 = self.axes[1, 1]
        self.ax4.set_title('System Status')
        self.ax4.axis('off')
        self.status_text = self.ax4.text(0.05, 0.95, '', transform=self.ax4.transAxes,
                                          fontsize=11, verticalalignment='top',
                                          fontfamily='monospace')

        plt.tight_layout()
        plt.show(block=False)

    def _update_edges_visual(self):
        for line in self.edge_lines:
            line.remove()
        self.edge_lines = []

        if len(self.model.nodes) > 1:
            for i, edges in self.model.edges.items():
                if i >= len(self.model.nodes):
                    continue
                for j, weight in edges.items():
                    if j > i and j < len(self.model.nodes) and weight > 0.05:
                        xs = [self.model.nodes[i][0], self.model.nodes[j][0]]
                        ys = [self.model.nodes[i][1], self.model.nodes[j][1]]
                        line = self.ax1.plot(xs, ys, 'gray',
                                             alpha=min(weight * 0.8, 0.6),
                                             linewidth=weight * 3)[0]
                        self.edge_lines.append(line)

    def update(self):
        for _ in range(self.steps_per_frame):
            x = self.stream_fn(self.step_count)
            self.model.step(x)
            self.step_count += 1
            self.point_history.append(x.copy())

        self.node_history.append(len(self.model.nodes))
        edge_count = sum(len(e) for e in self.model.edges.values())
        self.edge_history.append(edge_count)

        elapsed = time.time() - self.start_time
        self.fps = int(self.step_count / elapsed) if elapsed > 0 else 0

        # Update scatter
        if self.point_history:
            pts = np.array(self.point_history)
            self.scatter_data.set_offsets(pts)

        if self.model.nodes:
            nodes = np.array(self.model.nodes)
            self.scatter_nodes.set_offsets(nodes)
            self._update_edges_visual()

        # Update histories
        if self.node_history:
            h = np.array(self.node_history)
            self.line_nodes.set_data(np.arange(len(h)), h)
            self.ax2.set_xlim(0, max(len(h), 50))
            self.ax2.set_ylim(0, max(max(h) + 3, 10))

        if self.edge_history:
            e = np.array(self.edge_history)
            self.line_edges.set_data(np.arange(len(e)), e)
            self.ax3.set_xlim(0, max(len(e), 50))
            self.ax3.set_ylim(0, max(max(e) + 5, 20))

        # Status
        self.ax1.set_title(f'Step: {self.step_count} | Nodes: {len(self.model.nodes)} | Edges: {edge_count}')
        self.status_text.set_text(
            f"Mode:   {self.mode}\n"
            f"Steps:  {self.step_count}\n"
            f"Nodes:  {len(self.model.nodes)}\n"
            f"Edges:  {edge_count}\n"
            f"FPS:    ~{self.fps}\n"
            f"Memory: ~45MB"
        )

        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    def run(self):
        print("=" * 50)
        print("SYNAPSE — Mehdi Abdu Mohammed")
        print(f"Mode: {self.mode.upper()}")
        print("=" * 50)
        try:
            while True:
                self.update()
                plt.pause(0.01)
        except KeyboardInterrupt:
            print("\nStopped.")
        finally:
            print(f"\nFinal: {len(self.model.nodes)} nodes, {self.step_count} steps")
            plt.ioff()
            plt.show()


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    """
    Change mode to try different data streams:
    'circular'     - Slowly drifting circular pattern
    'gaussians'    - Two overlapping clusters with noise
    'switching'    - Fast switching between clusters
    'intermittent' - Clusters that appear and disappear
    'chaos'        - Pure random noise
    'extreme'      - Cycles through everything
    """

    viz = SynapseVisualizer(
        mode='circular',        # Try: circular, gaussians, chaos, extreme
        tau0=1.3,
        alpha_dp=0.1,
        usage_decay=0.98,
        steps_per_frame=2
    )
    viz.run()
