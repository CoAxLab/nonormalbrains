import numpy as np
from collections import defaultdict


def msprt(X, Y, g, C):
    """MSPRT."""

    H = np.log(np.sum(np.exp(Y)))
    Y += (g * X) - H
    test = Y > C

    return Y, test


def n_trials(n,
             means,
             sigma,
             g,
             C,
             max_iteration=5000,
             seed_value=None,
             save_traces=False,
             mode='normal'):
    """Run N msrpt experiments"""

    np.random.seed(seed_value)
    noise = getattr(np.random, mode)

    # Init
    decisions = []
    reaction_times = []

    if save_traces:
        traces = defaultdict(list)

    # Run N trials!
    for i in range(n):
        # Re-init
        Y = 0
        i = 0
        test = np.zeros(2, dtype=np.bool)

        # A trial:
        while not np.any(test) and (i < max_iteration):
            i += 1

            # Build X
            X = []
            for m in means:
                X.append((noise() + m) / sigma)
            X = np.asarray(X)

            if save_traces:
                traces[i].append(X)

            # Test
            Y, test = msprt(X, Y, g, C)

        # Save past N
        decisions.append(np.where(test)[0])
        reaction_times.append(i)

    for i, d in enumerate(decisions):
        if d.size == 0:
            decisions[i] = None

    # Build return
    out = [decisions, reaction_times]
    if save_traces:
        # Process traces
        for k, v in traces.items():
            traces[k] = np.vstack(v)

        # Add them to the return
        out.append(traces)

    return out
