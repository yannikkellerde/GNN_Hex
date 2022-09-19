1. Grow networks for curriculum -> Later
2. Fix next states when using epsilon greedy. This won't work. Rainbow DQN intentionally uses incorrect target, so that n-step returns and noisy dqn can be used without huge computational overhead.
3. Share parameters between maker and breaker. DONE.
4. Fix elo evaluation. E.g. keep best models. DONE.
5. Noisy DQN? -> Implemented. However, I am not sure i fully understand it.
6. N-Step updates could really help. Implemented, theoretically wrong q-targets, but that's how rainbow intends it.
