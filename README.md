# CAET
README

Title: Cost-Aware Optimal Pairwise Pure Exploration.

Abstract: Pure exploration is one of the fundamental problems in multi-armed bandits (MAB). However, existing works mostly focus on specific pure exploration tasks, without a holistic view of the general pure exploration problem. This work fills this gap by introducing a versatile framework to study pure exploration, with a focus on identifying the pairwise relationships between targeted arm pairs. Moreover, unlike existing works that only optimize the stopping time (i.e., sample complexity), this work considers that arms are associated with potentially different costs and targets at optimizing the cumulative cost that occurred during learning. Under the general framework of pairwise pure exploration with arm-specific costs, a performance lower bound is derived. Then, a novel algorithm, termed CAET (Cost-Aware Pairwise Exploration Task), is proposed. CAET builds on the track-and-stop principle with a novel design to handle the arm-specific costs, which can potentially be zero and thus represent a very challenging case. Theoretical analyses prove that the performance of CAET approaches the lower bound asymptotically. Special cases are further discussed, including an extension to regret minimization, which is another major focus of MAB. The effectiveness and efficiency of CAET are also verified through experimental results under various settings.

Code: 
- experiment.py: This project is made for the special case of the CAET algorithm: regret minimization in the three-armed bandit setting during the ranking identification task.
- Result of First Graph in Figure 1: $\log(\frac{1}{\delta}) = 30 \sim 100; r=0.4; (\mu_1, \mu_2, \mu_3) = (0.5, 0.9, 0.2); \theta = 1.2$.
- Result of Second Graph in Figure 1: $\log(\frac{1}{\delta}) = 30 \sim 100; r=0.4; (\mu_1, \mu_2, \mu_3) = (1.4, 0.8, 0.3); \theta = 1.2$.
- In this experiment, we take the threshold in the truncation function as $0.1\log^{-0.1}(\frac{1}{\delta})$.
