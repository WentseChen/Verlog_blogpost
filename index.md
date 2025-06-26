---
layout: default

## Model

**Instruct Model:**
For simplicity, we start with the Instruct model, **Qwen-2.5-3B-Instruct**. We chose this over the base model because it allows us to leverage BALROG for debugging and to use the benchmark's prompts with minimal modifications.

**Memory Mechanism:**
Rather than placing the entire trajectory into the context window, we include only the latest \$n+1\$ turns. Each turn, i.e.,
**data** = (`history_t`, \$s\_t\$, `think_t`, \$a\_t\$),
with
**history\_t** = {\$s\_{t-n}\$, `think_{t-n}`, \$a\_{t-n}\$, ..., \$s\_{t-1}\$, `think_{t-1}`, \$a\_{t-1}\$},
is treated as an individual training data point. As a result, each training batch consists of `batch_size` individual turns, not full trajectories.

We detail our memory design choices below:

* **(1) Choice of \$n\$:**
  For the 3B Qwen model, performance peaks at \$n = 1\$ or \$2\$ and degrades as \$n\$ increases to \$4\$ or \$8\$. We hypothesize that this results from difficulty handling long contexts—for example, \$n = 8\$ yields a prompt of approximately 4.6k tokens. Whether this trend holds for larger models is an open question. Notably, the tasks we evaluate can be framed as Markov Decision Processes (MDPs). In more complex or partially observable settings, a larger \$n\$ may help.

* **(2) Including reasoning paths in context:**
  We tested a variant that includes only the final action in history:
  **data** = (`history_t`, \$s\_t\$, `think_t`, \$a\_t\$), with
  **history\_t** = {\$s\_{t-n}\$, \$a\_{t-n}\$, ..., \$s\_{t-1}\$, \$a\_{t-1}\$}.
  This version failed to learn an effective policy.

**Prompt Template:**
Figure ⬜ illustrates the prompt template used for BabyAI (more examples in Section ⬜). The prompts are adapted from BALROG.

We recommend always examining the model’s zero-shot outputs before training. Specifically, evaluate:

1. Whether reasoning paths are diverse.
2. Whether the model reasons sufficiently before selecting an action.
3. The ratio of valid actions.
4. The types of failure cases.

These checks ensure the model understands the environment from the prompt. If not, revise the prompt before fine-tuning.

---

## Environment

**Valid Action:**
Improving the valid action ratio via prompt engineering is the simplest way to boost zero-shot performance. In our setup, we ensure the model produces valid actions over 95% of the time.

We find that truncating the entire trajectory upon encountering an invalid action reduces performance. Replacing the invalid action with a default one works better. With a high valid-action ratio, applying a format penalty has little impact.

**Reward:**
Rewards are rule-based and provided by the environment. In BabyAI and BabaIsAI, the reward (ranging from 0 to 1) is only available at the end of the trajectory. Longer trajectories generally get lower rewards.

We adopt a **binary reward** scheme:

* 1 for success
* 0 for failure

Combined with dual-discount GAE, this ensures earlier steps in suboptimal trajectories get lower credit than those in optimal ones.

**Batch Environment:**
Our framework supports asynchronous rollouts and works with any environment using the OpenAI Gym interface. Each training batch size is:
`n_env` × `e_len`,
where:

* `n_env` = number of parallel environments
* `e_len` = episode length per rollout

Note: `e_len` can be smaller than the environment's trajectory length. For example, we set `e_len = 8` and max trajectory length = 128 in BabyAI. This lets us use the value function to guide incomplete trajectories.

We find that using `e_len = 8`, `n_env = 32` performs better than `e_len = 16`, `n_env = 16`.

**Truncated Trajectory:**
If a trajectory is truncated at step \$t\$, we store the next state \$s\_{t+1}\$ but do not sample \$a\_{t+1}\$. Instead, we use the final token of \$s\_{t+1}\$ to estimate \$V(s\_{t+1})\$, used as the bootstrap value in GAE.

---

## Algorithm

**Dual Discounting GAE:**
We separate token-level discounting \$(\gamma\_{\text{token}}, \lambda\_{\text{token}})\$ from step-level \$(\gamma\_{\text{step}}, \lambda\_{\text{step}})\$.
We set:

* \$\gamma\_{\text{step}} = 0.99\$, \$\lambda\_{\text{step}} = 0.95\$
* \$\gamma\_{\text{token}} = 1.0\$, \$\lambda\_{\text{token}} = 1.0\$

The GAE is computed recursively:

$$
\hat{A}_t = \gamma\lambda \hat{A}_{t+1} + \delta_t^V
$$

where:

* \$\gamma\lambda = \gamma\_{\text{step}} \lambda\_{\text{step}}\$ if tokens are from different turns
* \$\gamma\lambda = \gamma\_{\text{token}} \lambda\_{\text{token}}\$ otherwise
* \$\delta\_t^V = -V(s\_t) + r\_t + \gamma V(s\_{t+1})\$

The recursion starts from the last token of the final turn and proceeds backward, turn by turn.

**Value Function Estimation:**
When \$\gamma = \lambda = 1.0\$, the value function acts only as a baseline. The advantage at token \$t\$ in the final turn is:
\$r - V\_{-1,t}\$

With \$\gamma < 1.0\$, the value function also shapes the GAE directly. Since the first token in each turn is more semantically meaningful, we weight it more during critic training.

**Critic Warmup:**
Reliable value estimates are essential for:

1. Bootstrapping truncated trajectories.
2. Computing GAE.

We warm up the critic before fine-tuning:

* Freeze the actor.
* Update only the critic for the first `w_epoch` epochs.

We collect `w_epoch × batch_size` turns of data at the beginning. During `w_iter` epochs, this dataset is used to compute GAE and train the critic. Afterward, we resume standard RL training.

**KL-Divergence in Reward:**
Adding a KL-divergence term stabilizes training. Without it, the policy quickly drifts from \$\pi\_0\$ and converges to poor solutions. KL rewards encourage local exploration around \$\pi\_0\$ before divergence.

We experimented with higher entropy coefficients (e.g., 3e-3, 1e-2), which caused instability. We set both entropy and KL reward coefficients to **1e-3**. It's unclear if this holds for larger batch sizes—a question for future work.

---

## Conclusion

In summary, when training an LLM agent using multi-turn reinforcement learning, the following aspects are critical (in order of importance):

1. Prompt design
2. Value loss optimization
3. Exploration

