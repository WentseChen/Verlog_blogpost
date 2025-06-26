---
layout: default
---
# Technical Report

We will go through each of the design choices in detail in the technical report.

## Model

**Instruct Model:** For simplicity, we're starting with the Instruct model, Qwen-2.5-3B-Instruct. We chose this over the base model because it allows us to leverage BALROG for debugging and to use the benchmark's prompts with minimal modifications.

**Memory Mechanism:** Rather than placing the entire trajectory into the context window, we include only the latest $n+1$ turns. To support this, we treat each turn, i.e., $\text{data}=(\text{history\_t}, s_t, \text{think}_t, a_t)$ and $\text{history\_t} = \{s_{t-n}, \text{think}_{t-n}, a_{t-n},...,s_{t-1}, \text{think}_{t-1}, a_{t-1}\}$, as an individual training data point. Consequently, each training batch consists of $\textit{batch\_size}$ individual turns, rather than $\textit{batch\_size}$ full trajectories.

### Memory Design Choices

1. **Choice of $n$:** For the 3B Qwen model, performance is highest when $n = 1$ or $2$, and degrades as $n$ increases to $4$ or $8$. We hypothesize that this is due to the model's difficulty handling longer input contexts. In more complex settings, a larger $n$ may be beneficial.
   
2. **Including Reasoning Paths in Context:** We examined whether to include full reasoning paths in the context. A variant that includes only the final action, i.e., $\text{data}=(\text{history\_t}, s_t, \text{think}_t, a_t)$ and $\text{history\_t} = \{s_{t-n}, a_{t-n},...,s_{t-1}, a_{t-1}\}$, was unable to learn an effective policy.

**Prompt Template:** Figure~\ref{} illustrates the prompt template used for BabyAI, with additional examples provided in Section~\ref{}. The prompts are adapted from BALROG. We recommend always examining the model’s zero-shot outputs before initiating training.

## Environment

**Valid Action:** The easiest way to improve zero-shot performance is through prompt engineering that increases the ratio of valid actions. In our experiments, we ensure that the zero-shot model produces valid actions more than 95% of the time.

**Reward:** In this work, the reward is provided by the environment and follows a rule-based structure. We adopt a binary reward scheme: a reward of 1 for successful trajectories and 0 for failures. This binary reward, combined with dual-discount GAE, ensures that earlier steps in sub-optimal trajectories receive lower credit compared to those in optimal ones.

**Batch Environment:** The framework supports asynchronous environment rollouts and is compatible with any custom environment that follows the OpenAI Gym interface. Each training batch has a size of $\textit{n\_env} \times \textit{e\_len}$.

**Truncated Trajectory:** If the trajectory is truncated at time step $t$, we also store the subsequent state $s_{t+1}$ in the buffer. However, we do not query the policy to obtain the corresponding action $a_{t+1}$. Instead, we use the last token of $s_{t+1}$ to estimate the value of the next state $V(s_{t+1}$), serving as the bootstrap for the GAE computation.

## Algorithm

**Dual Discounting GAE:** We decouple the token-level discount factors $(\gamma_{\text{token}}, \lambda_{\text{token}})$ from the step-level discount factors $(\gamma_{\text{step}}, \lambda_{\text{step}})$ when computing GAE.

**Value Function Estimation:** When both $\gamma$ and $\lambda$ are set to 1.0, the value function serves purely as a baseline for PPO's advantage estimation. However, when $\gamma$ and $\lambda$ are set to values less than 1.0, the value function also influences the GAE objective directly.

**Critic Warmup:** To ensure the value function is reliable for RL, we perform a critic warmup phase before fine-tuning begins. During the warmup phase, we collect pre-collected data to compute the GAE objective and update the critic network.

**Use KL-Divergence in Reward:** Incorporating a KL-divergence term in the reward function improves performance by encouraging exploration and stabilizing training.

## Conclusion

In summary, when training an LLM agent with multi-turn reinforcement learning, it is crucial to focus on prompt design, value loss optimization, and exploration. These aspects are essential for achieving robust performance across various environments and tasks.

