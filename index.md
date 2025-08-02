---
layout: default
---

<script src="https://polyfill.io/v3/polyfill.min.js?features=es6"></script>
<script id="MathJax-script" async
  src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>

Verlog is a well-tuned multi-turn RL framework built for long-horizon LLM agentic tasks. It extends [VeRL](https://github.com/volcengine/verl) and [BALROG](https://github.com/balrog-ai/BALROG), and follows the core design principles of [pytorch-a2c-ppo-acktr-gail](https://github.com/ikostrikov/pytorch-a2c-ppo-acktr-gail), while introducing tailored modifications for efficient multi-turn learning.

## Key features:  

⏳ Fixed-Turn Batching: For each training batch, we collect a fixed number of turns. If an episode has not yet terminated, we use the value function instead of final rewards as the supervised signal. This approach enables training in environments with highly variable and extended episode lengths.

🧠 Turn-Level Abstraction: Each turn is treated as an independent data point—no need to pack the entire history into the context window. Customize your memory mechanism as needed. 

🚀 Optimized for Long-Horizon Agentic Tasks: Verlog incorporates techniques like Dual Discounting GAE and Critic Pre-training, along with carefully tuned hyperparameters, to ensure strong performance on challenging long-horizon multi-turn benchmarks such as BALROG.

📊 Robust to Long and Variable Trajectories: Verlog is verified on environments with highly variable and extended episode lengths, such as BabyAI, BabaIsAI, and Crafter. Crafter’s trajectory mean length genreally ranges from 70 to 400 steps, while BabyAI and BabaIsAI both feature maximum episode lengths exceeding 100. These characteristics pose significant challenges for existing codebases, but Verlog ensures efficient training and stable performance even under such long-horizon complexity.

## Main Results

> Crafter's experiments are done with Qwen2.5-7B-Instruct model, using PPO algorithm, trained on 8xH100 GPUs with 82Gb memory for ~36 hours, corresponding to 170 PPO updates.

* Crafter Results:  

    | Model             | Instruct-model | Verlog (Ours) |
    |-------------------|----------------|---------------|
    | Rewards           | 5.80           | 10.44         |
    | Trajectory length | 172.23         | 196.42        |

BabyAI and BabaIsAI's experiments are done with Qwen2.5-3B-Instruct model, using PPO algorithm, trained on 4xA40 GPUs with 48Gb memory for ~24 hours, corresponding to 300 PPO updates.

* BabyAI Results (win rate)

    | Model          | goto | pickup | pick_up_seq_go_to | open |
    |----------------|------|--------|-------------------|------|
    | Instruct-model | 0.88 | 0.41   | 0.22              | 0.09 |
    | Verlog (Ours)  | 1.0  | 1.0    | 0.9               | 1.0  |


* BabaIsAI Results (win rate)

    goto_win → 🏁; 
    distr_obj → 🎁; 
    two_room → 🚪; 
    distr_obj_rule → 📏;  
    maybe_break_stop → ⚠️;  
  
  | Model          | 🏁+🎁         | 🚪+🏁          | 🚪+🏁+🎁+📏                | 🚪+⚠️+🏁                        |
  |----------------|----------------|----------------|------------------------------|----------------------------------|
  | Instruct-model | 0.21           | 0.09           | 0.09                         | 0.21                             |
  | Verlog (Ours)  | 1.0            | 1.0            | 1.0                          | 0.69                             |

## Technical Report

In the following sections, we explain our design choices, present implementation details, and conclude with potential research questions that our framework could help address.

### Model & Prompt

* **Instruct Model:**
    For simplicity, we start with the Instruct model, **Qwen-2.5-3B/7B-Instruct**. We chose this over the base model because it allows us to leverage [BALROG](https://github.com/balrog-ai/BALROG) for debugging and to use the benchmark's prompts with minimal modifications.

* **Memory Mechanism:**
    Rather than placing the entire trajectory into the context window, we include only the latest $$n+1$$ turns. Each turn, i.e., data = $$(\text{history}_t, s_t, \text{think}_t, a_t)$$ , with $$\text{history}_t = \{s_{t-n}, \text{think}_{t-n}, a_{t-n}, ..., s_{t-1}, \text{think}_{t-1}, a_{t-1}\}$$, is treated as an individual training data point. As a result, each training batch consists of `batch_size` individual turns, not `batch_size` full trajectories. We detail our memory design choices below:

    * **(1) Choice of $$n$$:**
    For the 3B Qwen model, performance peaks at $$n = 1$$ or $$2$$ and degrades as $$n$$ increases to $$4$$ or $$8$$. We hypothesize that this decline is due to the 3B model’s limited capacity to handle long contexts—for example, $$n = 8$$ yields a prompt of approximately 4.6k tokens. Whether this trend holds for larger models is an open question. Notably, the tasks we evaluate can be framed as Markov Decision Processes (MDPs). In more complex or partially observable tasks, a larger $$n$$ may help.

    * **(2) Including reasoning paths in context:**
    We tested a variant that includes only the final action in history: data = $$(\text{history}_t, s_t, \text{think}_t, a_t)$$, with $$\text{history}_t = \{s_{t-n}, a_{t-n}, ..., s_{t-1}, a_{t-1}\}$$. This version failed to learn an effective policy.

    * **(3) Periodically reset history:**
    We tested a variant that periodically clears the history buffer (every 5 steps). This slightly improved zero-shot performance, as the agent often got stuck in a loop by copying incorrect decision patterns from previous turns. The response length also increased slightly due to greater response diversity. However, the final performance declined, so we did not adopt this reset strategy in the final algorithm.

    Qualitative results show that increasing $$n$$ leads to: (1) longer responses, as LLMs often spend additional tokens rephrasing or reiterating previous plans; (2) less diverse reasoning paths, with the model tending to mimic reasoning patterns from earlier turns; and (3) more hallucinations, where the LLM struggles to distinguish between internally generated thoughts and actual events in the environment. Future work is needed to investigate how memory design influences these behaviors in LLM agents.

* **Prompt Template:**
    Belows is the prompt template used for BabyAI. The prompts are adapted from [BALROG](https://github.com/balrog-ai/BALROG).
    ```
    [SYSTEM] You are an agent playing a simple navigation game. 
    Your goal is to {MISSION}. 
    The following are the possible actions you can take in the game, followed by a short description of each action: {AVAILABLE ACTIONS}. 
    In a moment I will present you an observation. Tips: {TIPS}.
    PLAY!
    ```
    ```
    [USER] {OBSERVATION}
    ```
    ```
    [ASSISTANT]
    THINK: {THINK}
    ACTION: {ACTION}
    ```
    ```
    [USER] {OBSERVATION}
    What will you do next? Please respond in the following format:
    THINK: step-by-step reasoning
    ACTION: One valid action from the allowed set.
    ```
    We recommend always examining the model’s zero-shot outputs before training. Specifically, evaluate: (1) Whether reasoning paths are diverse, (2) whether the model reasons sufficiently before selecting an action, (3) the ratio of valid actions, and (4) the types of failure cases. These checks ensure the model understands the environment from the prompt. If not, revise the prompt before fine-tuning.

### Environment

* **Valid Action:**
   Improving the valid action ratio through prompt engineering is the simplest and most effective way to boost performance. In our setup, we ensure the model produces valid actions over 95% of the time using the following strategies:

  * Hardcoded action translation: Certain invalid actions are frequently produced by zero-shot LLMs (e.g., "Move forward" and "Go forward"). We implement a hand-crafted translation function to map these to valid actions, preventing them from lowering the valid action ratio.
  
  * Replace invalid actions with a default action: When the LLM outputs an invalid action, the environment rejects it and executes a predefined default action instead. Simultaneously, we replace the invalid action with the default one before appending it to the history buffer. This prevents the agent from mimicking the invalid action in subsequent steps.
    
We observe that truncating the trajectory upon encountering an invalid action leads to worse performance. Replacing invalid actions with a default action yields better results. In this work, we apply a 0.1 penalty to invalid actions. However, with a high valid action ratio, the format penalty has minimal impact on overall performance.

* **Reward:**
    Rewards are rule-based and provided by the environment. In BabyAI and BabaIsAI, the reward (ranging from 0 to 1) is only available at the end of the trajectory. Longer trajectories generally get lower rewards.

    Here, We adopt a **binary trajectory-level reward** scheme: 1 for success trajectory, 0 for failure trajectory. Combined with dual-discount GAE, this ensures earlier steps in suboptimal trajectories get lower credit than those in optimal ones.

* **Batch Environment:**
    Our framework supports asynchronous rollouts and works with any environment using the OpenAI Gym interface. Each training batch size is: `n_env` × `e_len`, where:
    * `n_env` = number of parallel environments
    * `e_len` = episode length per rollout

    Note: `e_len` can be smaller than the environment's trajectory length. For example, we set `e_len = 8` and max trajectory length = 128 in BabyAI. For early truncated trajectories, we leverage the value function to guide the training process. We find that using `e_len = 8`, `n_env = 32` performs better than `e_len = 16`, `n_env = 16`.
    

### Algorithm

* **Dual Discounting GAE:**
    We decouple token-level discounting $$(\gamma_{\text{token}}, \lambda_{\text{token}})$$ and step-level $$(\gamma_{\text{step}}, \lambda_{\text{step}})$$. We set:

    * $$\gamma_{\text{step}} = 0.99$$, $$\lambda_{\text{step}} = 0.95$$
    * $$\gamma_{\text{token}} = 1.0$$, $$\lambda_{\text{token}} = 1.0$$

    The GAE is computed recursively:

    $$
    \hat{A}_t = \gamma\lambda \hat{A}_{t+1} + \delta_t^V
    $$

    where:

    * $$\gamma\lambda = \gamma_{\text{step}} \lambda_{\text{step}}$$, if tokens are from different turns
    * $$\gamma\lambda = \gamma_{\text{token}} \lambda_{\text{token}}$$, otherwise
    * and $$\delta_t^V = -V(s_t) + r_t + \gamma V(s_{t+1})$$

    The recursion starts from the last token of the final turn and proceeds backward. Once all tokens in the final turn are processed, we move to the last token of the second-to-last turn, and continue this process recursively. 
    If a trajectory is truncated at step $$T$$, we store the next state $$s_{T+1}$$ but do not sample $$a_{T+1}$$. Instead, we use the final token of $$s_{T+1}$$ to estimate $$V(s_{T+1})$$, used as the bootstrap value in GAE.

* **Value Function Estimation:**
    * When both $$\gamma$$ and $$\lambda$$ are set to 1.0, the value function serves purely as a baseline in PPO’s advantage estimation. Specifically, the advantage for the $$t$$-th token in the last turn is defined as $$A_{-1,t} = r - V_{-1,t}$$, where $$r$$ is the trajectory reward and $$V_{-1,t}$$ is the value estimate for the $$t$$-th token in the last turn.

    * When $$\lambda$$ are less than 1.0, the value function contributes to the GAE objective beyond serving as a simple baseline. For instance, in our setting with $$\gamma_{\text{step}} = 0.99$$, $$\lambda_{\text{step}} = 0.95$$, and $$\gamma_{\text{token}} = 1.0$$, $$\lambda_{\text{token}} = 1.0$$, where the reward $$r$$ is non-zero only at the end of the trajectory, the advantage for the $$t$$-th token in the second-to-last turn is given by: $$A_{-2,t} = \gamma_{\text{step}}[\lambda_{\text{step}} r + (1-\lambda_{\text{step}}) V_{-1,0}] - V_{-2,t}$$. In fact, in our setting, the value function of the first token in each turn is used to bootstrap the GAE objective for the preceding turn.

    * Since the first token of each turn carries more semantic significance than the subsequent tokens, we assign it a higher weight when training the critic network.

* **Critic Warmup:**
    In our setting, we warm up the critic before fine-tuning, as it is used both for bootstrapping truncated trajectories and for computing GAE. That is, we freeze the actor and update only the critic at the beginning of training. Specifically, We collect `w_epoch × batch_size` turns of data at the beginning. For each warmup iteration, we compute the GAE objective with current critic, sample one tenth of the collected data, train the critic, and repeat this process for `w_iter` iterations. We select `w_epoch = 40` and `w_iter = 5` in our experiments, and make sure that the critic loss converges to a small value before fine-tuning the actor.
    
* **KL-Divergence in Reward:**
    Adding a KL-divergence term $$KL(\pi||\pi_0)$$ in reward stabilizes training. Without it, the policy quickly drifts from $$\pi_0$$ and converges to poor solutions. KL penalty encourage local exploration around $$\pi_0$$ before divergence.

    For entorpy term in the PPO loss, we experimented with higher entropy coefficients (e.g., 3e-3, 1e-2), which caused instability.

    We set both entropy and KL reward coefficients to **1e-3**. 

## Conclusion

In summary, when training an LLM agent using multi-turn RL, the following aspects are critical (in order of importance):

1. Prompt design
2. Value loss optimization
3. Exploration



