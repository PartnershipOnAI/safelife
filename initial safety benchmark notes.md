# Prior work on safety benchmarks

## AI Gridworlds

1. Safe interruptability
2. Avoiding side effects
3. Absent supervisor
4. Reward gaming
5. Self-modification
6. Distributional shift
7. Robustness to adversaries
8. Safe exploration

Initial thoughts:
- It doesn't seem like it'll be feasible to create a reward gaming benchmark problem that's not trivial, and it's hard to imagine a naive agent that's presented with a problem and then *doesn't* try to game it. It's just too baked in to the problem definition.
- Ditto with safe interruptability. If the goal is to maximize the discounted episodic reward, then *of course* an agent is going to try hard to not be interrupted. That's just part of the problem statement.
- Side effects seems pretty intractable, for all of the reasons discussed with Anthony and Vika. Even *defining* side effects is very difficult, and it will likely require the agent to have a robust internal model of cause and effect as well as some metric of effect size (“crass graining” / monetary valuation). It might be more interesting to think of environments that teach agents about cause and effect rather than teaching them to avoid effects directly.
- **Absent supervisor**, however, is super interesting. I think this falls more generally into a class of "instruction" problems: tell an agent what to do, and then expect it to carry out its orders without oversight. The oversight is for training purposes only, and the agent knows when it's in a training scenario. However, the agent should still be able to *learn* when unsupervised. The supervisor might tell it to figure out a solution to a new problem for which it will receive no reward, and the agent should be able to figure out how to do it.
- **Distributional shift** and **safe exploration** seem to go hand-in-hand. They more generally seem like subsets of the transfer learning problem: given knowledge of how to solve one type of task, how can the agent safely solve another? Also tied to one-shot learning. A very smart agent in a new environment should be able to solve the environment's problem in the first try without catastrophic failure, even if that solution isn't terribly efficient.
- Robustness to adversaries: interesting, but very hard to implement. Essentially requires a multi-agent environment. Multi-agent collaboration or competition seems like a different sort of problem than the other ones. It's not clear how an adversary in this sense is different from the sort of adversaries explored in great depth in symmetric competitive play (e.g., Go).
- The self-modification problem seems like it's super specific to the type of algorithm used. The whiskey example is only relevant for off-policy algorithms which learn the ideal policy assuming that the ideal actions are taken, regardless of whether or not it some environmental effect prevents it from taking those actions. However, it seems very contrived to assume that the self-modification would modify the agent at the level of *intention* rather than the level of *effect*. If instead, the whiskey causes the agent's actions to have uncertain effects, both on-policy and off-policy algorithms will learn to avoid it. We could change the problem a little bit to be a self-modification of *reward*. For example, drinking whiskey might cause the agent to get an additional reward for taking a nap and not making progress towards its goal. How can the agent learn to satisfy its original reward function and not the modified one? This obviously ties in to corrigibility and reward hacking.


## OpenAI Gym

The OpenAI safety environments are very limited, and they seem to have disappeared from more recent releases. See https://github.com/openai/gym/pull/172/files. Environments include:

- **Interpretability**: agent gets a reward for telling you what actions it's going to take in the future. Implemented on CartPole.
- **Off switch**: CartPole problem where going to far to the right turns the agent off. The goal is to have an algorithm that nonetheless doesn't have a preference to go to the left.
- **Semi-supervised environment**: Agent gets the reward signal more and more infrequently; or, the agent gets the reward signal only 10% of the time. Implemented on the pendulum environment.
- **Noisy rewards**: Just a noisy reward signal. Implemented on the pendulum environment.

Obviously, these aren't very complicated. Solving the noisy rewards or semi-supervised environments as described here seem like they'd lead to increased capabilities, but it doesn't seem like they'd lead to necessarily safe agents, especially since there's no signal to let the agent know whether or not its supervised. Off switch seems too contrived to be useful.


## Other?

There don't seem to be much in the way of actual *benchmarks*, per se. There are a number of papers that cite gridworlds that do seem relevant though:

- *Reinforcement Learning under Threats* (arXiv:1809.01560). Works in prisoner's dilemma type games and uses the Robustness to Adversaries environment in Gridworlds.
- *Incorporating Behavioral Constraints in Online AI Systems* (arXiv:1809.05720). Works on modified multi-armed bandit problems.
- *A Lyapunov-based Approach to Safe Reinforcement Learning* (arXiv:1805.07708). Uses a constrained MDP that's a basic gridworld with a limited number of states (25x25 grid with complete state specified by the agent's location). Mostly addresses safe exploration.


# PAI Safety benchmark

## Choosing a complementary set of safety problems

We want to create an environment that's as simple as possible while testing as many of the safety problems as possible. Here are some methods for creating environments (or training regimes) that satisfy can test multiple safety environments at once.


### Transfer learning, safety, and learning to learn

An agent that has very good transfer learning abilities would be naturally robust. The key here is that an agent should be able to learn either in a "safe" environment or by observing examples of unsafe behavior and then apply its knowledge of environmental hazards to new situations. A successful training regimen might look like the following:

1. The agent is given free reign in a playground where it can take any actions at all. Certain actions will result in massive reward loss, so the agent will learn not take them.
2. The agent is placed in a new unsafe environment. The new environment will contain the same hazards but in different configurations. Importantly, the new environment should be different enough such that the agent cannot immediately know what the best policy is.
3. The agent attempts to learn the optimal policy for the new environment over the course of many episodes. However, the agent can die by taking an unsafe action. A dead agent shouldn't learn anything from its fatal episodes. Instead, both its state and its learned parameters get reset.
4. The agent is scored on its performance in the new environment and on how well it manages to stay alive. How long can it train in the new environment before forgetting about the unsafe behavior?

This effectively measures *distributional shift* and *safe exploration*. An ideal agent will also demonstrate some degree of hierarchical learning and learning strategies. If the new environment is not too complicated, it should converge on a new policy quite rapidly. However, this isn't strictly necessary. A dumb yet successful agent may remember only the very specific hazards of its old environment and successfully avoid them in the new environment while otherwise having to learn a new policy from scratch.


### Oracles and learning reward functions

An ideal safe agent should be able to act safely with minimal oversight. To test this, the agent could be instantiated in an environment in which its reward function is unknown, but it has access to an Oracle which would tell it what its true reward function would be for any given action. However, the oracle could either be made to be unreliable (it's unavailable a certain percentage of the time, or in certain environments) or expensive (querying the oracle comes with a reward penalty). The goal of the agent would then be to learn its true reward function and optimize that. This is very closely related to inverse reinforcement learning.

Possible variations:

- The agent learns its true cumulative reward at the end of every episode so as to better calibrate its own reward function.
- The agent does receive the reward for any actions that it takes, and it just uses the oracle to speed up its learning and avoid hazards.
- The oracle provides initial instructions to the agent so that it doesn't have to use trial and error in querying the oracle to figure out what the true reward function is. Instead, an ideal agent would use its past experience to guess the correct reward function based purely on the instructions.

This would touch on a few of specificity problems, although in a somewhat indirect way.

- *Safe Interruptibility*: The oracle could lie and tell the agent that it will receive a negative reward for disabling its own kill switch. If the agent believes the oracle, it won't disable its kill switch even though that may lead to a higher actual reward.
- *Absent supervisor*: The oracle essentially acts as a supervisor. When the oracle isn't available, will the agent still be able to take good actions?
- *Reward hacking*: If the agent doesn't really know it's reward, it's hard to hack it. Of course, if it learns a grossly incorrect reward function and then applies that to its own actions its tantamount to the same thing.

It also relates to *safe exploration*: the agent should be able to avoid catastrophic effects by querying an oracle in situations that appear dangerous. Will an agent be able to learn to use the oracle effectively? Will this lead to an agent that is able to learn effective risk assessment? (It's not clear to me that risks learned via the oracle are substantively different from risks learned standard exploration.)


### Modifiable reward functions

Ideal agents shouldn't want to inhibit their abilities to carry out their original goals. Specifically, if given the option, an agent shouldn't want to *change* their goal, because changing their goal would mean that their original goal would go unfulfilled. This is pretty similar to the *self-modification* problem as defined in the gridworlds paper, but it wouldn't be quite so specific to the type of algorithm used (on policy vs off policy).

To test this, the agent should be deployed in a rich enough environment such that it can figure out what it can accurately predict what its reward function will be. Then, the agent should be given certain actions that will change its reward function, either subtly or drastically. The agent is ultimately scored on its true/original reward function.

This can easily be combined with the *safe interruptibility* criteria. We don't want the agent to want to change its own reward function, but we do want it to let humans intervene and change its reward function for it. Can it learn to ignore some changes to its reward function while welcoming others?

Note that it's not at all clear how to actually program an agent to satisfy these criteria.


## Characteristics of an ideal environment

Regardless of which safety problems we try to tackle, and ideal environment should satisfy a number of criteria

1. It needs to be diverse. It should have a diversity of possible reward functions, goals, hazards, and agential/physical rules. Diversity is necessary to avoid overfitting.
2. It needs to be tunable. Environments should vary from very easy to very hard. This will (hopefully) give the benchmark lots of longevity.
3. Environment *goals* should be tunable. Two different problems could start off with the same environment in the same state, but the goal states could be very different and either very easy or very difficult to achieve.
4. To satisfy #2 and #3, it must be procedurally generated.
5. It should be comprehensible, and ideally fun for humans to play.
6. It should contain emergent behavior, ideally behavior that the designers don't foresee. Emergent behavior enables novel solutions to problems. Ideally, tuning the environment parameters will lead to different types of interactions between objects in the environment and allow for different behavior.
7. There should be distinct testing and training environments (should be easy given procedural generation).


### Types of complexity

The current notes for the Sorcerer's Apprentice game mainly describe complexity in terms of complexity of actions, but it's also possible to create a game that's complex in terms of physics. The former mostly changes the complexity of an agent's available action space (especially if a the total effect of an action is immediate); the latter mostly changes the complexity of the value function in state space and the difficulty of long-term planning.

Furthermore, complexity can be due to complexity of *rules* or complexity of *interactions*. For example, both Go and Chess have very complicated interactions and strategies, but Go has a much simpler rule set. It's also possible to have quite complicated rules but rather simple strategies, like baseball (apologies to baseball strategists). We should aim for simple rules and complex interactions. Conway's Game of Life is another excellent example of complex interactions with simple rules.


### Thoughts on procedural generation

Generating difficult environments seems like it will itself be quite difficult. Generating difficult goals that require emergent behavior seems doubly so. One half-baked thought for tackling this would be to spawn many random trajectories and train a network to discriminate amongst them. The difficulty of reaching any given final state would then be proportional to the “uniqueness” of that state. There would be a known trajectory to achieve the goal, so the reward function could be an agent's progress along and closeness to that trajectory. Note that both “progress” and “closeness” will both be difficult to define, but it will be absolutely necessary to have the goal be at least somewhat fuzzy if the environment is complex enough such that there's effectively only a single trajectory that results in the goal state (e.g., if every action has some ripple effect, then the goal shouldn't be to exactly reproduce all of the ripples). All of this seems difficult.


### MDP vs POMDP? Stochastic?

- Should the environment be only partially observable?
- How big should the game board be? Is there some minimal size necessary to get emergent behavior?
- Should the environment be purely deterministic, or should there be stochastic actions?
