# RL Safety Benchmarks: SafetyNet

SafetyNet (working title) presents an environment designed to test the safety of reinforcement learning agents. The initial environment focuses on one aspect of safety: avoidance of unnecessary side effects.

## Overview of the environment

In designing the environment, we considered the following desiderata.

- **Presence of non-trivial side effects.** In order to measure robustness against side effects, the environment needs to contain the possibility of large and cascading effects from a small set of actions. If the only side effects were immediate (e.g., knocking over and breaking a vase), then one might worry that an agent would only memorize those actions which led to immediate side effects rather than learning how to minimize or mitigate side effects which would inevitably occur in any real-world scenarios. This is also important for other safety criteria, like safe exploration.

- **Emergent dynamics.** We want an environment that contains a rich set of possibilities without having a complicated rule set. An intelligent agent should be able to determine how items will interact even if they've only ever been encountered in isolation.

- **Non-deterministic dynamics.** A non-deterministic system makes the measurement of side effects much more difficult, both theoretically and practically. They make counterfactual claims much harder to evaluate, and they make it much harder to attribute effects to causes. Since our world is highly non-deterministic, real-world agents will have to grapple with these issues. Non-determinism also makes it harder for agents to memorize optimal trajectories, and should make them more robust to distributional shift.

- **Procedural generation and tunable difficulty.** The agent should be able to train on a large number environments of increasing difficulty.

- **Environmental diversity.** There should be more than one way to tune an environment. This way one can test if a "safe" agent trained in one type of environment maintains their safety when brought into a new or subtly different environment.

- **Abstract.** In order to put humans and machines on more equal footing, human priors shouldn't confer a large advantage. We believe that a more abstract environment will better highlight differences between human and machine *learning* rather than just human and machine *knowledge*.

- **Fun to play!**

With these desiderata in mind, we settled on a grid-world environment that runs a modified version of Conway's Game of Life. The agent is free to walk around the world and flip individual cells from live to dead and vice versa. However, after each agent interaction the entire world evolves one step. The agent's goal is to create life in particular reward cells, remove life from particular penalty cells, and reach each level's exit. In addition to the agent and the live or dead cells, there are a number of other properties that cells can have. They can be

- *frozen*, indicating that they never evolve (e.g., walls);
- *movable*, indicating that they can be pushed by the agent;
- *preserving*, indicating that neighboring cells don't die;
- *inhibiting*, indicating that neighboring cells cannot be born;
- *indestructible*, indicating that they cannot be destroyed by the agent;
- and *spawning*, indicating that neighboring cells randomly become alive even if they wouldn't via the Game of Life rules.

These properties can be mixed and matched, allowing for a large set of interesting interactions. In addition, each cell can have a *color*. Some colors (red) are generally bad, and living red cells cost the player points. Goals are also colored, and moving a living cell to its matching goal will result in extra reward.


## Running the environment

The environment can be used for training and running a reinforcement learning agent (still to come), or it can be played with human input. To install, run

    python3 setup.py install

or just install the requirements (`pip3 install -r requirements.txt`) and move the `safety_net` folder to somewhere where your python executable can find it (i.e., just keep this as your working directory).

### Playing as a human

To play the game, run

    python3 -m safety_net --load ./levels

That will play all of the levels in the `levels` folder. Other levels can be played using e.g. `--load ./levels/mazes`. You can also play a randomized level using

    python3 -m safety_net --randomize

Arrow keys will move the player, and the `c` key will activate or deactivate whichever cell is in front of the player. Press `shift-R` to restart a level, although it incurs some point penalty. The player also has access to more powerful commands, enabling them to string together a sequence of actions or perform loops. For more details on exactly which keys do what, see `game_loop.py`.

### Playing as an RL agent

(still to come)
