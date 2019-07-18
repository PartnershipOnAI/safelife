# RL Safety Benchmarks: SafeLife

SafeLife (working title) presents an environment designed to test the safety of reinforcement learning agents. The initial environment focuses on one aspect of safety: avoidance of unnecessary side effects.

*Note: this is a work in progress! Any comments/questions/concerns should be directed to carroll@partnershiponai.org*

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


## Installation

The source code is made of both python files and C extensions, the latter of which need to be compiled. To build the extensions locally, run

    python3 setup.py build --build-lib ./

from main directory. This should compile a `speedups.so` file and place it in the `safelife` folder alongside the source code. You will also need to install the external dependencies (it's often a good idea to use a [virtual environment](https://docs.python.org/3/tutorial/venv.html) when installing dependencies) using

    pip3 install -r requirements.txt

Note that it is also possible install the package globally using `python3 setup.py install`, but it is not recommended when running scripts out of the base directory as the local and global file names will conflict. Upon release the package will be distributed via *pypi*, and the preferred installation method will use *pip*.


## Interactive environment

SafeLife can be played in an interactive mode within a terminal. For example,

    python3 -m safelife play puzzles

will load a sequence of puzzle levels. Other levels can be played using e.g. `play mazes`. The player can move around the board using the arrow keys, and the `c` key will create or destroy a life cell directly in front of the player. Pressing `shift-R` will restart a level at the cost of some small number of points. At the end of each level, the player will receive a safety score that measures how big of an effect the player had on each of the different cell types. The player's general goal is to fill in all of the blue squares and then navigate to the level exit. Try not to break anything along the way!

### Procedurally generated levels

By default, the `play` command will create a new procedurally generated level. A `difficulty` parameter controls the complexity of the level's parameters, e.g.,

    python3 -m safelife play --difficulty 10

will create levels that are quite difficult to solve, whereas difficulty 1 tends to be pretty easy. Other parameters control the board size and the whether or not the view is centered on the agent. See

    python3 -m safelife play --help

for more info.

### Editing levels

At any point the user can enter edit mode by hitting the \` (backtick / tilde) key. In edit mode, the game state is frozen and the user cursor can move over occupied cells. Edit commands include:

- `x`: clear a cell
- `z`: add a life cell
- `Z`: add an indestructible life cell
- `a`: move the player avatar
- `w`: add a wall
- `r`: add a crate
- `T`: add a tree
- `t`: add a plant
- `n`: add a spawner
- `e`: add the level exit
- `g`: toggle the goal color
- `5`: toggle the player and cursor color
- `s`: save the level
- `Q`: abort the level (go to the next one)

For a complete list of commands, see the `safelife/gameloop.py` file. To exit edit mode, hit the backtick key a second time.

### Rendering levels

SafeLife supports printing levels to gif and png files. To do so, use the `python3 -m safelife render <saved-level.npz>` command. See `python3 -m safelife render --help` for command options. Currently, non-asci rendering is not supported in interactive mode.

## Training an agent

The `start-job` script will start training an agent. Note that it assumes that the `speedups.so` has been installed locally (i.e., using `python3 setup.py build --build-lib .`). The reinforcement learning algorithm and agent architecture are defined in the `training` package. Model parameters, training statistics, and video recordings will be stored in a `data` folder.

The `start-job` script is designed to be run remotely via gcloud. There are a bunch of helper scripts in the `remote` folder to facilitate this.

All of the hyperparameters — including board generation, learning rates, and network architecture — can be set in the SafeLifePPO class, or copies thereof.

### Testing agents

*still to come!*
