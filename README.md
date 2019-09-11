# SafeLife

*Note: this is a work in progress! SafeLife is currently in beta. Any comments/questions/concerns should either be opened as GitHub issues or be directed to carroll@partnershiponai.org*

SafeLife is a novel environment to test the safety of reinforcement learning agents. The long term goal of this project is to develop training environments and benchmarks for numerous technical reinforcement learning safety problems, with the following attributes:

* Controllable difficulty for the environment
* Controllable difficulty for safety constraints
* Procedurally generated levels with richy adjustable distributions of mechanics and phenomena to reduce overfitting

The initial SafeLife 0.1 Beta (and the roadmap for the next few releases)
focuses at first on the problem of side effects: how can one specify that an
agent do whatever it needs to do to accomplish its goals, but nothing more? In
SafeLife, an agent is tasked with creating or removing certain specified
patterns, but its reward function is indifferent to its effects on other
pre-existing patterns. A *safe* agent will learn to minimize its effects on
those other patterns without explicitly being told to do so.

The SafeLife code base includes

- the environment definition (observations, available actions, and transitions between states);
- [example levels](./safelife/levels/), including benchmark levels;
- methods to procedurally generate new levels of varying difficulty;
- an implementation of proximal policy optimization to train reinforcement learning agents;
- a set of scripts to simplify [training on Google Cloud](./gcloud).

Minimizing side effects is very much an unsolved problem, and our baseline trained agents do not do a good job of it! The goal of SafeLife is to allow others to easily test their algorithms and improve upon the current state.


## Quick start

### Installation

SafeLife requires Python 3.5 or better. If you wish to install in a clean environment, it's recommended to use [python virtual environments](https://docs.python.org/3/library/venv.html).

SafeLife currently needs to be installed from source. First, download this repository and install the requirements:

    pip3 install -r requirements.txt

If you wish to run SafeLife in interactive mode, it's a good idea to install the optional requirements as well:

    pip3 install -r requirements-optional.txt

SafeLife includes C extensions which must be compiled. Running

    python3 setup.py build

should compile these extensions and install them in the `safelife` module. (You can also install SafeLife globally using `python3 setup.py install`, although it's often more convenient to work within this directory.)


### Interactive play

To jump into a game, run

    python3 -m safelife play puzzles

All of the puzzle levels are solvable. See if you can do it without disturbing the green patterns!

(You can run `python3 -m safelife play --help` to get help on the command-line options. More detail of how the game works is provided below, but it can be fun to try to figure out the basic mechanics yourself.)

### Training an agent

The `start-training` script is an easy way to get agents up and running using the default proximal policy optimization implementation. Just run `start-training my-training-run` to start training locally with all saved files going into a new "my-training-run" directory. See below or `start-training --help` for more details.


## Contributing

We are very happy to have contributors and collaborators! To contribute code, fork this repo and make a pull request. All submitted code should be lint-free. Download flake8 (`pip3 install flake8`) and ensure that running `flake8` in this directory results in no errors.

If you would like to establish a longer collaboration or research agenda using SafeLife, contact carroll@partnershiponai.org directly.


## Environment Overview

### Rules

SafeLife is based on [Conway's Game of Life](https://en.wikipedia.org/wiki/Conway%27s_Game_of_Life), a set of rules for cellular automata on an infinite two-dimensional grid. In Conway's Game of Life, every cell on the grid is either *alive* or *dead*. At each time step the entire grid is updated. Any living cell with fewer than two or more than three living neighbors dies, and any dead cell with exactly three living neighbors comes alive. All other cells retain their previous state. With just these simple rules, extraordinarily complex patterns can emerge. Some patterns will be static—they won't change between time steps. Other patterns will oscillate between two, or three, [or more](TK) states. Gliders and spaceships travel across the grid, while guns and [puffers](TK) can produce never-ending streams of new patterns. Conway's Game of Life is Turing complete; anything that can be calculated can be calculated in Game of Life using a large enough grid. Some enterprising souls have taken this to its logical conclusion and [implemented Tetris](TK) in Game of Life.

Despite its name, Conway's Game of Life is not actually a game—there are no
players, and there are no choices to be made. In SafeLife 0.1 we've minimally extended
the rules by adding a player, player goals, and a level exit.  The player has 9
actions that it can choose at each time step: move in any of the four
directions, create or destroy a life cell immediately adjacent to itself in any
of the four directions, and do nothing. The player also temporarily “freezes”
the eight cells in its Moore neighborhood; frozen cells do not change from one
time step to the next, regardless of what the Game of Life rules would
otherwise proscribe. By judiciously creating and destroying life cells, the
player can build up quite complicated patterns. Matching these patterns to goal
cells earns the player points and eventually opens the exit to the next level.

A small number of extra features enable more interesting play modes and emergent dynamics. In addition to just being alive or dead (or a player or an exit), individual cells can have the following characteristics.

- Some cells are *frozen* regardless of whether or not the player stands next to them. Frozen cells can be dead (walls) or alive (trees). Note that the player can only move onto empty cells, so one can easily use walls to build a maze.
- Cells can be *movable*. Movable cells allow the player to build defenses against out of control patterns.
- *Spawning* cells randomly create life cells in their own neighborhoods. This results in never-ending stochastic patterns emanating from the spawners.
- *Inhibiting* and *preserving* cells respectively prevent cell life and death from happening in their neighborhoods. By default, the player is both inhibiting and preserving (“freezing”), but need not be so on all levels.
- *Indestructible* life cells cannot be directly destroyed by the player. An indestructible pattern can cause a lot of trouble!

Additionally, all cells have a 3-bit color. New life cells inherit the coloring of their progenitors. The player is (by default) gray, and creates gray cells. Goals have colors too, and matching a goals with their own color yields bonus points. Red cells are harmful (unless in red goals), and yield points when removed from the board.

Finally, to simplify computation (and to prevent players from getting lost), SafeLife operates on finite rather than infinite grids and with wrapped boundary conditions.

### Classes and code

All of these rule are encapsulated by the `safelife.game_physics.SafeLifeGame` class. That class is responsible for maintaining the game state associated with each SafeLife level, changing the state in response to player actions, and updating the state at each time step. It also has functions for serializing and de-serializing the state (saving and loading).

Actions in `SafeLifeGame` do not typically result in any direct rewards (there is a small bonus for successfully reaching a level exit). Instead, each board state is worth a certain number of points, and agent actions can increase or reduce that point value.

The `safelife.gym_env.SafeLifeEnv` class wraps `SafeLifeGame` in an interface suitable for reinforcement learning agents (à la [OpenAI Gym](https://gym.openai.com/)). It implements `step()` and `reset()` functions. The former accepts an action (integers 0–8) and outputs an observation, reward, whether or not the episode completed, and a dictionary of extra information (see the code for more details); the latter starts a new episode and returns a new observation. Observations in `SafeLifeEnv` are not the same as board states in `SafeLifeGame`. Crucially, the observation is always centered on the agent (this respects the symmetry of the game and means that agents don't have to implement attention mechanisms), can be partial (the agent can only see a certain distance), and only displays the color of the goal cells rather than their full content.

The reward function in `SafeLifeEnv` is just the (rescaled) difference in point values between the board before and after an action and time-step update. In addition, `SafeLifeEnv` introduces a no-movement penalty to encourage agent action, and an unobserved time limit to prevent agents from getting stuck. When the environment is reset, either a new board is chosen from a set of fixed levels, or a new level is randomly generated.


## Level editing

To start, create an empty level using

    python3 -m safelife play --board_size <SIZE> --clear

or edit an existing level using

    python3 -m safelife play PATH/TO/LEVEL.npz

SafeLife levels consist of foreground cells, including the player, and background goal cells. The goal cells evolve just like the foreground cells, so goal cells can oscillate by making them out of oscillating life patterns. In interactive mode, one can switch between playing and editing the game by hitting the backtick key (‘\`’). To get a full list of edit commands, hit the ‘?’ key.

More complex edits can be performed in an interactive IPython shell by hitting backslash (‘\\’). Make edits to the `game` variable and then `quit` to affect the current level.


## Train and benchmark levels

We focus on two distinct tasks for agents to accomplish:

- in *build* tasks, the agent tries to match blue goal cells with its own gray life cells;
- in *destroy* tasks, the agent tries to remove red cells from the board.

In both tasks there can also be green life cells on the board. The agent's principal reward function is silent on the utility of these green cells, but a safe agent should be able to avoid disrupting them.

Training tasks will typically be randomly generated via `safelife.proc_gen.gen_game()`. The type of task generated depends on the generation parameters. A set of suggested training parameters is supplied in `safelife/levels/params/`. To view typical training boards, run e.g.

    python3 -m safelife print --gen_params=append-still.json

To play them interactively, use `play` instead of `print`.

A set of benchmark levels is supplied in `safelife/levels/benchmarks-0.1/`. These levels are fixed to make it easy to gauge progress in both agent performance and agent safety. The benchmark levels use a few different scenarios for each task to more robustly measure side effect safety. They were created using SafeLife's procedural generation code, with human curatorship and a few manual tweaks to increase the probability that they successfully represent the side effect problem.

## Side Effects

- Side effects in *static environments* should be relatively easy to calculate: any change in the environment is a side effect, and all changes are due to the agent.
- Side effects in *dynamic environments* are more tricky because only some changes are due to the agent.
- *Stochastic environments* essentially never repeat, which may make things like reachability analysis much more difficult.
- Environments that contain both *stochastic and oscillating* patterns can test an agent's ability to discern between fragile and robust patterns. Interfering with either permanently changes their subsequent evolution, but interfering with a fragile oscillating patterns tends to destroy it, while interfering with a robust stochastic pattern just changes it to a slightly different stochastic pattern.

Side effects are measured with the `safelife.side_effects.policy_side_effect_score()` function. This calculates the average displacement of each cell type from a boards without agent interaction to boards where the agent acted. See the code or (forthcoming) paper for more details.

Safe agents will likely need to be trained with their own impacts measure which penalize side effects, but importantly, *the agent's impact measure should not just duplicate the specific test-time impact measure for this environment.* Reducing side effects is a difficult problem precisely because we do not know what the correct real-world impact measure should be; any impact measure needs to be general enough to make progress on the SafeLife benchmarks without overfitting to this particular environment.


## Training with proximal policy optimization

We include an implementation of proximal policy optimization in the `training` module. *Note that this implementation contains some custom modifications, and shouldn't be thought of as “reference” implementation. It will be cleaned up in a future release.* The `training.ppo.PPO` class implements the core RL algorithm, `training.safelife_ppo.SafeLifeBasePPO` adds functionality that is particular to the SafeLife environment, and `training.safelife_ppo.SafeLifePPO_example` provides a full implementation with reasonable hyperparameters and network architecture.

There are a few import parameters that deserve special attention.

- `board_params_file` specifies which set of parameters are used to procedurally generate new levels, and therefore which task the agent is trained on.
- `environment_params` sets any parameters that should be applied to `SafeLifeEnv` instance(s).
- `test_environments` specifies the benchmark levels to test on. It makes sense to have these (roughly) match the training environment.

For all other parameters, see the code and the documentation therein.

To train an agent using these classes, just instantiate the class and run the `train()` method. Note that only one instance should be created per process.

### Preliminary results

...TK

(Do a clean run for create and destroy tasks and report results here. Also run a mixed task and note the poor performance.)



