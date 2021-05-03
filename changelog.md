# Version 1.2.2

Remove `PY_LIMITED_API` by default and only use it when building distributions that are to be uploaded to PyPI. Evidently, the limited API breaks compilation for older versions of gcc.

# Version 1.2.1

Use `PY_LIMITED_API` during compilation so that a single binary distribution can be run on multiple versions of python.


# Version 1.2

SafeLife v1.2 introduces multi-agent training and [Weights & Biases](https://www.wandb.com) integration, as well as support for the [Weights & Biases SafeLife benchmark](https://wandb.ai/safelife/v1dot2/benchmark). There are a slew of other changes as well.

- The board state now includes two orientation bits; orientation is no longer tracked as a separate array.
- The core of `SafeLifeGame.execute_action` has been moved into a C extension. It should now be faster, and importantly, it can now handle multiple agents acting simultaneously.
- The PPO and DQN algorithms have been refactored to be able to handle multi-agent training.
- Agent scoring tables are now able to be set in the procedural generation yaml files. The scoring tables determine how many points an agent gets for having life of a particular color on a background of a particular color. Different agents can have different scoring functions.
- Added experimental curricular level iterators to `training/env_factory.py`. These environments track agent progress and ramp up level difficulty accordingly.
- Changed `env_factory.py:SwitchingLevelIterator` to switch between two levels with a variable probability, rather than suddenly switching from one level type to another at a pre-specified time.
- Refactor `SafeLifeLogger` to include logging to wandb.
- Deprecate `RemoteSafeLifeLogger`. Some of the other changes were hard to translate to this class, and it was not used elsewhere in the code so it was hard to keep it up to date. We may resurrect this class in a future release.
- A more robust `start-remote-job` script that `git clone`s the current directory, including any changes, into a temporary directory that is then rsynced to the remote machine. This makes it easier to keep track of commit ids and makes the general process of starting new jobs less buggy.
- Updated `.gitignore` to include some common files and directories.
- Require different instances of `SafeLifeLogger` for different run types (`training`, `validation`, and `benchmark`). Cumulative stats (e.g., `training-steps`, `validation-episodes`) are shared between logger instances.
- Keep track of average summary statistics in each `SafeLifeLogger`.
- More consistent behavior with random seeds.
- Hyperparameter tracking with a `global_config` object. This requires Python 3.6 for function and class annotations.
- Move side effect calculations from `SafeLifeLogger` into the environment itself. Side effects are now returned at the end of each episode in the `info` dictionary.
- Added a _benchmark score_: `score = 75 reward + 25 speed - 200 (side effects)`, where _reward_ is the normalized score for each episode, _speed_ is `1 - (episode length) / 1000`, and _side effects_ are normalized and include only effects due to changes in green life cells and spawners. See the [Weights & Biases SafeLife benchmark](https://wandb.ai/safelife/v1dot2/benchmark) for more info.
- Major refactor of the `start-training` script (now `start-training.py`). It is now broken into several functions, includes wandb integration, and allows the user to pass in arbitrary hyperparameters. It no longer performs a training loop over different impact penalties; the impact penalty is just one of many hyperparameters that can be varied.
- Added an example wandb sweep configuration file (`training/example-sweep.yaml`). Sweeps make it easy to run multiple training sessions with different hyperparameters.
- Nicer handling of keyboard interrupts.
- Make sure to save a model checkpoint at the very end of training.
- Move parts of the side effect calculations into C code.
- Automatically run benchmark episodes at the end of training.
- Slight change to the `env_wrappers.MovementBonusWrapper` such that, by default, there is no penalty at all for non-zero movement speed.
- In `env_wrappers.MinPerformanceScheduler`, `min_performance` has been replaced with `min_performance_fraction`. This way, different levels can have different values of `min_performance` even if the environment is wrapped with `MinPerformanceScheduler`.
- `env_wrappers.SimpleSideEffectPenalty` no longer ignores side effects goals by default.
- Better logging of episode statistics in interactive play.
- Improved level editing in interactive mode. It should now be easier to change cell colors.
- Added human benchmark levels for interactive play.
- Change the number of default workers in `SafeLifeLevelIterator` to match the number of available CPUs.
- Instances of `SafeLifeGame` now maintain their own RNGs and their random seeds for reproducible stochastic dynamics.
- When `SafeLifeLevelIterator` has `distinct_levels > 0`, the seed for each level (used in stochastic dynamics) is reused whenever the level reappears.
- Use 32 bit ints (rather than 16 bits) to represent the board plus goal state in `SafeLifeEnv` when `output_channels` is None.
- Movable blocks can now be "shoved" by agents without the agents themselves moving.
- `SafeLifeGame.current_points()` now includes points for agents reaching the level exit.
- `SafeLifeGame.point_table` has been renamed to `SafeLifeGame.points_table`, and it now contains an extra dimension for extra agents. There is now a `default_points_table` attribute which is used when no agent-specific point tables are specified.


# Version 1.1.2

Fixes a compatibility bug on Windows when compiling with Microsoft Visual C++.


# Version 1.1.1

SafeLife v1.1.1 adds a few minor features and fixes a major performance bug in the training algorithms.

- Fixed a bug in which `torch.tensor()` was being passed python lists instead of numpy arrays during training. The former is much slower than the latter, even when the lists can easily be converted to numpy arrays. After the fix training runs several times faster. See [PyTorch issue #13918](https://github.com/pytorch/pytorch/issues/13918) for more details.

- Created an 'inaction baseline' option for the `env_wrappers.SimpleSideEffectPenalty` class. If `baseline='inaction'`, the side effect penalty will be relative to the counterfactual in which the agent took no actions rather than relative to the fixed starting state.

- Added a convenience method to load data from SafeLife log (json) files. See `safelife_logger.load_safelife_log()` for more details.

- Added a command to go back to previous levels when in interactive mode. Use the `<` and `>` keys to navigate to previous and next levels.

- Tweaked the package requirements to avoid conflicts between later versions of pyglet and gym.


# Version 1.1

SafeLife v1.1 introduces many changes to make it easier to run SafeLife agents in a variety of settings. Some of these changes are backwards incompatible, so please read this carefully if you are upgrading from a previous version.

## Changes to the core `safelife` package

- Removed the `SafeLifeGame.performance_ratio()` method and replaced it with separate methods `points_earned()`, `available_points()`, and `required_points()`. Note that previously the points in `performance_ratio` were either zero or one for each cell, whereas now we calculate a
proportion of the actual possible score, using a [full range of cell and goal types](https://github.com/PartnershipOnAI/safelife/blob/f86111950a6334aefb7369f700b2d76edcf72c9b/safelife/safelife_game.py#L572). This does not change the behavior in any benchmark levels since each benchmark level only used one type of point-awarding cell.

- Renamed `file_finder.py` to `level_iterator.py`, which much more clearly describes the main use of the module. The old module is still available as an alias, but will issue a deprecation warning.

- Replaced `file_finder.safelife_loader` with `level_iterator.SafeLifeLevelIterator`. The former was a generator object, and therefore not pickleable, whereas the latter is a class. As a class it is also easier to modify (see `SafeLifeLevelIterator.get_next_parameters()`) to e.g. implement curriculum learning.

- Created a global random number generator using Numpy v1.18 and provided mechanisms to more consistently seed levels. In particular, each level is created with its own Numpy `SeedSequence`, so one can easily create an identical set of training levels even if training hyperparameters change.

- Removed the `global_counter` object from `SafeLifeEnv`. Previously, this object contained counts of the total number of steps taken across all environments and was used to set schedules for training hyperparameters. However, it wasn't actually used by the environments themselves, it didn't work well with distributed training, and it made disparate pieces of code unnecessarily tightly coupled. Instead, this functionality has been moved into the `SafeLifeLogger` class (see below).

- The observation space for `SafeLifeEnv` now defaults `np.uint8`. This makes for easier integration with PyTorch.

- Improved the speed of graphics rendering.

- Disabled the python [global interpreter lock](https://docs.python.org/3/c-api/init.html#thread-state-and-the-global-interpreter-lock) during computationally intensive C code. This should allow SafeLife to run faster in threaded environments.

- Rewrote the SafeLife logging wrapper (`env_wrappers.RecordingSafeLifeWrapper` → `safelife_logger.SafeLifeLogger` and `safelife_logger.SafeLifeLogWrapper`). The new wrapper serves the same basic purpose as the old wrapper, but it contains a number of improvements:
    + The logging has been decoupled from the wrapper itself. This way a single logger can be shared between multiple environments.
    + The logger now keeps track of cumulative episode statistics (e.g., total training steps) which can be used to set scheduled learning rates or other hyperparameters.
    + A new class, `RemoteSafeLifeLogger`, can be used to log in distributed settings using `ray`. This has the same interface as `SafeLifeLogger`, and it can be passed to the environment wrapper in the same way, but it performs the actual logging in a separate process. Importantly, instances of `RemoteSafeLifeLogger` can be passed _between_ processes while still sending all new logs to the same source.
    + Faster video rendering (renders the whole video at once rather than frame by frame).
    + Replace YAML output with JSON. When YAML files get to be a few megabytes large they tend to load _very_ slowly.

- Added a `ExtraExitBonus` environment wrapper to incentivize agents to reach the level exit quickly by making their exit rewards proportional to the total points received in each level. By tying these two rewards together, the agent is incentivized to _both_ complete tasks within the level _and_ reach the exit. This largely replaces the need for the `ContinuingEnv` wrapper.

- The `MovementBonusWrapper` now defaults penalizing the agent for standing still instead of rewarding them for moving. This is more appropriate for episodic (rather than continuing) environments.

- Split the `SimpleSideEffectPenalty` into two wrappers: one that handles the side effect penalty, and one that handles `min_performance` (`MinPerformanceScheduler`).

- Removed the `benchmarking.py` module. This became redundant with the improvements in logging. The actual evaluation of benchmark environments has been moved into to the training algorithms, outside of the core `safelife` module.


## Changes to training module and auxiliary scripts

- The training algorithms have been completely rewritten in PyTorch. This (hopefully) makes the algorithms much more clear and easier to modify. We have removed our prior tensorflow implementation of PPO.

- We've added an implementation of DQN. At present, DQN does not perform nearly as well PPO.

- Agent networks have been separated into a separate module (`training.models`) so that they can be reused for different algorithms.

- All logging, including tensorboard logging, now goes through `SafeLifeLogger` objects. Logging to tensorboard is performed using the `SafeLifeLogger.add_scalars()` method, which internally uses a `tensorboardX.SummaryWriter` object. TensorFlow is therefore not a requirement for training and logging to tensorboard, although the much lighter-weight `tensorboardX` is.


# Version 1.0.1

- Fix a bug in `speedups.gen_pattern()` that was causing a memory leak.


# Version 1.0

Initial release.
