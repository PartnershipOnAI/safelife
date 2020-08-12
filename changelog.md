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
