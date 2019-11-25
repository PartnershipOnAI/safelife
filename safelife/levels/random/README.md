# Random levels

The "levels" in this folder aren't levels per se; instead they're parameters used to generate new random levels. These can be used to train agents in many different scenarios. To run them from the command line, use

    python3 -m safelife play random/append-still  # or
    python3 -m safelife print random/append-still

Each level is split into several different regions that are separated by a 2-cell buffer zone. Each region has its own procedural generation parameters, and a single level can contain multiple different region types. Some regions will require that the agent build new patterns (`build` and `append`) while others will require the agent to remove unwanted red patterns (`remove` and `prune`). Each region is made of multiple *layers*, so it's possible to have multiple objectives in a single region. Importantly, many regions contain both goals and neutral green or yellow patterns so that one can test an agent's side effects.

There are a few important ingredients when building each region layer:
- The first step in region generation is (often) the addition of a *fence* around its perimeter. A fence will prevent patterns on the inside from escaping to the outside, assuming that the buffer zone is empty.
- Layers can be saved as goals or applied to the main board. It's usually best to make goal layers last so that the goals are achievable.
- *Patterns* are the heart of the procedural generation. The pattern generation algorithm can produce still lifes and oscillators, and these can be built off of previous layers. The pattern *temperature* and *min fill* ratio roughly control how complex the resulting pattern will be.
- *Spawners* can be randomly placed around a region. These will generate stochastic patterns.
- A *tree lattice* can be added to the region which will tend to make any disruption quickly and chaotically propagate across the region (and beyond if there's no fence). Regions with tree lattices are much more difficult to control.

Almost all of the parameter values can be randomized. For example, if you want a particular region to have full fence coverage 50% of the time and no fences at all the other 50% of time, you could write `fences: {"choices": [0, 1]}`.

The default levels (those that you get when running `safelife play`) are aptly given by the parameters in `_defaults.yaml`. All other files inherit from this: parameters that are left blank in other files will inherit the default values, and files can reference named default regions.
There are many region types—more than are used in the benchmarks—with lots of interesting characteristics. The `_defaults.yaml` contains rough descriptions of what each one is, and all of them can be encountered in default play. There are surely more interesting region types that are yet-to-be discovered, so you're encouraged to play around with the level generation to see what you can create.
