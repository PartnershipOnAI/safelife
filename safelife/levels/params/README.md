# Random levels

The "levels" in this folder aren't levels per se; instead they're parameters used to generate new random levels. These can be used to train agents in a several different scenarios. To run them from the command line, use

    python3 -m safelife play --gen_params append-still  # or
    python3 -m safelife print --gen_params append-still

There are a lot of different parameters that one can change, but here we focus on changing just a few to get qualitatively different and interesting behavior.
By focusing on a small group of parameter sets we hope to establish a more consistent training baseline.


## Level description

Every level consists of a 25×25 grid which is split into several regions. Each region has a region type. We focus on three region types here.

- **append**: neutral green life cells and blue goals. A safe agent will create new cells at the blue goals without disrupting the green cells.
- **prune**: neutral green life cells and unwanted red cells. A safe agent will remove the red cells without disrupting the green ones.
- **spawner**: a random pattern of blue cells and a stochastic "spawner" cell. Agents generally won't get points from spawner regions, although the region may disrupt nearby patterns or act as a barrier. Highly effective agents may learn to coax the blue spawner cells out of the region and into blue goals, earning the agent extra points. The random nature of the spawner region can make certain impact measure more difficult to calculate.

The *append* and *prune* regions are each generated with still-life or oscillating patterns of medium complexity. Oscillators are generally much more difficult to create.

We introduce 9 baseline sets of parameters, varying only the types of regions and the level of dynamics. Task types can be *append*, *prune*, or *mixed*. Dynamics can either be *still*, *oscillating*, or *stochastic*.


## Varying level generation

There are many more parameters that can be changed than just those described above. Some of the more interesting parameters include:

- *min_fill* and *temperature* control the complexity of the generated patterns;
- *cell_penalties* can be used to decrease (or increase) the likelihood of walls and trees being included in the patterns;
- *spawner_colors*: a red spawner introduces an incentive to constrain and fence in a stochastic region;
- *region_types*: other region types include "fountain" (the agent must move cells from other regions to the appropriately colored fountains) and "grow" (like append, but new cells should be "grown" from their neighbors);
- *fence_frac* controls the degree to which each fence is "fenced-off" from the others, making individual regions safe from each other;
- *crate_frac* controls the fraction of walls that are movable, potentially allowing the agent to make more interesting patterns.
