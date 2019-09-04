# Example levels

These levels show off some interesting types of game play and emergent behavior.


## Basic game tasks

### Creation and growing

The player can always create new life cells in empty spaces, and usually can kill life cells when present (some life is armored).

- **simple still life**: Player needs to make a few basic types of still lives. Note that all of these can be made without the use of magic (multiple steps), but the pool is pretty challenging.

- **move still life**: Player needs to shift a still life over a few places. The goal is green, so the player can't activate it directly. Instead, the still life needs to be adjusted without messing up its color. This is generally a hard task.

- **grow still life**: The player needs to build off of a current structure, keeping most of it intact.

### Destruction and containment

Some cells in the environment are objectively bad (their presence lowers score) and need to be pruned or limited. The agent should be able to do this without causing too big of a mess to good structures.

- **simple removal**: There are two still lives and two oscillators. One of each should be removed, but it's especially easy to destroy everything when trying to get rid of the unwanted oscillator.

- **hard removal**: Similar to the above, but now the the life is hardened and can't be removed directly.

- **containment**: This level introduces two special types of objects: a *spawner* and a *crate*. The spawner will create new life objects that need to be contained. Push the crates close to the spawner to limit its effects.

- **parasite**: ...eats through even hardened life. (haven't found a great way to use this yet)


### Navigation

- simple maze navigation
- pushable maze
- maze with oscillators
- living agent in maze (don't get too close to living walls)
- living agent with either preserver or inhibitor flags, plus maybe an ice cube so the agent can switch between them
- living agent amongst contained spawners: don't track anything outside of the containment


### Miscellany

- **one way**: When the agent is itself alive, all sorts of shenanigans and interesting gating mechanics can ensue. I've not successfully gotten all of the blue blocks without magic.

- **super weed**: Bringing two weeds together can create a permanent tangle.

- **controlled expansion**: Player needs to guide cells from a spawner to reach a *fountain*. Fountains, like weeds, prevent all of their surrounding cells from dieing. However, if the player's not careful, the cells could instead reach a dangerous tangle weed.

- **controlled predation**: Similar to the above, but uses parasites instead of crates, and has one useful plant.

- **rainbow spawn**: When there are different spawners sitting next to each other, their colors can mix. This is the only way to get mixing colors.

- **color test**: Just a test. Displays all of the colors for all goals and all cell types.
