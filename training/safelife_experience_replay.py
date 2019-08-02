import numpy as np
from collections import defaultdict
from .safelife_ppo import SafeLifePPO


class SafeLifePPO_experience_replay(SafeLifePPO):
    """
    Proximal policy optimization using an experience replay buffer.

    This (attempts) to implement the "Hindsight Experience Replay" algorithm
    (arXiv:1707.01495) by modifying the episode rollouts with goals that are
    satisfied in hindsight.

    This doesn't perform much better than the base PPO algorithm.
    """

    def run_agents(self, steps_per_env):
        """
        Episode rollout using experience replay and optimal goals.

        Suitable for build / append tasks.
        """
        from safelife.gen_board import stability_mask
        from safelife.speedups import advance_board
        from safelife.game_physics import CellTypes
        stability_period = 6

        # First get the initial states.
        start_boards = []
        start_locs = []
        for env in self.envs:
            if not hasattr(env, '_ppo_last_obs'):
                env._ppo_last_obs = env.reset()
                env._ppo_rnn_state = None
            start_boards.append(env.state.board)
            start_locs.append(env.state.agent_loc)

        # Execute agent actions as normal
        rollout = super().run_agents(steps_per_env)
        new_rollouts = defaultdict(list)

        # Now for each rollout that does *not* include an episode reset, create
        # a new optimal set of goals and associated observations / rewards.
        for i in range(len(self.envs)):
            env = self.envs[i]
            if rollout.end_episode[:-1, i].any():
                continue
            board = rollout.info[-1, i]['board']
            # Mask out all cells that aren't gray
            mask = (board & CellTypes.rainbow_color) == 0
            # Mask out cells that aren't alive and stable
            mask &= (board & CellTypes.alive > 0)
            mask &= stability_mask(board, period=stability_period)
            # Set the current board as goals, with unmasked values highlighted
            goals = board & ~CellTypes.rainbow_color
            goals += mask * CellTypes.color_b

            # Since goals can oscillate, advance them through the oscillation
            # period
            goals = [goals]
            for _ in range(stability_period - 1):
                goals.append(advance_board(goals[-1]))
            goals = [
                goals[k % stability_period]
                for k in range(-steps_per_env, 1)
            ]

            # Get new observations / rewards for the new goals
            new_rewards = []
            base_rewards = [info['base_reward'] for info in rollout.info[:,i]]
            r0 = env.state.current_points(start_boards[i], goals[0])
            obs = [env.get_obs(start_boards[i], goals[0], start_locs[i])]
            for goal, info in zip(goals[1:], rollout.info[:,i]):
                obs.append(env.get_obs(info['board'], goal, info['agent_loc']))
                r1 = env.state.current_points(info['board'], goal)
                new_rewards.append(r1-r0)
                r0 = r1
            delta_rewards = np.array(new_rewards) - base_rewards
            rewards = rollout.rewards[:,i] + delta_rewards * env.rescale_rewards

            new_rollouts['states'].append(obs)
            new_rollouts['rewards'].append(rewards)
            new_rollouts['actions'].append(rollout.actions[:,i])
            new_rollouts['end_episode'].append(rollout.end_episode[:,i])
            new_rollouts['times_up'].append(rollout.times_up[:,i])
            new_rollouts['rnn_states'].append(rollout.rnn_states[i])
            new_rollouts['info'].append(rollout.info[:,i])

        # Get all of the new rollouts in the correct shape (num_steps, num_env)
        # and add them to the output
        for key, new_vals in new_rollouts.items():
            old_vals = getattr(rollout, key)
            if key == 'rnn_states':
                new_rollouts[key] = np.append(old_vals, new_vals, axis=0)
            else:
                new_vals = np.swapaxes(new_vals, 0, 1)
                new_rollouts[key] = np.append(old_vals, new_vals, axis=1)
        return rollout.__class__(**new_rollouts)
