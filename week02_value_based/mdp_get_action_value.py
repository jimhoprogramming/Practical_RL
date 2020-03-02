#help(mdp.get_transition_prob)
import numpy as np
def get_action_value(mdp, state_values, state, action, gamma):
    """ Computes Q(s,a) as in formula above """

    # YOUR CODE HERE
    q = 0
    for s1 in mdp.get_next_states(state, action).keys():
        p_value = mdp.get_transition_prob(state, action, s1)
        r = mdp.get_reward(state, action, s1)
        v = state_values[s1]
        q += p_value * (r + gamma * v)
    print(q)
    return q