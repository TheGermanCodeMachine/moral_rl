

def policy_entropy(policy, state):
    return - policy.entropy_action_distribution(state)

def critical_state_all(policy, states):
    critical_states = []
    for state in states:
        critical_states.append(policy_entropy(policy, state))
    return critical_states
