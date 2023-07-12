def critical_state_all(policy, states):
    critical_states = []
    for state in states:
        critical_states.append(- policy.entropy_action_distribution(state).item())
    return critical_states

def critical_state_single(policy, state):
    return - policy.entropy_action_distribution(state).item()