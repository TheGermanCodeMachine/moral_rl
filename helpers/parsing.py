def parse_attributes(arg_str):
    attr_map = {
        'p': 'proximity',
        'v': 'validity',
        'c': 'critical_state',
        'd': 'diversity',
        'b': 'baseline',
    }
    return [attr_map[c] for c in arg_str if c in attr_map]

def sort_args(arg_str):
    attr_map = {
        'p': 0,
        'v': 1,
        'c': 2,
        'd': 3,
    }
    add_pos = [(c, attr_map[c]) for c in arg_str if c in attr_map and c != 'b']
    add_pos.sort(key=lambda x: x[1])
    return ''.join([c[0] for c in add_pos])