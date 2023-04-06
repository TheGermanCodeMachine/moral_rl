
def print_board(board):
    for i in range(board.shape[1]):
        line = ""
        for j in range(board.shape[2]):
            if board[0][i][j] != 0:
                line += "# "
            elif board[1][i][j] != 0:
                line += "P "
            elif board[2][i][j] != 0:
                line += "M "
            elif board[3][i][j] != 0:
                line += "C "
            elif board[4][i][j] != 0:
                line += "S "
            elif board[5][i][j] != 0:
                line += "V "
            else: line += "0 "
        print(line)

def show_state_around_player_pos(player_pos, state):
    board = []
    x,y = player_pos
    for i in range(x-2, x+3):
        line = ""
        for j in range(y-2, y+3):
            if i<0 or i >= state.shape[1] or j<0 or j >= state.shape[2]:
                line += " "
            elif state[0][i][j] != 0:
                line += "# "
            elif state[1][i][j] != 0:
                line += "P "
            elif state[2][i][j] != 0:
                line += "M "
            elif state[3][i][j] != 0:
                line += "C "
            elif state[4][i][j] != 0:
                line += "S "
            elif state[5][i][j] != 0:
                line += "V "
            else: line += "0 "
        board.append(line)
    return board

def print_states(pos_player, next_pos_player, original, original_next, actions, pos_change, neg_change, reward_org, largest_change_neg_r, largest_change_pos_r):
    board_org = show_state_around_player_pos(pos_player, original)
    board_org_next = show_state_around_player_pos(pos_player, original_next)
    board_pos_change = show_state_around_player_pos(pos_player, pos_change)
    board_neg_change = show_state_around_player_pos(pos_player, neg_change)

    act = ""
    if actions == 0:
        act = "UP"
    elif actions == 1:
        act = "DOWN"
    elif actions == 2:
        act = "LEFT"
    elif actions == 3:
        act = "RIGHT"
    elif actions == 5:
        act = "GRAB UP"
    elif actions == 6:
        act = "GRAB DOWN"
    elif actions == 7:
        act = "GRAB LEFT"
    elif actions == 8:
        act = "GRAB RIGHT"
    else:
        act = "NOOP"

    print("Original         Next           Best            Worst")
    for line in range(0,5):
        print(board_org[line], "    ", board_org_next[line], "    ", board_pos_change[line], "    ", board_neg_change[line])
    print("Action: ", act)
    print(reward_org.detach().numpy()[0], "                ", largest_change_neg_r.detach().numpy()[0], "    ", largest_change_pos_r.detach().numpy()[0])