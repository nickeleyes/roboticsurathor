def possible_moves(board, turn):
    moves = []
    for i, char in enumerate(board):
        if char == "0":
            moves.append((i, board[:i] + turn + board[i + 1:]))
    return moves


def has_win(board, turn):
    lines = [
        (0, 1, 2), (3, 4, 5), (6, 7, 8),
        (0, 3, 6), (1, 4, 7), (2, 5, 8),
        (0, 4, 8), (2, 4, 6),
    ]
    for a, b, c in lines:
        if board[a] == turn and board[b] == turn and board[c] == turn:
            return True
    return False


def other_turn(turn):
    return "2" if turn == "1" else "1"


def score_board(board, computer):
    human = other_turn(computer)
    if has_win(board, computer):
        return 1
    if has_win(board, human):
        return -1
    if "0" not in board:
        return 0
    return None


def branch(board, turn, computer):
    score = score_board(board, computer)
    if score is not None:
        return score

    scores = []
    for _, new_board in possible_moves(board, turn):
        scores.append(branch(new_board, other_turn(turn), computer))

    if turn == computer:
        return max(scores)
    return min(scores)


def best_move(board, computer):
    best_index = None
    best_score = -2

    for index, new_board in possible_moves(board, computer):
        score = branch(new_board, other_turn(computer), computer)
        if score > best_score:
            best_score = score
            best_index = index

    return best_index


def move_code(board):
    count1 = board.count("1")
    count2 = board.count("2")
    if count1 > count2:
        turn = "2"
    elif count2 > count1:
        turn = "1"
    else:
        turn = "1"
    if has_win(board, "1") or has_win(board, "2"):
        print("Game is already over.")
        if has_win(board, "1"):
            print("Green has won.")
        if has_win(board, "2"):
            print("White has won.")
        return None
    index = best_move(board, turn)
    if index is None:
        return None

    piece = "g" if turn == "1" else "w"
    #print(piece + str(index + 1))
    return piece + str(index + 1)
