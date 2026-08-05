def possible_moves(board, turn):
    return [(i, board[:i] + turn + board[i + 1 :]) for i, char in enumerate(board) if char == "0"]


def has_win(board, turn):
    lines = ((0, 1, 2), (3, 4, 5), (6, 7, 8), (0, 3, 6), (1, 4, 7), (2, 5, 8), (0, 4, 8), (2, 4, 6))
    return any(board[a] == board[b] == board[c] == turn for a, b, c in lines)


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
    turn = "2" if board.count("1") > board.count("2") else "1"
    for piece, color in (("1", "Green"), ("2", "White")):
        if has_win(board, piece):
            print(f"Game is already over. {color} has won.")
            return None
    index = best_move(board, turn)
    if index is None:
        return None
    piece = "g" if turn == "1" else "w"
    return piece + str(index + 1)
