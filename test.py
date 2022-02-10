import numpy as np
import torch

import game
import mcts
import model

MCTS_SEARCHES = 20
MCTS_BATCH_SIZE = 4
clear = "\n" * 100
model_file = '/media/Usuario/Documentos/Documentos/AAMaestría/2. Segundo semestre/Aprendizaje Reforzado/DaBtrain/DaB/' \
             'best_001_00200.dat'


class Session:
    BOT_PLAYER = 1
    USER_PLAYER = 0

    def __init__(self, model_file, player_moves_first, player_id):
        self.model_file = model_file
        self.model = model.Net(input_shape=model.OBS_SHAPE, actions_n=game.EDGES)
        self.model.load_state_dict(torch.load(model_file, map_location=lambda storage, loc: storage))
        self.state = game.INITIAL_STATE
        self.value = None
        self.player_moves_first = player_moves_first
        self.player_id = player_id
        self.moves = []
        self.mcts_store = mcts.MCTS()

    def move_player(self, edge):
        self.moves.append(edge)
        self.state, won, next_player = game.move(self.state, edge, self.USER_PLAYER)
        return won, next_player

    def move_bot(self):
        self.mcts_store.search_batch(MCTS_SEARCHES, MCTS_BATCH_SIZE, self.state, self.BOT_PLAYER, self.model)
        probs, values = self.mcts_store.get_policy_value(self.state, tau=1)
        action = np.random.choice(game.EDGES, p=probs)
        self.value = values[action]
        self.moves.append(action)
        self.state, won, next_player = game.move(self.state, action, self.BOT_PLAYER)
        return won, next_player

    def is_valid_move(self, move_col):
        return move_col in game.possible_moves(self.state)

    def is_draw(self):
        return len(game.possible_moves(self.state)) == 0

    def render(self):
        game.render(self.state)


session = Session(model_file, True, '001')
won = False
next_player = 0

while not won:
    print(clear)
    session.render()
    if next_player == 0:
        edge = input("Tu turno")
        edge = int(edge)
        if session.is_valid_move(edge):
            won, next_player = session.move_player(edge)
        else:
            print("Movimiento inválido")
    else:
        won, next_player = session.move_bot()
