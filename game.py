import numpy as np

GAME_SIZE = 3
SCORE_0 = 0
SCORE_1 = 0
EDGES = ((GAME_SIZE ** 2) + ((GAME_SIZE + 1) ** 2) - 1)


def encode_lists(field_lists):
    encoded = 0
    score_0 = field_lists[-2]
    score_1 = field_lists[-1]
    for bits in range((len(field_lists)) - 2):
        encoded += field_lists[(-bits - 3)] * pow(2, bits)  # Suma cada bit elevado a su respectiva potencia
    encoded *= 10000
    encoded = encoded + (score_0 * 100) + score_1
    return encoded


def decode_binary(state_int):
    if state_int == 0: state_int = '00000'
    res = []
    score_0 = int(str(state_int)[-4:-2])
    score_1 = int(str(state_int)[-2:])
    state = int(str(state_int)[:-4])
    binary_string = bin(state).replace("0b", "")
    binary_string = f"{binary_string :0>{EDGES}}"  # Rellena con ceros a la izquierda hasta el número de bordes
    for bit in binary_string:
        res.append(int(bit))
    res.append(score_0)
    res.append(score_1)
    return res


INITIAL_STATE = encode_lists([0] * (EDGES + 2))


def possible_moves(state_int):
    assert isinstance(state_int, int)
    field = decode_binary(state_int)
    moves = []
    for i in range(0, len(field[0:-2])):
        if field[i] == 0:
            moves.append(i)
    return moves


def node_valency(field):
    """
    Método que, dado un estado, devuelve una lista con la valencia de cada nodo.
    Esta lista es de tamaño GAME_SIZE^2. Contiene valores entre 0 (no se ha jugado ningún borde rodeando al nodo)
    a 4 (ya se jugaron todos los bordes)
    Los bordes y los nodos se cuentan de arriba a abajo y de derecha a izquierda.
    La fórmula para saber el índice del borde de arriba de cada nodo es súper interesante y vale la pena detallarla,
    el índice de los demás bordes se puede calcular fácilmente sabiendo el de arriba.
    """
    up, node = [], []
    for i in range(0, GAME_SIZE ** 2):
        div, mod = divmod(i, GAME_SIZE)
        up.append((((2 * GAME_SIZE) + 1) * div) + mod)
    down = [((2 * GAME_SIZE) + 1) + up_i for up_i in up]
    left = [up_i + GAME_SIZE for up_i in up]
    right = [lf_i + 1 for lf_i in left]
    for i in range(0, GAME_SIZE ** 2):
        node.append(field[up[i]] + field[down[i]] + field[left[i]] + field[right[i]])
    return node


def move(state_int, edge, player):
    assert isinstance(edge, int)
    assert isinstance(state_int, int)
    field = decode_binary(state_int)
    assert field[edge] == 0
    """
    A continuación, se guarda la valencia de cada nodo antes de la jugada. Luego se marca el borde seleccionado.
    Se guarda la valencia después de la jugada. Se compara cuántas valencias hay iguales a 4 antes y después.
    Se asignan los puntos correspondientes. En teoría, un movimiento puede otorgar como máximo dos puntos.
    """
    pre_node = node_valency(field)
    field[edge] = 1
    pos_node = node_valency(field)
    points = pos_node.count(4) - pre_node.count(4)
    player_to_field = [-2, -1]
    field[player_to_field[player]] += points
    state_new = encode_lists(field)
    won = field[player_to_field[player]] >= np.ceil(GAME_SIZE ** 2 / 2)
    new_player = 1 - player if points == 0 else player
    return state_new, won, new_player


def render(state_int):
    rlist = decode_binary(state_int)
    for idx in range(2 * GAME_SIZE + 1):
        line = ""
        range_inf = idx * GAME_SIZE + int(np.floor(idx / 2))
        if idx % 2 == 0:
            for jdx in range(range_inf,range_inf + GAME_SIZE ):
                line += "   |" if rlist[jdx] == 0 else "    "
        else:
            for jdx in range(range_inf,range_inf + GAME_SIZE):
                line += "---o" if rlist[jdx] == 0 else "   o"
            line += "---" if rlist[jdx + 1] == 0 else "   "
        print(line)


