import numpy as np
from copy import deepcopy

GAME_ROWS = 4
GAME_COLS = 4
BITS_IN_LEN = 3

def bits_to_int(bits):
    res = 0
    for b in bits:
        res *= 2
        res += b
    return res

def int_to_bits(num, bits):
    res = []
    for _ in range(bits):
        res.append(num % 2)
        num //= 2
    return res[::-1]

def encode_lists(field_lists):
    assert isinstance(field_lists, list)
    assert len(field_lists) == GAME_COLS

    bits = []
    len_bits = []
    for col in field_lists:
        bits.extend(col)
        free_len = GAME_ROWS-len(col)
        bits.extend([0] * free_len)
        len_bits.extend(int_to_bits(free_len, bits=BITS_IN_LEN))
    bits.extend(len_bits)
    return bits_to_int(bits)

def decode_binary(state_int):
    assert isinstance(state_int, int)
    bits = int_to_bits(state_int, bits=GAME_COLS*GAME_ROWS + GAME_COLS*BITS_IN_LEN)
    res = []
    len_bits = bits[GAME_COLS*GAME_ROWS:]
    for col in range(GAME_COLS):
        vals = bits[col*GAME_ROWS:(col+1)*GAME_ROWS]
        lens = bits_to_int(len_bits[col*BITS_IN_LEN:(col+1)*BITS_IN_LEN])
        if lens > 0:
            vals = vals[:-lens]
        res.append(vals)
    return res

class Borde:
    def __init__(self, n1, n2, bordeID):
        self.cortado = False
        self.nodo1 = n1
        self.nodo2 = n2
        self.bordeID = bordeID

    def cortar(self, n):
        self.cortado = True
        if self.nodo1 != -1:
            n[self.nodo1] -= 1
        if self.nodo2 != -1:
            n[self.nodo2] -= 1

    def printBorde(self):
        print(self.cortado, self.nodo1, self.nodo2)


class DabBoard:

    def __init__(self, rows, columns, recompensa_terminal=False):

        self.rows = rows
        self.columns = columns

        self.score1 = 0
        self.score2 = 0

        self.turno = 1

        # Guarda la información de la valencia de todos los nodos. La valencia se inicializa en 4.
        # El índice de cada nodo es su posición de izquierda a derecha y de arriba a abajo
        self.nodos = np.array([4] * (self.rows * self.columns))

        self.num_bordes = self.rows * self.columns + (self.rows + 1) * (self.columns + 1) - 1

        # Define el arreglo de bordes
        self.bordes = np.array([Borde(-1, -1, -1)] * self.num_bordes)

        num_col = self.columns + self.columns + 1
        num_row = self.rows + self.rows + 1
        id_borde = 0
        ix = 0  # Conteo de bordes horizontales
        iy = 0  # Conteo de bordes verticales

        # Inicializar el arreglo de bordes. Cuando un borde está al límite del tablero,
        # se considera conectado a un nodo -1
        for i in range(0, num_row):
            if i % 2 == 1:
                for j in range(0, self.columns + 1):

                    if j == 0:
                        nodo_izq = -1
                    else:
                        nodo_izq = id_borde - (self.rows + 1) - (2 * self.rows + 1) * ix + self.rows * ix

                    if j == columns:
                        nodo_der = -1
                    else:
                        nodo_der = id_borde - self.rows - (2 * self.rows + 1) * ix + self.rows * ix

                    self.bordes[id_borde] = Borde(nodo_izq, nodo_der, id_borde)
                    id_borde += 1
                ix += 1
            else:
                for j in range(0, self.columns):
                    if i == 0:
                        nodo_sup = -1
                    else:
                        nodo_sup = id_borde - self.rows - (self.rows + 1) * iy
                    if i == num_row - 1:
                        nodo_inf = -1
                    else:
                        nodo_inf = id_borde - (self.rows + 1) * iy

                    self.bordes[id_borde] = Borde(nodo_sup, nodo_inf, id_borde)
                    id_borde += 1
                iy += 1

    def Hash(self):
        ide = []
        for i in self.bordes:
            if i.cortado:
                ide.append(1)
            else:
                ide.append(0)
        return np.asarray(ide)

    def copy(self, db):

        self.bordes = deepcopy(db.bordes)

        self.nodos = deepcopy(db.nodos)
        self.num_bordes = db.num_bordes
        self.rows = db.rows
        self.columns = db.columns
        self.score1 = db.score1
        self.score2 = db.score2
        self.turno = db.turno

    def jugada(self, borde, verbose=False):
        e = self.bordes[borde]
        state_id = self.Hash()
        if not e.cortado:
            e.cortar(self.nodos)
            termina, j_puntua, recompensa = self.puntuacion(e, verbose)
            tupla = (state_id, recompensa, self.Hash(), termina, j_puntua)
            return tupla
        else:
            print("¡Ese borde ya fue cortado!")

    def puntuacion(self, e, verbose):

        # Entero indicando qué jugador completó un cuadrito este turno. 0 si ninguno

        j_puntua = 0
        recompensa = 0

        v1 = self.nodos[e.nodo1]
        v2 = self.nodos[e.nodo2]

        if e.nodo1 == -1:
            v1 = 4
        if e.nodo2 == -1:
            v2 = 4

        if v1 == 0:
            if self.turno == 1:
                self.score1 += 1
                j_puntua = 1
            else:
                self.score2 += 1
                j_puntua = 2
            recompensa += 1
        if v2 == 0:
            if self.turno == 1:
                self.score1 += 1
                j_puntua = 1
            else:
                self.score2 += 1
                j_puntua = 2
            recompensa += 1

        # Si ningún jugador completó una caja, cambia el turno.
        if j_puntua == 0:
            self.turno = 3 - self.turno

        if verbose:
            print("Puntuación Jugador 1:")
            print(self.score1)
            print("Puntuación Jugador 2:")
            print(self.score2)

            # if recompensa_terminal:
            # if(self.score1 + self.score2) == len(self.nodos):
            # if self.score1 > self.score2:        #Pendiente por implementar...
            # ¿Cómo dar la recompensa cuándo no es mi turno?
        # else: Indentar la siguiente línea
        if (self.score1 + self.score2) == len(self.nodos):
            return True, j_puntua, recompensa  # El juego termina
        else:
            return False, j_puntua, recompensa  # Continúa

    # Devuelve un array con todos los bordes no cortados en torno a un nodo n
    def mostrarBordes(self, n):
        i = int(n / self.rows)
        e = []

        izq = n + self.rows + i * (2 * self.rows + 1) - self.rows * i
        if not self.bordes[izq].cortado:
            e.append(izq)

        der = n + (self.rows + 1) + i * (2 * self.rows + 1) - self.rows * i
        if not self.bordes[der].cortado:
            e.append(der)

        sup = n + i * (self.rows + 1)
        if not self.bordes[sup].cortado:
            e.append(sup)

        inf = n + (self.rows + 1) * (i + 2) - 1
        if not self.bordes[inf].cortado:
            e.append(inf)

        return e

    def InfoBorde(self):
        for i in range(0, self.num_bordes):
            print(i, self.bordes[i].cortado, self.bordes[i].nodo1, self.bordes[i].nodo2)

    # Todos los bordes que pueden ser cortados
    def action_space(self):
        bordes_disponibles = []
        for borde in self.bordes:
            if not borde.cortado:
                bordes_disponibles.append(borde.bordeID)

        return bordes_disponibles

    def action_space_sample(self):
        muestra = np.random.choice(self.action_space())
        return muestra

    def render(self):
        e = 0
        for i in range(2 * self.rows + 1):
            linea = ""
            if i % 2 == 0:
                for j in range(self.columns):
                    if not self.bordes[e].cortado:
                        linea += "   |"
                    else:
                        linea += "    "
                    e += 1
            else:
                for j in range(self.columns):
                    if not self.bordes[e].cortado:
                        linea += "---o"
                    else:
                        linea += "   o"
                    e += 1
                if not self.bordes[e].cortado:
                    linea += "---"
                else:
                    linea += "   "
                e += 1

            print(linea)

    def reset(self):
        self.score1 = 0
        self.score2 = 0
        self.turno = 1
        self.nodos = np.array([4] * (self.rows * self.columns))
        for borde in self.bordes:
            borde.cortado = False
        estado = self.Hash()
        return estado
