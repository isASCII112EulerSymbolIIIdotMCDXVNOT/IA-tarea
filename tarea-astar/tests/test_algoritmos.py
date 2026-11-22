import random
import unittest
from collections import deque
from astar_demo.algoritmos import astar, bfs, cero, euclidiana, manhattan, vecinos
from astar_demo.mapas import cargar_mapa, obtener_mapa


def distancias_reales_a_meta(grid, meta):
    dist = {meta: 0}
    cola = deque([meta])
    while cola:
        n = cola.popleft()
        for v in vecinos(grid, n):
            if v not in dist:
                dist[v] = dist[n] + 1
                cola.append(v)
    return dist


def mapa_aleatorio(rng, filas=12, cols=12, densidad=0.3):
    grid = [[1 if rng.random() < densidad else 0 for _ in range(cols)] for _ in range(filas)]
    inicio, meta = (0, 0), (filas - 1, cols - 1)
    grid[0][0] = grid[filas - 1][cols - 1] = 0
    return grid, inicio, meta


class PruebasAlgoritmos(unittest.TestCase):
    def test_astar_es_optimo_igual_que_bfs(self):
        rng = random.Random(1)
        for _ in range(300):
            grid, s, g = mapa_aleatorio(rng)
            ref = bfs(grid, s, g)
            for h in (manhattan, euclidiana, cero):
                res = astar(grid, s, g, h)
                self.assertEqual(res.costo, ref.costo, f"heurística {h.__name__}")

    def test_manhattan_y_euclidiana_son_admisibles(self):
        rng = random.Random(2)
        for _ in range(100):
            grid, s, g = mapa_aleatorio(rng)
            reales = distancias_reales_a_meta(grid, g)
            for nodo, costo_real in reales.items():
                self.assertLessEqual(manhattan(nodo, g), costo_real)
                self.assertLessEqual(euclidiana(nodo, g), costo_real)

    def test_camino_es_valido(self):
        grid, s, g = obtener_mapa("trampa")
        res = astar(grid, s, g)
        self.assertEqual(res.camino[0], s)
        self.assertEqual(res.camino[-1], g)
        self.assertEqual(len(res.camino) - 1, res.costo)
        for a, b in zip(res.camino, res.camino[1:]):
            self.assertEqual(manhattan(a, b), 1)      
            self.assertEqual(grid[b[0]][b[1]], 0)     

    def test_sin_solucion(self):
        grid, s, g = cargar_mapa("S#G\n.#.\n.#.\n")
        self.assertEqual(astar(grid, s, g).camino, [])
        self.assertIsNone(astar(grid, s, g).costo)
        self.assertEqual(bfs(grid, s, g).camino, [])

    def test_inicio_igual_a_meta(self):
        grid = [[0, 0], [0, 0]]
        res = astar(grid, (0, 0), (0, 0))
        self.assertEqual(res.costo, 0)
        self.assertEqual(res.camino, [(0, 0)])

    def test_astar_expande_menos_que_bfs_en_mapas_de_ejemplo(self):
        for nombre in ("demo", "trampa", "aleatorio"):
            grid, s, g = obtener_mapa(nombre)
            self.assertLess(astar(grid, s, g).expandidos, bfs(grid, s, g).expandidos, nombre)

    def test_astar_con_h_cero_equivale_a_bfs_en_expansiones(self):
        grid, s, g = obtener_mapa("demo")
        self.assertEqual(astar(grid, s, g, cero).expandidos, bfs(grid, s, g).expandidos)


if __name__ == "__main__":
    unittest.main()
