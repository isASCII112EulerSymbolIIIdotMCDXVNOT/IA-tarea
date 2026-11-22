"""Algoritmos de búsqueda sobre cuadrículas: A* y BFS (búsqueda no informada).

Cada algoritmo devuelve un `Resultado` que, además del camino, guarda la
bitácora completa de la búsqueda: qué nodo se expandió en cada paso (con sus
valores g, h y f) y qué nodos nuevos entraron a la frontera. Esa bitácora es lo
que permite animar el *proceso interno* en el video, no solo el resultado final.

Modelo del problema
-------------------
* Estados: celdas (fila, columna) de una cuadrícula.
* Acciones: moverse arriba/abajo/izquierda/derecha a una celda libre.
* Costo: 1 por movimiento.
"""
from __future__ import annotations

import heapq
import math
from collections import deque
from dataclasses import dataclass, field
from typing import Callable

Celda = tuple[int, int]        # (fila, columna)
Cuadricula = list[list[int]]   # 0 = libre, 1 = obstáculo

# Movimientos en 4 direcciones, cada uno con costo 1.
MOVIMIENTOS: list[Celda] = [(-1, 0), (1, 0), (0, -1), (0, 1)]


# --------------------------------------------------------------------------
# Heurísticas h(n, meta)
# --------------------------------------------------------------------------
def manhattan(a: Celda, b: Celda) -> float:
    """|Δfila| + |Δcolumna|. Admisible y consistente con movimientos en 4 direcciones."""
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def euclidiana(a: Celda, b: Celda) -> float:
    """Distancia en línea recta. Admisible, pero menos informada que Manhattan."""
    return math.hypot(a[0] - b[0], a[1] - b[1])


def cero(a: Celda, b: Celda) -> float:
    """h(n) = 0. Con esta 'heurística' A* se comporta como Dijkstra (sin información)."""
    return 0.0


# --------------------------------------------------------------------------
# Resultado de una búsqueda
# --------------------------------------------------------------------------
@dataclass
class Paso:
    """Un nodo expandido durante la búsqueda."""
    nodo: Celda
    g: float                      # costo real desde el inicio
    h: float | None               # estimación a la meta (None en búsqueda no informada)
    f: float | None               # g + h (None en búsqueda no informada)
    nuevos: list[Celda] = field(default_factory=list)  # nodos que entraron a la frontera


@dataclass
class Resultado:
    nombre: str
    camino: list[Celda]           # vacío si no hay solución
    costo: float | None
    pasos: list[Paso]

    @property
    def expandidos(self) -> int:
        return len(self.pasos)

    @property
    def generados(self) -> int:
        return 1 + sum(len(p.nuevos) for p in self.pasos)


# --------------------------------------------------------------------------
# Utilidades
# --------------------------------------------------------------------------
def vecinos(grid: Cuadricula, nodo: Celda) -> list[Celda]:
    filas, cols = len(grid), len(grid[0])
    resultado = []
    for df, dc in MOVIMIENTOS:
        f, c = nodo[0] + df, nodo[1] + dc
        if 0 <= f < filas and 0 <= c < cols and grid[f][c] == 0:
            resultado.append((f, c))
    return resultado


def _reconstruir(padre: dict[Celda, Celda | None], meta: Celda) -> list[Celda]:
    camino = []
    nodo: Celda | None = meta
    while nodo is not None:
        camino.append(nodo)
        nodo = padre[nodo]
    camino.reverse()
    return camino


# --------------------------------------------------------------------------
# A*
# --------------------------------------------------------------------------
def astar(
    grid: Cuadricula,
    inicio: Celda,
    meta: Celda,
    h: Callable[[Celda, Celda], float] = manhattan,
    nombre: str | None = None,
) -> Resultado:
    """A* con cola de prioridad ordenada por f = g + h.

    Desempate: ante igual f se prefiere el nodo con menor h (más cerca de la
    meta); después, el que entró primero. Esto reduce expansiones inútiles.
    """
    nombre = nombre or f"A* ({h.__name__})"
    g: dict[Celda, float] = {inicio: 0}
    padre: dict[Celda, Celda | None] = {inicio: None}
    h0 = h(inicio, meta)
    contador = 0
    abiertos: list[tuple[float, float, int, Celda]] = [(h0, h0, contador, inicio)]
    cerrados: set[Celda] = set()
    pasos: list[Paso] = []

    while abiertos:
        f_actual, h_actual, _, actual = heapq.heappop(abiertos)
        if actual in cerrados:          # entrada obsoleta (había una mejor)
            continue
        cerrados.add(actual)

        if actual == meta:
            pasos.append(Paso(actual, g[actual], h_actual, f_actual, []))
            return Resultado(nombre, _reconstruir(padre, meta), g[actual], pasos)

        nuevos: list[Celda] = []
        for vecino in vecinos(grid, actual):
            g_nuevo = g[actual] + 1
            if vecino not in g or g_nuevo < g[vecino]:
                g[vecino] = g_nuevo
                padre[vecino] = actual
                h_v = h(vecino, meta)
                contador += 1
                heapq.heappush(abiertos, (g_nuevo + h_v, h_v, contador, vecino))
                nuevos.append(vecino)
        pasos.append(Paso(actual, g[actual], h_actual, f_actual, nuevos))

    return Resultado(nombre, [], None, pasos)


# --------------------------------------------------------------------------
# Búsqueda no informada (BFS)
# --------------------------------------------------------------------------
def bfs(grid: Cuadricula, inicio: Celda, meta: Celda, nombre: str = "BFS (sin heurística)") -> Resultado:
    """Búsqueda en anchura. Con costo uniforme encuentra el camino más corto.

    Igual que en A*, se prueba si un nodo es la meta *al expandirlo* (no al
    generarlo), para que el conteo de nodos expandidos sea comparable.
    """
    padre: dict[Celda, Celda | None] = {inicio: None}
    dist: dict[Celda, int] = {inicio: 0}
    cola: deque[Celda] = deque([inicio])
    pasos: list[Paso] = []

    while cola:
        actual = cola.popleft()
        if actual == meta:
            pasos.append(Paso(actual, dist[actual], None, None, []))
            return Resultado(nombre, _reconstruir(padre, meta), dist[actual], pasos)

        nuevos: list[Celda] = []
        for vecino in vecinos(grid, actual):
            if vecino not in padre:
                padre[vecino] = actual
                dist[vecino] = dist[actual] + 1
                cola.append(vecino)
                nuevos.append(vecino)
        pasos.append(Paso(actual, dist[actual], None, None, nuevos))

    return Resultado(nombre, [], None, pasos)
