from __future__ import annotations
import heapq
import math
from collections import deque
from dataclasses import dataclass, field
from typing import Callable
Celda = tuple[int, int]  
Cuadricula = list[list[int]]  

MOVIMIENTOS: list[Celda] = [(-1, 0), (1, 0), (0, -1), (0, 1)]


def manhattan(a: Celda, b: Celda) -> float:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def euclidiana(a: Celda, b: Celda) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def cero(a: Celda, b: Celda) -> float:
    return 0.0

@dataclass
class Paso:
    nodo: Celda
    g: float                   
    h: float | None           
    f: float | None            
    nuevos: list[Celda] = field(default_factory=list) 

@dataclass
class Resultado:
    nombre: str
    camino: list[Celda]    
    costo: float | None
    pasos: list[Paso]

    @property
    def expandidos(self) -> int:
        return len(self.pasos)

    @property
    def generados(self) -> int:
        return 1 + sum(len(p.nuevos) for p in self.pasos)

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


def astar(
    grid: Cuadricula,
    inicio: Celda,
    meta: Celda,
    h: Callable[[Celda, Celda], float] = manhattan,
    nombre: str | None = None,
) -> Resultado:

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
        if actual in cerrados:        
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


def bfs(grid: Cuadricula, inicio: Celda, meta: Celda, nombre: str = "BFS (sin heurística)") -> Resultado:

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
