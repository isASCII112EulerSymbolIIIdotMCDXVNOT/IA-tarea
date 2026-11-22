"""Visualización: animación lado a lado (A* vs BFS) y gráfica comparativa.

La animación reproduce la bitácora (`Resultado.pasos`) paso a paso, de modo que
se ve en vivo qué nodos entran a la frontera, cuáles se expanden, en qué orden,
y finalmente el camino encontrado.
"""
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from .algoritmos import Celda, Cuadricula, Resultado

LIBRE, OBSTACULO, FRONTERA, EXPLORADO, CAMINO, INICIO, META = range(7)
COLORES = ["#ffffff", "#2b2b2b", "#ffd166", "#a8dadc", "#e63946", "#2a9d8f", "#9b5de5"]
CMAP = ListedColormap(COLORES)


class Reproductor:
    """Reconstruye el estado de la cuadrícula tras k pasos de una búsqueda."""

    def __init__(self, grid: Cuadricula, inicio: Celda, meta: Celda, resultado: Resultado):
        self.grid, self.inicio, self.meta, self.res = grid, inicio, meta, resultado
        self.reiniciar()

    def reiniciar(self) -> None:
        self.k = 0
        self.estado = np.array(self.grid, dtype=int)  # 0 libre, 1 obstáculo
        self.frontera: set[Celda] = {self.inicio}
        self.explorados: set[Celda] = set()
        self.ultimo = None
        self._marcar_extremos()

    def _marcar_extremos(self) -> None:
        self.estado[self.inicio] = INICIO
        self.estado[self.meta] = META

    @property
    def terminado(self) -> bool:
        return self.k >= len(self.res.pasos)

    def avanzar_hasta(self, k: int) -> None:
        if k < self.k:
            self.reiniciar()
        k = min(k, len(self.res.pasos))
        while self.k < k:
            p = self.res.pasos[self.k]
            self.frontera.discard(p.nodo)
            self.explorados.add(p.nodo)
            self.estado[p.nodo] = EXPLORADO
            for n in p.nuevos:
                if n not in self.explorados:
                    self.frontera.add(n)
                    self.estado[n] = FRONTERA
            self.ultimo = p
            self.k += 1
        self._marcar_extremos()

    @property
    def camino_visible(self) -> bool:
        """El camino final se dibuja como línea encima, solo cuando la búsqueda terminó."""
        return self.terminado and bool(self.res.camino)

    def texto(self) -> str:
        if self.k == 0 or self.ultimo is None:
            return "Listo para empezar"
        p = self.ultimo
        if self.res.pasos and self.terminado:
            if self.res.camino:
                return (f"META alcanzada | camino: {self.res.costo:.0f} pasos\n"
                        f"Nodos expandidos: {self.k}")
            return f"Sin solución | nodos expandidos: {self.k}"
        detalle = f"g={p.g:.0f}" if p.h is None else f"g={p.g:.0f}  h={p.h:.0f}  f={p.f:.0f}"
        return (f"Expandidos: {self.k}  |  Frontera: {len(self.frontera)}\n"
                f"Expandiendo {p.nodo}: {detalle}")


def _preparar_eje(ax, grid, titulo):
    filas, cols = len(grid), len(grid[0])
    imagen = ax.imshow(np.array(grid), cmap=CMAP, vmin=0, vmax=6, interpolation="nearest")
    ax.set_xticks(np.arange(-0.5, cols, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, filas, 1), minor=True)
    ax.grid(which="minor", color="#d0d0d0", linewidth=0.5)
    ax.tick_params(which="both", bottom=False, left=False, labelbottom=False, labelleft=False)
    ax.set_title(titulo, fontsize=13, fontweight="bold")
    (actual,) = ax.plot([], [], marker="o", markersize=9, markerfacecolor="none",
                        markeredgecolor="#f77f00", markeredgewidth=2.5, linestyle="none")
    (linea,) = ax.plot([], [], color=COLORES[CAMINO], linewidth=3.5, marker="o", markersize=4,
                       solid_capstyle="round")
    return imagen, actual, linea


def animar(
    grid: Cuadricula,
    inicio: Celda,
    meta: Celda,
    res_a: Resultado,
    res_b: Resultado,
    paso: int = 3,
    fps: int = 12,
    guardar: str | None = None,
    mostrar: bool = True,
    espera_final: int = 25,
    png_final: str | None = None,
) -> None:
    """Anima dos búsquedas lado a lado sobre el mismo mapa.

    paso: cuántos nodos expandidos avanza cada cuadro (más alto = animación más corta).
    """
    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(11, 6.6))
    fig.suptitle("Misma instancia, dos búsquedas: informada (A*) vs no informada (BFS)", fontsize=12)
    img_a, act_a, lin_a = _preparar_eje(ax_a, grid, res_a.nombre)
    img_b, act_b, lin_b = _preparar_eje(ax_b, grid, res_b.nombre)
    rep_a = Reproductor(grid, inicio, meta, res_a)
    rep_b = Reproductor(grid, inicio, meta, res_b)

    leyenda = [
        Patch(facecolor=COLORES[INICIO], label="Inicio"),
        Patch(facecolor=COLORES[META], label="Meta"),
        Patch(facecolor=COLORES[OBSTACULO], label="Obstáculo"),
        Patch(facecolor=COLORES[FRONTERA], label="Frontera (abiertos)"),
        Patch(facecolor=COLORES[EXPLORADO], label="Explorado (cerrados)"),
        Line2D([0], [0], color=COLORES[CAMINO], linewidth=3.5, label="Camino final"),
    ]
    fig.legend(handles=leyenda, loc="lower center", ncol=6, frameon=False, fontsize=9)
    fig.tight_layout(rect=(0, 0.1, 1, 0.95))

    def dibujar(k: int):
        for rep, img, act, lin, ax in ((rep_a, img_a, act_a, lin_a, ax_a),
                                       (rep_b, img_b, act_b, lin_b, ax_b)):
            rep.avanzar_hasta(k)
            img.set_data(rep.estado)
            if rep.ultimo is not None and not rep.terminado:
                f, c = rep.ultimo.nodo
                act.set_data([c], [f])
            else:
                act.set_data([], [])
            if rep.camino_visible:
                lin.set_data([c for _, c in rep.res.camino], [f for f, _ in rep.res.camino])
            else:
                lin.set_data([], [])
            ax.set_xlabel(rep.texto(), fontsize=10)
        return img_a, img_b

    total = max(len(res_a.pasos), len(res_b.pasos))
    marcos = list(range(0, total, paso)) + [total] * espera_final
    anim = FuncAnimation(fig, dibujar, frames=marcos, interval=1000 / fps, repeat=True)

    if png_final:
        dibujar(total)
        fig.savefig(png_final, dpi=110)
        dibujar(0)
    if guardar:
        anim.save(guardar, writer=PillowWriter(fps=fps), dpi=80)
    if mostrar:
        plt.show()
    plt.close(fig)


def grafica_comparacion(resultados: list[Resultado], ruta: str) -> None:
    """Barras con nodos expandidos por algoritmo (mismo mapa, mismo costo óptimo)."""
    nombres = [r.nombre.replace(" (", "\n(") for r in resultados]
    valores = [r.expandidos for r in resultados]
    colores = ["#2a9d8f" if r.nombre.startswith("A*") else "#8d99ae" for r in resultados]

    fig, ax = plt.subplots(figsize=(9, 5))
    barras = ax.bar(nombres, valores, color=colores)
    ax.bar_label(barras, padding=3, fontsize=11, fontweight="bold")
    ax.set_ylabel("Nodos expandidos")
    costo = resultados[0].costo
    ax.set_title(f"Nodos expandidos para llegar a la meta (camino óptimo de costo {costo:.0f} en todos)")
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_ylim(0, max(valores) * 1.15)
    fig.tight_layout()
    fig.savefig(ruta, dpi=120)
    plt.close(fig)
