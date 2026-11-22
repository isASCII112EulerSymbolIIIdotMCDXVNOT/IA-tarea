from __future__ import annotations
import argparse
import os
from astar_demo.algoritmos import astar, bfs, cero, euclidiana, manhattan
from astar_demo.mapas import MAPAS, obtener_mapa


def imprimir_paso_a_paso(res, n: int) -> None:
    print(f"\nPrimeras {n} expansiones de {res.nombre}")
    print("(A* siempre expande el nodo de la frontera con menor f = g + h)\n")
    print(f"{'#':>3}  {'nodo':<10} {'g':>3} {'h':>4} {'f':>4}  {'nuevos en frontera':>18}")
    print("-" * 52)
    for i, p in enumerate(res.pasos[:n], start=1):
        print(f"{i:>3}  {str(p.nodo):<10} {p.g:>3.0f} {p.h:>4.0f} {p.f:>4.0f}  {len(p.nuevos):>18}")


def imprimir_tabla(resultados) -> None:
    print("\nComparación en la misma instancia")
    print(f"{'algoritmo':<28} {'costo':>6} {'expandidos':>11} {'generados':>10}")
    print("-" * 59)
    for r in resultados:
        print(f"{r.nombre:<28} {r.costo:>6.0f} {r.expandidos:>11} {r.generados:>10}")
    base = resultados[-1].expandidos
    mejor = resultados[0].expandidos
    print(f"\nA* con Manhattan expandió {mejor} nodos vs {base} de BFS "
          f"({100 * (1 - mejor / base):.0f}% menos), con el mismo costo óptimo.")


def main() -> None:
    ap = argparse.ArgumentParser(description="A* vs BFS sobre una cuadrícula")
    ap.add_argument("--mapa", choices=[*MAPAS, "aleatorio"], default="demo")
    ap.add_argument("--semilla", type=int, default=7, help="semilla del mapa aleatorio")
    ap.add_argument("--paso", type=int, default=3, help="nodos expandidos por cuadro de animación")
    ap.add_argument("--fps", type=int, default=12)
    ap.add_argument("--guardar", metavar="ARCHIVO.gif", help="guardar la animación como GIF")
    ap.add_argument("--png", metavar="ARCHIVO.png", help="guardar el cuadro final como imagen")
    ap.add_argument("--tabla", action="store_true", help="tabla y gráfica comparativa")
    ap.add_argument("--paso-a-paso", type=int, metavar="N", help="imprimir las primeras N expansiones de A*")
    ap.add_argument("--sin-ventana", action="store_true", help="no abrir ventana (útil con --guardar)")
    ap.add_argument("--sin-animacion", action="store_true", help="solo texto, sin animación")
    args = ap.parse_args()

    grid, inicio, meta = obtener_mapa(args.mapa, args.semilla)
    print(f"Mapa: {args.mapa} ({len(grid)}x{len(grid[0])})  inicio={inicio}  meta={meta}")

    res_astar = astar(grid, inicio, meta, manhattan, "A* (h = Manhattan)")
    res_bfs = bfs(grid, inicio, meta, "BFS (sin heurística)")

    if not res_astar.camino:
        print("No existe camino entre inicio y meta en este mapa.")
        return

    if args.paso_a_paso:
        imprimir_paso_a_paso(res_astar, args.paso_a_paso)

    comparativa = [
        res_astar,
        astar(grid, inicio, meta, euclidiana, "A* (h = Euclidiana)"),
        astar(grid, inicio, meta, cero, "A* (h = 0, Dijkstra)"),
        res_bfs,
    ]
    imprimir_tabla(comparativa)

    if args.tabla:
        from astar_demo.visualizar import grafica_comparacion
        os.makedirs("salidas", exist_ok=True)
        ruta = f"salidas/comparacion_{args.mapa}.png"
        grafica_comparacion(comparativa, ruta)
        print(f"Gráfica guardada en {ruta}")

    if not args.sin_animacion:
        from astar_demo.visualizar import animar
        if args.guardar:
            carpeta = os.path.dirname(args.guardar)
            if carpeta:
                os.makedirs(carpeta, exist_ok=True)
        animar(grid, inicio, meta, res_astar, res_bfs, paso=args.paso, fps=args.fps,
               guardar=args.guardar, mostrar=not args.sin_ventana, png_final=args.png)
        if args.guardar:
            print(f"Animación guardada en {args.guardar}")


if __name__ == "__main__":
    main()
