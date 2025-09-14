import math
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
import argparse
from pathlib import Path
from typing import Tuple, Dict

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR.parent / 'data'
OUT_DIR = BASE_DIR / 'out'
OUT_DIR.mkdir(parents=True, exist_ok=True)

def load_graph_undirected(path: Path, directed: bool = False) -> nx.Graph:
    """
    Lê edgelist (duas colunas src dst). Se directed=True, cria DiGraph e converte; caso contrário cria Graph.
    Retorna SEMPRE um grafo não direcionado para medições de conectividade.
    """
    df = pd.read_csv(path, sep='\s+', header=None, usecols=[0,1], names=['src','dst'], engine='python', comment='#')
    if directed:
        Gd = nx.from_pandas_edgelist(df, source='src', target='dst', create_using=nx.DiGraph())
        return Gd.to_undirected()
    return nx.from_pandas_edgelist(df, source='src', target='dst', create_using=nx.Graph())

def sample_induced_subgraph(G: nx.Graph, n: int, seed: int = 42) -> nx.Graph:
    """Retorna subgrafo induzido por uma amostra de n nós do grafo G."""
    rng = np.random.default_rng(seed)
    nodes = list(G.nodes())
    if len(nodes) <= n:
        return G.copy()
    idx = rng.choice(len(nodes), size=n, replace=False)
    chosen = [nodes[i] for i in idx]
    return G.subgraph(chosen).copy()

def largest_component_nodes(G: nx.Graph) -> list:
    """Retorna lista de nós da maior componente do grafo não direcionado equivalente."""
    H = G.to_undirected() if G.is_directed() else G
    comps = list(nx.connected_components(H))
    if not comps:
        return []
    return list(max(comps, key=len))

def edge_percolation(G: nx.Graph, q: float, seed: int | None = None) -> nx.Graph:
    """Mantém cada aresta com probabilidade q (percolação por arestas)."""
    rng = np.random.default_rng(seed)
    H = nx.Graph()
    H.add_nodes_from(G.nodes())
    for u, v in G.edges():
        if rng.random() < q:
            H.add_edge(u, v)
    return H

def giant_component_fraction(G: nx.Graph) -> Tuple[float, float]:
    """
    Retorna (S, s_finite):
    - S: fração de nós na componente gigante
    - s_finite: tamanho médio das componentes FINITAS (exclui a gigante), em fração de nós
    """
    comps = [len(c) for c in nx.connected_components(G)]
    if not comps:
        return 0.0, 0.0
    comps.sort(reverse=True)
    giant = comps[0]
    finite = comps[1:] if len(comps) > 1 else []
    S = giant / G.number_of_nodes()
    if finite:
        s_mean = float(np.mean(finite)) / G.number_of_nodes()
    else:
        s_mean = 0.0
    return S, s_mean

def estimate_curves_percolation(G_base: nx.Graph, q_values: np.ndarray, samples: int, seed: int = 42) -> Dict[str, np.ndarray]:
    """Para cada q, estima E[S] e E[s_finite] sobre 'samples' percolações em G_base."""
    rng = np.random.default_rng(seed)
    S_means = []
    s_means = []
    for q in q_values:
        S_list, s_list = [], []
        for _ in range(samples):
            H = edge_percolation(G_base, q, seed=int(rng.integers(0, 1_000_000)))
            S, s_fin = giant_component_fraction(H)
            S_list.append(S)
            s_list.append(s_fin)
        S_means.append(float(np.mean(S_list)))
        s_means.append(float(np.mean(s_list)))
    return { 'q': q_values, 'S': np.array(S_means), 's_finite': np.array(s_means) }

def plot_curves_percolation(data: Dict[str, np.ndarray], logx: bool = False, logy: bool = False, title: str = 'Percolação (n=100)') -> Path:
    """Plota S(q) e s_finite(q) com opção de eixos log."""
    q = data['q']
    S = data['S']
    s = data['s_finite']

    fig, ax = plt.subplots(1, 1, figsize=(7, 4))
    ax.plot(q, S, color='tab:blue', lw=2, label='S (fração GCC)')
    ax.plot(q, s, color='tab:red', lw=2, label='s_finite (média comp. finitas)')
    ax.set_xlabel('probabilidade de manter aresta (q)')
    ax.set_ylabel('fração de nós')
    ax.set_title(title)
    if logx:
        ax.set_xscale('log')
    if logy:
        ax.set_yscale('log')
    ax.set_ylim(0, 1.05)
    ax.legend(loc='best')
    fig.tight_layout()
    out = OUT_DIR / ('percolacao_logx%s_logy%s.png' % (int(logx), int(logy)))
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out

def save_snapshots_graph(G_base: nx.Graph, q_list: list[float], label: str, seed: int = 123) -> None:
    """Salva snapshots da percolação no subgrafo base para valores de q."""
    for q in q_list:
        H = edge_percolation(G_base, q, seed=seed)
        try:
            pos = nx.spring_layout(H, seed=seed)
        except Exception:
            # fallback sem SciPy
            pos = nx.random_layout(H, seed=seed)
        plt.figure(figsize=(5, 4))
        nx.draw_networkx(H, pos=pos, node_size=15, width=0.4, with_labels=False)
        plt.title(f"Snapshot {label}: n={H.number_of_nodes()}, q={q}")
        out = OUT_DIR / f"snapshot_{label}_n{H.number_of_nodes()}_q{q}.png"
        plt.tight_layout()
        plt.savefig(out, dpi=150)
        plt.close()

def write_markdown_answers(data: Dict[str, np.ndarray], out_linear: Path, out_logx: Path, snapshots: list[Path], dataset_label: str) -> Path:
    """
    Gera RESPOSTAS.md com as figuras e respostas curtas sobre o limiar e o efeito de escala log.
    """
    # data has keys: 'x', 'S', 's_finite', 'x_label'
    x = data['x']; S = data['S']; s = data['s_finite']
    # Heurísticas simples para pontos de referência
    def _first_cross(xs, ys, thr):
        for xi, yi in zip(xs, ys):
            if yi >= thr:
                return float(xi)
        return float('nan')
    x_10 = _first_cross(x, S, 0.10)
    x_90 = _first_cross(x, S, 0.90)
    lines = []
    lines.append(f'## Explicação simples – {dataset_label} (n=100)\n')
    lines.append('\n')
    # Cabeçalho acessível
    human_label = 'grau médio c' if data['x_label'] == 'c' else 'probabilidade q'
    lines.append('### O que estamos medindo\n')
    lines.append(f'- {human_label}: variamos este controle e observamos como a rede se conecta.\n')
    lines.append('- S: parte dos nós que ficou na “ilha gigante” (GCC). Vai de 0 a 1.\n')
    lines.append('- s_finite: tamanho médio das outras ilhas (as pequenas componentes).\n')
    lines.append('- Cada ponto é a média de 100 redes/simulações no mesmo valor do controle.\n')
    lines.append('\n')
    rel_linear = './' + str(out_linear.relative_to(BASE_DIR))
    rel_logx = './' + str(out_logx.relative_to(BASE_DIR))
    lines.append(f'![Curvas]({rel_linear})\n')
    lines.append(f'![Curvas (log x)]({rel_logx})\n')
    for p in snapshots:
        lines.append(f'![Snapshot](./{p.relative_to(BASE_DIR)})')
    lines.append('\n')
    lines.append('### O que vemos nos gráficos\n')
    lines.append(f'- “Virada” (início da rede gigante): S passa de ~0 para cima de 0.1 por volta de {data["x_label"]}≈{x_10:.2f}.\n')
    lines.append(f'- Rede “quase toda conectada”: S chega perto de 0.9 a partir de {data["x_label"]}≈{x_90:.2f}.\n')
    lines.append('- s_finite costuma formar um pico perto da virada: as ilhas pequenas se juntam antes de a ilha gigante dominar.\n')
    lines.append('\n')
    lines.append('### Dica de leitura\n')
    lines.append('- Ativar escala log no eixo x amplia a região inicial e deixa a transição mais fácil de ver.\n')
    lines.append('- Não usamos log no eixo y porque S é uma fração entre 0 e 1.\n')
    lines.append('\n')
    lines.append('### Em uma frase\n')
    lines.append('- A rede quase não tem conexão global, depois “engata” rápido perto do limiar e, em seguida, praticamente todo mundo fica conectado.\n')
    lines.append('\n')
    # Perguntas e respostas explícitas
    lines.append('### Perguntas e respostas\n')
    if data['x_label'] == 'c':
        lines.append('- Pergunta: “Gere redes ER com n=100, variando ⟨k⟩; estime S.” Resposta: geramos 60 valores de c entre 0.1 e 6.0, com 100–200 amostras por ponto; S(c) e s_finite(c) estão nos gráficos acima.\n')
        lines.append(f'- Pergunta: “Onde está o ponto crítico e o fim do supercrítico?” Resposta: a virada ocorre por volta de c≈{x_10:.2f} (S>0.1). Consideramos “supercrítico consolidado” quando S≈0.9, a partir de c≈{x_90:.2f}.\n')
        lines.append('- Pergunta: “Log deixa a transição mais chamativa?” Resposta: sim, log no eixo x enfatiza a vizinhança do limiar.\n')
        lines.append('- Pergunta: “E o tamanho das componentes isoladas?” Resposta: s_finite(c) exibe um pico perto do limiar, indicando coalescência das componentes pequenas.\n')
    else:
        lines.append('- Pergunta: “Com dados reais (percolação por q), vemos transição?” Resposta: sim; S(q) cresce em forma sigmoidal e s_finite(q) forma um pico próximo do limiar.\n')
        lines.append(f'- Limiar aproximado no nosso subgrafo: {data["x_label"]}≈{x_10:.2f} (início) até ≈{x_90:.2f} (S≈0.9).\n')
        lines.append('- Log no eixo x também ajuda a destacar a transição com dados reais.\n')
    out_md = BASE_DIR / 'RESPOSTAS.md'
    out_md.write_text('\n'.join(lines), encoding='utf-8')
    return out_md
    
def main():
    parser = argparse.ArgumentParser(description='Experimentos: ER sintético ou percolação em dados reais (subgrafo n=100).')
    parser.add_argument('--mode', choices=['er','protein','www'], default='er')
    parser.add_argument('--samples', type=int, default=100)
    parser.add_argument('--n', type=int, default=100, help='tamanho do subgrafo/amostra')
    # ER params
    parser.add_argument('--cmin', type=float, default=0.1)
    parser.add_argument('--cmax', type=float, default=6.0)
    parser.add_argument('--cpoints', type=int, default=60)
    # Percolation params
    parser.add_argument('--qmin', type=float, default=0.0)
    parser.add_argument('--qmax', type=float, default=1.0)
    parser.add_argument('--points', type=int, default=41)
    parser.add_argument('--from_gcc', action='store_true', help='amostrar nós apenas da maior componente')
    args = parser.parse_args()

    if args.mode == 'er':
        # ER sintético por c
        n = int(args.n)
        c_values = np.linspace(args.cmin, args.cmax, args.cpoints)

        def generate_er_graph(n: int, c: float, seed: int | None = None) -> nx.Graph:
            p = min(max(c / max(n - 1, 1), 0.0), 1.0)
            return nx.erdos_renyi_graph(n=n, p=p, seed=seed)

        # estimar curvas ER
        rng = np.random.default_rng(42)
        S_means, s_means = [], []
        for c in c_values:
            S_list, s_list = [], []
            for _ in range(args.samples):
                G = generate_er_graph(n, c, seed=int(rng.integers(0, 1_000_000)))
                S, s_fin = giant_component_fraction(G)
                S_list.append(S)
                s_list.append(s_fin)
            S_means.append(float(np.mean(S_list)))
            s_means.append(float(np.mean(s_list)))
        data = {'x': c_values, 'S': np.array(S_means), 's_finite': np.array(s_means), 'x_label': 'c'}

        # plots
        def plot_er_curves(data):
            c = data['x']; S = data['S']; s = data['s_finite']
            fig, ax = plt.subplots(1,1, figsize=(7,4))
            ax.axvspan(0, 1.0, color='#9e9e9e', alpha=0.15, label='subcrítico')
            ax.axvline(1.0, color='tab:orange', linestyle='--', linewidth=1.5, label='crítico (c≈1)')
            ax.axvspan(1.0, max(c), color='#4caf50', alpha=0.08, label='supercrítico')
            ax.plot(c, S, lw=2, label='S (fração GCC)')
            ax.plot(c, s, lw=2, label='s_finite (média comp. finitas)')
            ax.set_xlabel('grau médio ⟨k⟩ = c')
            ax.set_ylabel('fração de nós')
            ax.legend(); fig.tight_layout()
            out = OUT_DIR / 'er_curvas_linear.png'
            fig.savefig(out, dpi=150); plt.close(fig)
            # log x
            fig, ax = plt.subplots(1,1, figsize=(7,4))
            ax.plot(c, S, lw=2, label='S (fração GCC)')
            ax.plot(c, s, lw=2, label='s_finite (média comp. finitas)')
            ax.set_xscale('log'); ax.set_xlabel('c (log)'); ax.set_ylabel('fração de nós')
            ax.legend(); fig.tight_layout()
            out_log = OUT_DIR / 'er_curvas_logx.png'
            fig.savefig(out_log, dpi=150); plt.close(fig)
            return out, out_log
        out1, out2 = plot_er_curves(data)

        # snapshots ER
        def save_er_snaps():
            snaps = []
            for c in [0.5, 1.0, 4.0]:
                G = nx.erdos_renyi_graph(n, p=min(max(c/(n-1),0),1), seed=123)
                pos = nx.spring_layout(G, seed=123)
                plt.figure(figsize=(5,4))
                nx.draw_networkx(G, pos=pos, node_size=15, width=0.4, with_labels=False)
                out = OUT_DIR / f'snapshot_ER_n{n}_c{c}.png'
                plt.tight_layout(); plt.savefig(out, dpi=150); plt.close()
                snaps.append(out)
            return snaps
        snaps = save_er_snaps()
        md = write_markdown_answers(data, out1, out2, snaps, dataset_label='ER (sintético)')
        print(f'Gerado: {out1}\nGerado: {out2}\nMarkdown: {md}')
        return

    # modos com dados reais
    if args.mode == 'protein':
        dataset_label = 'Protein'
        path = DATA_DIR / 'protein.edgelist.txt'
        G_full = load_graph_undirected(path, directed=False)
    else:
        dataset_label = 'WWW'
        path = DATA_DIR / 'www.edgelist.txt'
        G_full = load_graph_undirected(path, directed=True)

    # amostragem de subgrafo
    if args.from_gcc:
        gcc_nodes = largest_component_nodes(G_full)
        G = sample_induced_subgraph(G_full.subgraph(gcc_nodes), n=args.n, seed=42)
    else:
        G = sample_induced_subgraph(G_full, n=args.n, seed=42)
    q_values = np.linspace(args.qmin, args.qmax, args.points)
    data = estimate_curves_percolation(G_base=G, q_values=q_values, samples=args.samples)
    # adapt x-label
    data = {'x': data['q'], 'S': data['S'], 's_finite': data['s_finite'], 'x_label': 'q'}
    out1 = plot_curves_percolation({'q': data['x'], 'S': data['S'], 's_finite': data['s_finite']}, logx=False, logy=False, title=f'Percolação no subgrafo {dataset_label} (n=100)')
    out2 = plot_curves_percolation({'q': data['x'], 'S': data['S'], 's_finite': data['s_finite']}, logx=True, logy=False, title=f'Percolação {dataset_label} (log x)')
    qs = [0.2, 0.5, 0.9]
    save_snapshots_graph(G, qs, label=dataset_label)
    snap_paths = [OUT_DIR / f"snapshot_{dataset_label}_n{G.number_of_nodes()}_q{q}.png" for q in qs]
    out_md = write_markdown_answers(data, out1, out2, snap_paths, dataset_label=dataset_label)
    print(f"Gerado: {out1}\nGerado: {out2}\nSaída em: {OUT_DIR}\nMarkdown: {out_md}")

if __name__ == '__main__':
    main()


