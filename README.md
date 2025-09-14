# Evolução da GCC – Experimento Sintético (ER) e Dados Reais

Este diretório contém um script único (`er_gcc.py`) que roda:
- ER sintético (G(n,p) controlado por grau médio c)
- Percolação em subgrafos de dados reais (WWW/Proteína) – opcional

## Recomendado (ER sintético)

```bash
python er_gcc.py --mode er --n 100 --samples 300 --cmin 0.1 --cmax 6.0 --cpoints 80
```
Saídas em `out/`:
- `er_curvas_linear.png` e `er_curvas_logx.png`
- `snapshot_ER_n100_c{0.5,1.0,4.0}.png`
- `RESPOSTAS.md` com perguntas e respostas objetivas

## Modos com dados reais 
- WWW (subgrafo na maior componente):
```bash
python er_gcc.py --mode www --from_gcc --n 300 --samples 200 --points 61
```
- Proteína (subgrafo na maior componente):
```bash
python er_gcc.py --mode protein --from_gcc --n 300 --samples 200 --points 61
```
- Configuration Model (aleatório com mesma sequência de graus do WWW/Protein):
```bash
python er_gcc.py --mode config --base www --from_gcc --n 300 --samples 200 --points 61
```

## Métricas e plots
- S: fração média de nós na componente gigante (GCC)
- s_finite: tamanho médio das componentes finitas (exclui a GCC)
- Limiar: onde S começa a subir; s_finite tende a ter um pico próximo
- Log no eixo x destaca a região perto do limiar

Ajuste `--samples`, `--n` e número de pontos para curvas mais lisas se necessário.
