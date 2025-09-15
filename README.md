# Evolução da GCC – Experimento Sintético (ER)

Este diretório contém o script `er_gcc.py` para reproduzir a evolução da componente gigante no modelo Erdős–Rényi.

## Como rodar (recomendado)
```bash
python er_gcc.py --mode er --n 100 --samples 400 --cpoints 120
```
Saídas em `out/`:
- `er_curvas_linear.png` e `er_curvas_logx.png`
- `snapshot_ER_n100_c{0.5,1.0,4.0}.png`
- `RESPOSTAS.md` com perguntas e respostas objetivas

## Métricas e plots
- S: fração média de nós na componente gigante (GCC)
- s_finite: tamanho médio das componentes finitas (exclui a GCC)
- Limiar: onde S começa a subir; s_finite tende a ter um pico próximo
- Log no eixo x destaca a região perto do limiar

Ajuste `--samples`, `--n` e número de pontos para curvas mais lisas se necessário.
