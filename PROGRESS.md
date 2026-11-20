# PROGRESSO — lora-finetune-nicho

Último ponto onde parei (atualize ao fim de cada sessão de 1h):

- **Data:** 2026-11-20 (bloco de SÁBADO 14/11, atrasado, feito pelo Hermes a pedido do Lucas)
- **Onde parei:** decodificação restrita ao esquema pronta e medida (`src/restrito.py`, `--restrito`): árvore com as 144 respostas válidas, aplicada via `prefix_allowed_tokens_fn`. Resultado: rótulo no esquema do LoRA 95,1% → 100% no teste novo, **acerto exato idêntico** (54,6% na média das sementes; os rótulos inventados viravam outro rótulo errado). 35 testes (inclui um modelo aleatório que só acerta o formato com a restrição, e lote com padding à esquerda). Mutação: 13 mutantes, 3 sobreviveram de início: 2 eram a fiação do `--restrito` no avaliador (sem teste: virou `tests/test_eval_restrito.py`, agora pegos) e 1 é **equivalente** (construir a árvore sem usá-la quando não é restrito: muda só o tempo, não o resultado). Um erro no meu primeiro teste do avaliador (chdir antes de carregar dados) fazia os mutantes parecerem pegos por engano; refeito e rodado de novo.
- **O que NÃO foi feito:** treino de novo (nenhum adaptador mudou), diversidade de dados, ablação, modelo 1,5B, latência com/sem restrição.
- **Próximo passo:** (1) mais diversidade nos dados de treino e remedir o teste novo (é o que move o acerto, 55%); (2) ablação de r, épocas e taxa com 3 sementes (cada treino de 3 sementes leva tempo de GPU: avisar antes); (3) 1,5B (cabe em 6 GB? testar antes); (4) demo.
