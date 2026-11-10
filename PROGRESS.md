# PROGRESSO — lora-finetune-nicho

Último ponto onde parei (atualize ao fim de cada sessão de 1h):

- **Data:** 2026-11-10 (bloco de SÁBADO 07/11, atrasado, feito pelo Hermes a pedido do Lucas)
- **Onde parei:** pipeline completo e medido, 100% local e sem custo. Chamados sintéticos de TI (900 treino, 300 teste igual, 300 teste novo com vocabulário nunca visto). Qwen2.5-0.5B + LoRA (3 sementes) contra classe majoritária, regras de palavras-chave, base zero-shot e base few-shot. Frases novas: LoRA 0,55 de acerto exato (0,50 a 0,58 entre sementes) contra 0,12 de regras e few-shot. Frases iguais ao treino: LoRA e regras empatam em 1,00, então esse teste não prova generalização. 26 testes, 8 mutantes pegos (um sobreviveu na 1ª versão e levou a um teste de gradiente melhor).
- **Limites:** dados sintéticos, 1 modelo base, hiperparâmetros sem busca, perda de validação 0,0001 (decorou os moldes), rede e e-mail com 0% a 13% de acerto nas frases novas, 5% de rótulo inventado fora do esquema.
- **Próximo passo:** (1) mais diversidade nos dados de treino (moldes e vocabulário variados, ou chamados reais anonimizados) e remedir o teste novo; (2) ablação de r, épocas e taxa de aprendizado com as 3 sementes; (3) modelo 1,5B para separar erro de capacidade de erro de dados; (4) decodificação restrita ao esquema para zerar rótulo inventado; (5) demo.
