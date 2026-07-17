# Roteiro de Apresentação: IntraSOM Kohonen Maps Analyzer

Este documento traz o roteiro estruturado para a apresentação do seminário, alternando entre os slides e o dashboard em execução.

---

## Estrutura da Apresentação

### Slide 1: Capa (IntraSOM Kohonen Maps Analyzer)
**Conteúdo Visual:**
* Título do projeto
* Autor: Cauã Vitor
* UFRN - Julho de 2026

**FALA:**
* "Bom dia, professor. Bom dia, pessoal. Vou apresentar nosso trabalho sobre o uso de mapas auto-organizáveis de Kohonen. A ideia aqui é agrupar e visualizar dados complexos sem precisar de rótulos. A gente aplicou isso em séries temporais e também na organização de notícias."

---

### Slide 2: 1. O Desafio da Alta Dimensionalidade
**Conteúdo Visual:**
* O problema da alta dimensionalidade em séries temporais e textos.
* Limitações de métodos lineares (PCA) e puramente gráficos (t-SNE/UMAP).
* Gráfico comparativo.

**FALA:**
* "Trabalhar com dados de muitas dimensões é sempre difícil. Se a gente usa PCA, perde relações não-lineares. Se usa t-SNE ou UMAP, a visualização fica boa, mas não dá pra classificar novos dados em tempo real porque eles não geram uma grade matemática parametrizada. É aí que entra o SOM. Ele projeta os dados em uma grade 2D e mantém os vizinhos próximos."

---

### Slide 3: 2. SOM em 60 Segundos: Intuição Visual
**Conteúdo Visual:**
* Malha hexagonal e neurônios-protótipo.
* Preservação da topologia original.
* Exemplo de U-Matrix.

**FALA:**
* "Pensem no SOM como uma malha de hexágonos. Cada hexágono é um neurônio que atua como representante de um grupo de dados parecidos. A escolha de células hexagonais é boa porque cada neurônio tem exatamente seis vizinhos à mesma distância, o que facilita o aprendizado de vizinhança. A U-Matrix ajuda a ver onde estão as fronteiras de distância entre esses grupos."

---

### Slide 4: 3. Os Datasets de Teste
**Conteúdo Visual:**
* Dataset 1: Synthetic Control (600 séries, 6 classes, malhas 5x5 a 20x20).
* Dataset 2: 20 Newsgroups (400 notícias, 4 classes, malha 10x10).
* Dataset 3: 6News (317 notícias brasileiras, 6 classes, malha 10x10).

**FALA:**
* "A gente testou o algoritmo em três cenários reais. Primeiro, nas séries temporais do Synthetic Control, que têm 600 sinais de controle divididos em 6 tipos de ondas. Depois, partimos para os textos com o 20 Newsgroups, que tem 4 classes temáticas de notícias em inglês. E por fim, o 6News, que é uma base brasileira com 6 categorias e 317 documentos."

---

### 🖥️ AÇÃO NO DASHBOARD (Mudar de aba)
* **Link:** `https://intrasom-kohonen.vercel.app`
* **Passo 1:** Mostrar a aba "Synthetic Control".
* **Passo 2:** Mudar os tamanhos de mapa (5x5, 10x10, 20x20) e mostrar as células mudando.
* **Passo 3:** Clicar em um neurônio para ver o gráfico lateral com os sinais e a média.
* **Passo 4:** Trocar o modo de cor superior de "Classes" para "U-Matrix" para mostrar as fronteiras matemáticas.

**FALA (Intercalando com a navegação):**
* "Aqui no dashboard da aba de séries sintéticas, dá pra ver a malha em ação. Se eu mudar o tamanho da malha, a rede se reajusta. Clicando em qualquer célula, o gráfico lateral mostra os sinais brutos agrupados ali e a curva média de controle. Se eu trocar para o modo U-Matrix, a gente enxerga as barreiras de distância matemática entre os agrupamentos."

---

### Slide 5: 4. O Coração do Projeto: Dashboard Interativo
**Conteúdo Visual:**
* Integração da malha com dados brutos em tempo real.
* Processamento local das projeções de PCA no próprio navegador.
* Uso de chamadas de nuvem (Hugging Face API) para Sentence-BERT de forma serverless.

**FALA:**
* "A gente montou essa interface para ligar a malha matemática diretamente com os gráficos de dados brutos. Em vez de rodar um backend pesado para tudo, a parte de projeção de PCA e busca do neurônio vencedor roda localmente no navegador do usuário. E o modelo Sentence-BERT roda por chamada de API na nuvem da Hugging Face, deixando a infraestrutura limpa."

---

### Slide 6: 5. Resultado 1: Séries Temporais Organizadas
**Conteúdo Visual:**
* Agrupamento visual das classes de controle na malha 10x10.
* Zonas de transição e comportamento de pureza.

**FALA:**
* "Nos resultados de séries temporais, dá pra ver que o mapa funciona bem. Muitas classes com comportamento parecido, como as de tendência crescente e de deslocamento, acabam ficando próximas no mapa. Os neurônios com pureza menor indicam as zonas de transição física dos sinais originais."

---

### Slide 7: 6. Resultado 2: Métricas vs. Baselines
**Conteúdo Visual:**
* Tabela comparativa (SOM, K-Means, Clustering Aglomerativo, DBSCAN).
* Métricas: ARI (0.6557) e NMI (0.7715).

**FALA:**
* "Para validar matematicamente, a gente comparou com outros métodos clássicos. O SOM 10x10 alcançou ARI de 0.6557 e NMI de 0.7715. Ele superou o K-Means comum e ficou muito próximo de métodos hierárquicos, com a vantagem de dar uma representação de malha contínua em vez de apenas grupos isolados."

---

### Slide 8: 7. Resultado 3: Representação de Texto (Comparativo)
**Conteúdo Visual:**
* Comparação de TF-IDF vs SBERT nos datasets de texto.
* 20 Newsgroups (ARI: TF-IDF 0.1981 vs SBERT 0.4177).
* 6News (ARI: TF-IDF 0.1286 vs SBERT 0.1178).

**FALA:**
* "Aqui vale reparar na comparação de textos. No 20 Newsgroups em inglês, o SBERT melhorou muito os resultados porque ele captura a semântica, enquanto o TF-IDF depende de termos exatos. Mas na base brasileira 6News, o TF-IDF foi um pouco melhor. O dataset 6News é menor, tem mais classes e termos técnicos específicos em português. Além disso, existe uma sobreposição real de temas entre as classes 'Política' e 'Polícia e Direitos', já que ambas falam de governança e políticas públicas. Isso confunde o SBERT sem um fine-tuning prévio."

---

### 🖥️ AÇÃO NO DASHBOARD (Mudar de aba)
* **Link:** `https://intrasom-kohonen.vercel.app`
* **Passo 1:** Selecionar a aba "Clusterização de Textos".
* **Passo 2:** Escolher uma notícia pré-carregada do dropdown e ver o card de resultado surgir.
* **Passo 3:** Mostrar o neurônio vencedor pulsando no mapa com o ripple.
* **Passo 4:** Escrever uma pequena frase na caixa de texto ao vivo (ex: "O campeonato de futebol terminou com vitória do time da casa") para mostrar o sistema projetando o texto digitado em tempo real na malha de Kohonen.

**FALA (Intercalando com a navegação):**
* "Agora, mudando para a aba de textos, a gente vê a classificação semântica. Ao escolher uma das notícias de exemplo, o card de resultado mostra a classe predita e a confiança da rede. Ao mesmo tempo, o neurônio vencedor brilha com um efeito de onda no mapa, mostrando onde aquela ideia está localizada. Se eu escrever qualquer frase na caixa de texto, o sistema atualiza o neurônio vencedor na hora."

---

### Slide 9: 8. Aplicações em IA, Direito e Gestão Pública
**Conteúdo Visual:**
* Jurimetria e Visual Law (organização de processos e acórdãos).
* Triagem inteligente de documentos na administração pública.

**FALA:**
* "As aplicações reais disso são amplas. No Direito, por exemplo, dá pra organizar milhares de acórdãos e petições por similaridade no mapa para ajudar juízes na triagem de casos. Na gestão pública, a gente consegue fazer o mesmo para relatórios e pareceres de forma automática, agrupando temas parecidos."

---

### Slide 10: 9. Conclusão
**Conteúdo Visual:**
* Síntese das contribuições e pontos fortes.
* Limitações do estudo e ideias de trabalhos futuros.

**FALA:**
* "Para concluir, o SOM se mostrou uma ferramenta útil e bastante intuitiva para visualizar alta dimensão. Como autocrítica, vale destacar que nossos resultados vêm de um experimento com semente (seed) fixa de treino, sem uma repetição estatística usando múltiplas sementes aleatórias, o que seria necessário para comprovar a estabilidade absoluta. Como próximos passos, a gente quer testar malhas toroidais para eliminar os efeitos de borda nas extremidades do mapa."
