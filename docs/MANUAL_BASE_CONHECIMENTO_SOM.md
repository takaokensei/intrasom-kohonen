# 📘 MANUAL DE REFERÊNCIA E BASE DE CONHECIMENTO: MAPAS AUTO-ORGANIZÁVEIS (SOM) & BIBLIOTECA INTRASOM

**Autor do Documento:** Compilação Técnica e Acadêmica  
**Área de Domínio:** Aprendizado Não-Supervisionado, Redes Neurais de Kohonen, Processamento de Linguagem Natural, Geociências e Visual Law  
**Status:** Documento Oficial de Consulta e Referência Rápida (Base de Conhecimento)

---

## 📋 SUMÁRIO EXECUTIVO

1. [FUNDAMENTOS TEÓRICOS DE MAPAS AUTO-ORGANIZÁVEIS (SOM / KOHONEN)](#1-fundamentos-teóricos-de-mapas-auto-organizáveis-som--kohonen)
   - 1.1. O Desafio da Alta Dimensionalidade e Limitações dos Métodos Tradicionais
   - 1.2. Inspiração Biológica e Princípios de Auto-Organização (Entropia e Emergência)
   - 1.3. Arquitetura da Rede e o Mecanismo Triplo (Competição, Cooperação e Adaptação)
   - 1.4. Preservação Topológica e Medidas de Distância Dupla
   - 1.5. Quantização Vetorial e Representação de Protótipos (BMU)
2. [VISUALIZAÇÃO, MATRIZ U E SEGMENTAÇÃO DE CLUSTERS](#2-visualização-matriz-u-e-segmentação-de-clusters)
   - 2.1. A Matriz U (Unified Distance Matrix) e Interpolação de Voronoi
   - 2.2. Matriz de Gradiente (GC) e Análise de Fronteiras
   - 2.3. Agrupamento Automático: Transgressão Watershed e Índice Davies-Bouldin
   - 2.4. Estruturas Hierárquicas (Árvores de Mapas e Sub-mapas)
3. [GUIA COMPLETO DE PARAMETRIZAÇÃO TÉCNICA (OS 7 PARÂMETROS CHAVE)](#3-guia-completo-de-parametrização-técnica-os-7-parâmetros-chave)
   - 3.1. Dimensionamento do Mapa (Fórmula Empírica de Vesanto: $5\sqrt{N}$)
   - 3.2. Vizinhança Inicial (Raio Global ~80%) e Vizinhança Final (Raio Unitário)
   - 3.3. Épocas de Treinamento e Convergência (500 Épocas)
   - 3.4. Modo de Treinamento: Batch (Lote) vs. Online/Sequencial (Stochastic)
   - 3.5. Inicialização de Pesos: Linear (PCA) vs. Aleatória
   - 3.6. Geometria da Grade: Malha Hexagonal (HEX) vs. Retangular (RECT) (A Estabilidade dos 120º)
   - 3.7. Topologia da Rede: Toroidal (Toroid ON) vs. Plana (Toroid OFF) e o "Efeito Mapa de Karnaugh"
4. [A BIBLIOTECA INTRASOM: ARQUITETURA, DIFERENCIAIS E IMPLEMENTAÇÃO](#4-a-biblioteca-intrasom-arquitetura-diferenciais-e-implementação)
   - 4.1. Visão Geral da Biblioteca Python (`intrasom`)
   - 4.2. Tratamento Exclusivo de Dados Faltantes (*Previous Epoch Method*)
   - 4.3. Treinamento de Big Data com Bootstrap
   - 4.4. Mapeamento Automático, Exportação JSON/Parquet e Relatórios de Qualidade
5. [APLICAÇÕES PRÁTICAS E CASOS DE USO](#5-aplicações-práticas-e-casos-de-uso)
   - 5.1. Geociências e Georrecursos (Perfuração, Perfis de Poços, Sísmica)
   - 5.2. Análise Socioeconômica e Demográfica (O Estudo Mundial da Pobreza)
   - 5.3. Processamento de Linguagem Natural e Semântica de Textos (20 Newsgroups & 6News)
   - 5.4. Jurimetria, Visual Law e Administração Pública
6. [MATRIZ COMPARATIVA E GLOSSÁRIO TÉCNICO](#6-matriz-comparativa-e-glossário-técnico)
   - 6.1. Tabela Consolidada de Parâmetros e Impactos no Treinamento
   - 6.2. Glossário de Siglas e Termos-Chave

---

# 1. FUNDAMENTOS TEÓRICOS DE MAPAS AUTO-ORGANIZÁVEIS (SOM / KOHONEN)

### 1.1. O Desafio da Alta Dimensionalidade e Limitações dos Métodos Tradicionais
Na ciência de dados moderna, sistemas reais são caracterizados por um elevado número de variáveis ($n$-dimensionalidade). Tentar visualizar e agrupar esses dados apresenta desafios estruturais severos quando utilizados algoritmos convencionais:

* **K-means:** Exige a especificação prévia e rígida do número de agrupamentos ($k$). Representa os grupos através de um único ponto central (centroide médio), falhando drasticamente em conjuntos com distribuições não esféricas, contínuas ou de geometrias complexas.
* **PCA (Análise de Componentes Principais):** Redução linear que busca projetar os dados na direção de máxima variância. Por ser estritamente linear, ignora correlações complexas e não-lineares inerentes a fenômenos biológicos, físicos e linguísticos.
* **t-SNE / UMAP:** Ótimas ferramentas visuais para separação em 2D/3D. No entanto, são técnicas puramente gráficas não-paramétricas: não criam protótipos de classificação parametrizados que permitam inferir dados novos em tempo real e frequentemente destroem a noção de distância contínua global entre clusters distantes.

O **Map de Kohonen (Self-Organizing Map - SOM)** resolve essa lacuna ao realizar um mapeamento não-linear, ordenado e contínuo de dados de alta dimensão em uma grade regular de baixa dimensão (geralmente 2D), preservando simultaneamente as relações de vizinhança topológica e criando neurônios-protótipo paramétricos.

---

### 1.2. Inspiração Biológica e Princípios de Auto-Organização
Os Mapas de Kohonen foram propostos na década de 1980 pelo físico e engenheiro finlandês **Teuvo Kohonen**, baseados nas primeiras ideias do físico alemão Christoph von der Malsburg (1973) sobre a neurofisiologia do córtex cerebral.

#### Princípios de Auto-Organização e Entropia:
* **Entropia (Boltzmann, 1872):** Medida da desordem ou do número de estados microscópicos possíveis de um sistema. Quanto mais informação é necessária para descrever um sistema, maior a sua entropia/desordem.
* **Emergência:** Padrões globais organizados emergem espontaneamente de interações locais descentralizadas entre os neurônios, sem a necessidade de um supervisor ou rótulos prévios (*Aprendizado Não-Supervisionado*).
* **Mapeamento Somatotópico:** No cérebro humano, estímulos motores ou sensoriais vizinhos (como os movimentos dos dedos da mão ou da língua) ativam regiões de neurônios contíguas no córtex. O SOM replica computacionalmente essa ordenação espacial.

---

### 1.3. Arquitetura da Rede e o Mecanismo Triplo
A rede SOM é constituída por uma arquitetura simples de duas camadas:
1. **Camada de Entrada:** Recebe os vetores $n$-dimensionais das amostras $x = [x_1, x_2, \dots, x_n]$.
2. **Camada de Saída (Mapa):** Grade bidimensional de neurônios. Cada neurônio $i$ possui um vetor de pesos sinápticos $w_i = [w_{i1}, w_{i2}, \dots, w_{in}]$ de mesma dimensão que o dado de entrada.

```
       Espaço de Entrada (n-dimensional)
          [X1, X2, X3, ..., Xn]
             /   |   \   \
            /    |    \   \
     ======/=====|=====\===\======  Conexões Sinápticas (Pesos W)
    |     /      |      \   \     |
    |  Neurônio 1   Neurônio 2   |  Camada de Saída 2D
    |     (O)  <---->  (O)       |  (Interação de Vizinhança
    |      ^            ^        |   Inibitória/Excitadora)
    |      |            |        |
    |  Neurônio 3   Neurônio 4   |
    |     (O)  <---->  (O)       |
     =============================
```

O ciclo de aprendizado do SOM ocorre através de três etapas fundamentais:

#### 1. Competição:
Para cada amostra de entrada $x$, todos os neurônios da grade calculam sua distância matemática em relação a $x$. O neurônio com o vetor de pesos mais próximo de $x$ é declarado o **BMU (Best Matching Unit / Unidade de Melhor Correspondência)**:

$$i(x) = \arg\min_j \|x - w_j\|$$

Comumente utiliza-se a distância Euclidiana:

$$\|x - w_j\| = \sqrt{\sum_{k=1}^n (x_k - w_{jk})^2}$$

#### 2. Cooperação:
O neurônio vencedor (BMU) determina uma área de vizinhança topológica ao seu redor na grade de saída. Neurônios situados dentro dessa vizinhança são ativados cooperativamente. A intensidade da ativação é modulada por uma função de vizinhança $h_{ij}(t)$, tipicamente Gaussiana:

$$h_{ij}(t) = \exp\left(-\frac{d_{ij}^2}{2\sigma(t)^2}\right)$$

Onde $d_{ij}$ é a distância física entre os neurônios $i$ e $j$ na grade 2D, e $\sigma(t)$ é o raio de vizinhança que diminui com o tempo (épocas $t$).

#### 3. Adaptação Sináptica:
Os pesos do BMU e de seus vizinhos são ajustados para se moverem em direção à amostra de entrada $x$:

$$w_j(t+1) = w_j(t) + \alpha(t) \cdot h_{ij}(t) \cdot [x(t) - w_j(t)]$$

Onde $\alpha(t)$ é a taxa de aprendizado (*learning rate*), que também decai ao longo do treinamento.

---

### 1.4. Preservação Topológica e Medidas de Distância Dupla
Em um SOM treinado com sucesso, existem duas noções distintas de distância que devem ser compreendidas:
* **Distância no Espaço de Atributos ($n$-dimensional):** Distância matemática entre os vetores de pesos dos neurônios.
* **Distância na Grade de Saída (2D):** Distância física das coordenadas dos neurônios na grade geométrica.

A **Preservação Topológica** garante que se duas amostras estão próximas no espaço $n$-dimensional original, elas ativarão o mesmo neurônio ou neurônios vizinhos na grade 2D.

---

### 1.5. Quantização Vetorial e Representação de Protótipos
O SOM realiza **Quantização Vetorial (Vector Quantization)**: ele reduz o espaço contínuo de entrada a um número discreto de vetores de referência (os pesos dos neurônios). Cada neurônio atua como um **protótipo** da sub-região do espaço de atributos que ele representa.

---

# 2. VISUALIZAÇÃO, MATRIZ U E SEGMENTAÇÃO DE CLUSTERS

### 2.1. A Matriz U (Unified Distance Matrix) e Interpolação de Voronoi
Inventada por **Alfred Ultsch**, a Matriz U é a principal ferramenta de visualização de estruturas de clusters em um SOM:

* Para cada neurônio, a Matriz U calcula a distância média (no espaço $n$-dimensional) entre o vetor de pesos daquele neurônio e os vetores de pesos de seus vizinhos diretos na grade.
* **Dimensões da Matriz U:** Para uma grade de neurônios de tamanho $N \times M$, a Matriz U completa interpolada possui dimensões de $(2N-1) \times (2M-1)$, pois exibe tanto as células dos neurônios quanto as células intermediárias que representam as distâncias de borda entre vizinhos.
* **Leitura Visual:** 
  * **Tons Frios / Baixos Valores (Vales):** Neurônios com vetores de pesos muito similares entre si. Indicam o centro de um cluster homogêneo.
  * **Tons Quentes / Altos Valores (Montanhas/Fronteiras):** Neurônios cujos pesos diferem drasticamente de seus vizinhos. Indicam **fronteiras de separação** entre clusters diferentes.

---

### 2.2. Matriz de Gradiente (GC) e Análise de Fronteiras
Para bases de dados altamente contínuas, a **Matriz de Gradiente (Gradient Matrix - GC)** calcula o vetor gradiente de alteração entre os pesos adjacentes, oferecendo maior nitidez visual nas linhas de separação em relação à Matriz U convencional.

---

### 2.3. Agrupamento Automático: Transgressão Watershed e Índice Davies-Bouldin
Para evitar que o usuário precise delimitar visualmente e manualmente quais neurônios pertencem a qual grupo, utilizam-se técnicas automáticas sobre os neurônios já treinados:

```
        Visualização 3D da Matriz U (Vales e Cristas)

         Fronteira (Crista)          Fronteira (Crista)
              /\                          /\
             /  \    Cluster A           /  \    Cluster B
            /    \   (Vale 1)           /    \   (Vale 2)
   ________/      \____________________/      \___________
   Inundação Watershed ------->            ------->
```

1. **Transformada Watershed (Linha Divisória de Águas):** Tratando a Matriz U como um relevo topográfico 3D (onde vales são mínimos de distância e cristas são barreiras), a água "inunda" os vales a partir de marcadores locais. Onde as bacias de inundação se encontram, constroem-se as linhas de separação dos clusters automatizados.
2. **Índice Davies-Bouldin (DB):** Métrica de validação de agrupamento usada para identificar o número ótimo de clusters ($k$). Avalia a razão entre a dispersão interna dos clusters e a separação entre eles.
   * *Recomendação Prática:* Em dados com zonas de transição contínua (ex: geologia e sinais temporais), deve-se prestar atenção tanto no **primeiro** quanto no **segundo mínimo global** do índice DB, pois o segundo mínimo frequentemente captura divisões físicas mais representativas.

---

### 2.4. Estruturas Hierárquicas (Árvores de Mapas e Sub-mapas)
Quando uma região do SOM exibe um cluster grande mas internamente ruidoso, é possível isolar apenas as amostras que ativaram os neurônios daquela região e treinar um **sub-mapa recursivo**. Isso gera uma estrutura hierárquica em árvore (*Tree-SOM*), permitindo explorar subclasses com alto grau de detalhamento sem re-treinar todo o mapa global.

---

# 3. GUIA COMPLETO DE PARAMETRIZAÇÃO TÉCNICA (OS 7 PARÂMETROS CHAVE)

Este capítulo consolida as diretrizes teóricas e práticas para a configuração ideal do SOM, respondendo diretamente aos requisitos de teste propostos.

```
+-----------------------------------------------------------------------------------+
|                  RESUMO DOS 7 PARÂMETROS CHAVE DE CONFIGURAÇÃO                    |
+-----------------------------------------------------------------------------------+
| 1. Tamanho do Mapa: Fórmula empírica de Vesanto: 5 * sqrt(N)                       |
| 2. Vizinhança Inicial: ~80% do diâmetro total da grade (cooperação global)        |
| 3. Vizinhança Final: Decaimento progressivo até 1 neurônio (ajuste fino local)    |
| 4. Épocas de Treinamento: 500 épocas (garante estabilização da U-Matrix e QE)     |
| 5. Modo de Treinamento: BATCH (Somatório vetorial síncrono por época)             |
| 6. Inicialização de Pesos: LINEAR VIA PCA (Eixos de maior variância dos dados)    |
| 7. Geometria & Topologia: MALHA HEXAGONAL + TOPOLOGIA TOROIDAL (Toroid ON)        |
+-----------------------------------------------------------------------------------+
```

---

### 3.1. Dimensionamento do Mapa (Fórmula Empírica de Vesanto)
Para evitar que o mapa seja pequeno demais (incapaz de separar grupos) ou grande demais (causando *overfitting* onde cada dado vira um neurônio isolado), utiliza-se a **Equação Empírica de Juha Vesanto**:

$$M_{total} = 5 \times \sqrt{N}$$

Onde $N$ é a quantidade total de instâncias do dataset.
* *Exemplo:* Para $N = 600$ amostras:
  $$M_{total} = 5 \times \sqrt{600} \approx 5 \times 24.49 = 122.45 \text{ neurônios}$$
  Um mapa próximo de $10 \times 10$ (100 neurônios) ou $12 \times 10$ (120 neurônios) é o tamanho matematicamente ideal.

---

### 3.2. Vizinhança Inicial (Raio Global ~80%) e Vizinhança Final (Raio Unitário)
O treinamento do SOM deve ser dividido em duas fases distintas moduladas pelo raio de vizinhança $\sigma(t)$:

1. **Fase de Ordenação Global (Vizinhança Inicial ~80%):**
   * No início ($t=0$), o raio de vizinhança deve cobrir cerca de **80% do raio/diâmetro máximo da grade**.
   * *Objetivo:* Garantir cooperação biológica global. Todos os neurônios do mapa se movem juntos de forma coordenada para desdobrar a rede sobre os dados sem nós ou torções.
2. **Fase de Ajuste Fino (Vizinhança Final = 1 neurônio):**
   * Conforme as épocas avançam, o raio decai exponencialmente ou linearmente até atingir a vizinhança mínima ($\sigma = 1$ neurônio ou raio nulo).
   * *Objetivo:* Permitir que cada neurônio se ajuste pontualmente aos detalhes locais das amostras da sua microrregião.

---

### 3.3. Épocas de Treinamento e Convergência (500 Épocas)
* Treinamentos muito curtos (< 50 épocas) deixam o mapa em um estado parcialmente ordenado com alto Erro Quantizado (*Quantization Error - QE*).
* **Configuração Recomendada de 500 Épocas:** Proporciona tempo computacional suficiente para que a transição do raio amplo (80%) para o raio unitário (1) ocorra de forma suave, permitindo acompanhar a estabilização do Erro Quantizado e do Erro Topográfico (*Topographic Error - TE*).

---

### 3.4. Modo de Treinamento: Batch (Lote) vs. Online/Sequencial (Stochastic)
Existem duas abordagens para atualizar os pesos da rede:

```
 Treinamento Online (Stochastic):       Treinamento Batch (Lote):
 [Amostra 1] -> Atualiza Pesos         [Todas as N Amostras]
 [Amostra 2] -> Atualiza Pesos                   |
 [Amostra 3] -> Atualiza Pesos        Acumula Somatório de Voronoi
      ...                                        |
 (Lento, ruidoso, depende da ordem)    Atualiza TODOS os Pesos 1x por Época
                                       (Rápido, Estável, Determinístico)
```

1. **Treinamento Online (Stochastic):** Os pesos são atualizados a cada amostra lida. Sensível à ordem dos dados, lento para grandes volumes e ruidoso.
2. **Treinamento Batch (Lote) - *ALTAMENTE RECOMENDADO*:**
   * O algoritmo lê **todas** as amostras da época primeiro, associa cada amostra ao seu BMU correspondente e acumula a influência ponderada nas células de Voronoi.
   * Ao final da época, atualiza o vetor de pesos de todos os neurônios de uma só vez via média ponderada:

$$w_j(t+1) = \frac{\sum_{k=1}^N h_{ij(k)}(t) \cdot x_k}{\sum_{k=1}^N h_{ij(k)}(t)}$$

   * **Vantagens:** Não requer uma taxa de aprendizado $\alpha(t)$ manual, é extremamente rápido, estável e completamente determinístico.

---

### 3.5. Inicialização de Pesos: Linear (PCA) vs. Aleatória
1. **Inicialização Aleatória:** Atribui valores aleatórios aos pesos iniciais dos neurônios. Pode levar o mapa a ficar preso em mínimos locais ou gerar torções topológicas ("nós") difíceis de desfazer.
2. **Inicialização Linear (PCA) - *ALTAMENTE RECOMENDADA*:**
   * Calcula os dois primeiros Componentes Principais (PC1 e PC2) do conjunto de dados de entrada.
   * Os vetores de pesos dos neurônios são dispostos ordenadamente ao longo do plano definido por esses dois eixos de maior variância.
   * **Vantagens:** Acelera drasticamente a convergência do treino (exige menos épocas) e garante reprodutibilidade rigorosa.

---

### 3.6. Geometria da Grade: Malha Hexagonal (HEX) vs. Retangular (RECT)
A escolha da disposição geométrica dos neurônios afeta diretamente a simetria da vizinhança:

```
    Grade Retangular (RECT)               Grade Hexagonal (HEX)
       (4 vizinhos diretos,                 (6 vizinhos equidistantes,
        distâncias d=1 e d=sqrt(2))           todas as distâncias d=1)

         [ ]---[ ]---[ ]                       / \ / \ / \
          |  X  |  X  |                       | O | O | O |
         [ ]---[N]---[ ]                       \ / \ / \ / \
          |  X  |  X  |                         | O | O |
         [ ]---[ ]---[ ]                       / \ / \ / \
                                              | O | O | O |
```

* **Grade Retangular (RECT):** Cada neurônio tem 4 vizinhos ortogonais (distância $d=1$) e 4 vizinhos diagonais (distância $d = \sqrt{2} \approx 1.414$). Essa assimetria introduz distorções anisotrópicas no aprendizado de vizinhança.
* **Grade Hexagonal (HEX) - *PADRÃO ABSOLUTO E RECOMENDADO*:**
  * Cada neurônio possui **6 vizinhos adjacentes estritamente equidistantes** ($d=1$).
  * *Fundamentação Biológica/Física:* Na natureza, a geometria hexagonal é a forma espacial mais estável para preenchimento de superfícies contínuas (ângulo mecânico de 120º em favos de abelhas, basaltos vulcânicos e bolhas de sabão). Oferece a melhor simetria espacial para a função de vizinhança Gaussiana.

---

### 3.7. Topologia da Rede: Toroidal (Toroid ON) vs. Plana (Toroid OFF) e o "Efeito Mapa de Karnaugh"
A topologia define as conexões de borda da grade de saída:

```
 Topologia Plana (Toroid OFF):          Topologia Toroidal (Toroid ON):
 Neurônios de borda têm menos            Bordas opostas são conectadas.
 vizinhos (Efeito de Borda).             Grade vira um Toroide ("Rosca").

   +-----------------------+              /-----------------------\
   | N1   N2   N3   N4   N5|             |  N1   N2   N3   N4   N5 |---(Conecta com N21-N25)
   | N6   ..   ..   ..  N10|             |  N6   ..   ..   ..  N10 |
   |N21  N22  N23  N24  N25|             | N21  N22  N23  N24  N25 |---(Conecta com N1-N5)
   +-----------------------+              \-----------------------/
                                           (N1 conecta lateralmente com N5)
```

1. **Topologia Plana (Toroid OFF):**
   * Apresenta o **Efeito de Borda (Edge Effect)**: Neurônios no centro da grade possuem 6 vizinhos, enquanto neurônios nos cantos possuem apenas 2 ou 3 vizinhos.
   * Isso atrai artificialmente protótipos para as bordas e distorce as distâncias da U-Matrix.
2. **Topologia Toroidal (Toroid ON) - *RECOMENDADA PARA TREINO*:**
   * Conecta a borda superior com a inferior e a borda esquerda com a direita. O mapa bidimensional se fecha na forma espacial de um **toroide** (formato de "rosca").
   * **Benefício:** **Eliminação total das bordas.** Todos os neurônios sem exceção possuem exatamente a mesma quantidade de vizinhos conexos.

#### 💡 A Conexão Conceitual com o "Efeito Mapa de Karnaugh":
Na eletrônica digital, o **Mapa de Karnaugh** simplifica expressões booleanas conectando as extremidades opostas da tabela (a primeira coluna é vizinha direta da última coluna; a primeira linha é vizinha da última). 

A topologia toroidal do SOM aplica a exata mesma regra de **adjacência circular**: um neurônio na extrema direita ($x=M$) é vizinho direto do neurônio na extrema esquerda ($x=1$).
* *Observação de Exibição:* Para exibição em telas 2D normais sem parecer um mapa "dobrado/confuso" estilo Karnaugh para leigos, desativa-se o Toroid no momento da plotagem final (Toroid OFF no display), mantendo o Toroid ON durante a fase matemática do treinamento.

---

# 4. A BIBLIOTECA INTRASOM: ARQUITETURA, DIFERENCIAIS E IMPLEMENTAÇÃO

### 4.1. Visão Geral da Biblioteca Python (`intrasom`)
A **IntraSOM** é uma biblioteca de código aberto desenvolvida nativamente em Python para uso em ensino, engenharia e pesquisa (instalável via PyPI: `pip install intrasom`).

* **Stack:** Construída sobre `numpy`, `pandas`, `scikit-learn` e `matplotlib`.
* **Foco:** Treinamento robusto de mapas auto-organizáveis em dados reais complexos, com suporte nativo a malhas hexagonais, topologia toroidal, treinamento Batch e tratamento avançado de dados faltantes.

---

### 4.2. Tratamento Exclusivo de Dados Faltantes (*Previous Epoch Method*)
Em dados de campo (geologia, sensores industriais, séries financeiras), é comum haver dados ausentes (*NaN / Missing Values*). Descartar a linha inteira reduz drasticamente o dataset.

A biblioteca IntraSOM implementa o método exclusivo da **Época Anterior (Previous Epoch Method)**:

```
                Fluxo de Imputação Dinâmica no IntraSOM

       [Dado com Falha: X = (12.4, NaN, 5.1)]
                          |
             Etapa 1: Ajuste Bruto (Ajuste Inicial)
             (Usa apenas os atributos presentes)
                          |
            Identifica BMU da Época Anterior
                          |
             Etapa 2: Ajuste Fino (Imputação)
             Imputa o NaN baseado no peso do BMU da época anterior,
             regularizado pela função do raio de vizinhança.
                          |
       [Vetor Completo Recalibrado para a Próxima Época]
```

* **Funcionamento:** Durante o ajuste fino, se uma amostra possui um atributo faltante, o algoritmo recupera o valor correspondente do vetor de pesos do BMU atribuído a essa amostra na **época anterior**. Esse valor imputado é ponderado dinamicamente pelo raio de vizinhança corrente.
* **Resultado:** Permite que a amostra continue se movendo livremente pela grade sem travar a convergência ou distorcer os centros dos clusters.

---

### 4.3. Treinamento de Big Data com Bootstrap
Para conjuntos de dados massivos (como imagens de satélite, dados sísmicos ou logs com milhões de linhas), a IntraSOM introduz o modo **Bootstrap**:
* A cada época, em vez de processar 100% da base, o algoritmo seleciona uma amostragem aleatória representativa (ex: 10% ou 20% do dataset).
* Isso reduz o tempo de processamento computacional em ordens de grandeza sem comprometer a ordenação topológica global.

---

### 4.4. Mapeamento Automático, Exportação JSON/Parquet e Relatórios de Qualidade
Ao finalizar o treino, a IntraSOM gera automaticamente:
1. `params_SOM_*.json`: Arquivo JSON contendo todos os hiperparâmetros (mapsize, épocas, topologia, métricas).
2. `SOM_*_neurons.parquet` e `results.parquet`: Matrizes de pesos e associações de BMU em formato binário comprimido Parquet de altíssima velocidade.
3. `Intrasom_report_*.txt`: Relatório com as métricas de qualidade do mapa:
   * **Quantization Error (QE):** Média das distâncias entre as amostras e seus respectivos BMUs (mede o ajuste do mapa aos dados).
   * **Topographic Error (TE):** Proporção de amostras cujo segundo BMU mais próximo não é vizinho físico do primeiro BMU na grade (mede a preservação da topologia; valores próximos de 0.0 indicam topologia perfeita).

---

# 5. APLICAÇÕES PRÁTICAS E CASOS DE USO

### 5.1. Geociências e Georrecursos
* **Descrição:** Classificação automática de lithofacies (tipos de rochas) em poços de petróleo a partir de perfis geofísicos (gama-ray, densidade, porosidade).
* **Resultado:** O SOM mapeia as transições suaves entre folhelhos, arenitos e carbonatos sem impor cortes bruscos arbitrários.

---

### 5.2. Análise Socioeconômica e Demográfica (O Estudo Mundial da Pobreza)
* **Descrição:** Agrupamento de países baseado em 39 indicadores socioeconômicos do Banco Mundial (PIB, mortalidade infantil, alfabetização, inflação).
* **Resultado:** Sem receber nenhuma informação geográfica ou econômica prévia, o SOM organiza os países em um gradiente contínuo: na extremidade superior esquerda países nórdicos altamente desenvolvidos (Suécia, Bélgica) e na extremidade oposta países em extrema vulnerabilidade (Etiópia, Zimbábue).

---

### 5.3. Processamento de Linguagem Natural e Semântica de Textos

#### Comparativo de Representações em Datasets de Notícias:

```
            Mapeamento Semântico de Textos na Grade SOM 10x10

      Representação TF-IDF                   Representação SBERT
     (Frequencial / Palavras)               (Embeddings Densos 384D)
     
    [ Cluster 1 ]   [ Misturado ]          [ Tecnologia ]    [ Esportes ]
    (Termos exatos)  (Sinônimos)            (Conceito)        (Conceito)
          |              |                       |                |
     ARI: 0.1981    NMI: 0.2245            ARI: 0.4177      NMI: 0.4828
```

1. **Dataset 20 Newsgroups (4 classes em Inglês - 400 notícias):**
   * **TF-IDF + LSA (20D):** ARI = 0.1981 | NMI = 0.2245. Limitação grave por depender da repetição de palavras exatas.
   * **Sentence-BERT + PCA (20D):** **ARI = 0.4177 | NMI = 0.4828**. O SBERT agrupa notícias por significado e contexto conceitual, gerando territórios perfeitamente definidos na grade hexagonal.
2. **Dataset 6News (6 classes em Português - 317 notícias acadêmicas):**
   * **TF-IDF + LSA (20D):** ARI = 0.1286 | NMI = 0.2120.
   * **Sentence-BERT + PCA (20D):** ARI = 0.1178 | NMI = 0.1822.
   * *Fenômeno Científico Notado:* Em datasets pequenos com vocabulário técnico extremamente específico em português, o TF-IDF pode atuar como uma heurística forte. Além disso, existe sobreposição semântica real entre as categorias "Política" e "Polícia e Direitos" (ambas tratam de governança e políticas públicas), o que exige um *fine-tuning* especializado do modelo de linguagem.

---

### 5.4. Jurimetria, Visual Law e Administração Pública
* **Triagem Inteligente de Processos:** Leitura e vetorização de petições iniciais e decisões judiciais. O SOM agrupa processos por similaridade tese-jurisprudencial em um mapa interativo, permitindo que magistrados identifiquem causas repetitivas instantaneamente.
* **Mapeamento de Editais e Pareceres:** Agrupamento automático de documentos administrativos sem necessidade de rotulagem manual prévia.

---

# 6. MATRIZ COMPARATIVA E GLOSSÁRIO TÉCNICO

### 6.1. Tabela Consolidada de Parâmetros e Impactos no Treinamento

| Parâmetro / Configuração | Opção Recomendada | Impacto no Treinamento e Resultado |
|---|---|---|
| **Tamanho da Grade ($M_{total}$)** | $5 \times \sqrt{N}$ (ex: 10x10 para N=600) | Garante resolução suficiente para separar clusters sem causar overfitting. |
| **Vizinhança Inicial ($\sigma_0$)** | ~80% do diâmetro da grade | Promove cooperação global e desdobra a malha sobre os dados sem nós. |
| **Vizinhança Final ($\sigma_f$)** | Raio unitário ($\sigma = 1$) | Permite o ajuste fino dos vetores de pesos locais. |
| **Número de Épocas** | 500 épocas | Garante tempo suficiente para estabilização do QE e do TE. |
| **Modo de Atualização** | **Batch (Lote)** | Treinamento rápido, síncrono, estável e 100% determinístico. |
| **Inicialização dos Pesos** | **Linear (via PCA)** | Alinha os pesos iniciais aos eixos de maior variância; evita mínimos locais. |
| **Geometria dos Células** | **Hexagonal (HEX)** | Oferece 6 vizinhos equidistantes (120º); elimina distorções diagonais do retângulo. |
| **Topologia da Rede** | **Toroidal (Toroid ON)** | Conecta as bordas da grade em formato de rosca; **elimina o efeito de borda**. |

---

### 6.2. Glossário de Siglas e Termos-Chave

* **BMU (Best Matching Unit):** O neurônio cujo vetor de pesos possui a menor distância matemática em relação à amostra de entrada corrente.
* **QE (Quantization Error / Erro Quantizado):** Média das distâncias entre todas as amostras de entrada e seus respectivos BMUs. Mede o quão bem o mapa se ajustou aos dados.
* **TE (Topographic Error / Erro Topográfico):** Porcentagem de amostras onde o primeiro e o segundo BMUs mais próximos não são vizinhos adjacentes na grade 2D. Mede a preservação da continuidade topológica (ideal: próximo de 0.0).
* **U-Matrix (Unified Distance Matrix):** Matriz visual que exibe a distância Euclidiana entre os vetores de pesos de neurônios vizinhos. Vales (tons frios) representam clusters; cristas (tons quentes) representam fronteiras.
* **Batch Training:** Modo de treinamento onde os pesos dos neurônios são atualizados uma única vez por época a partir do somatório vetorial acumulado de todas as amostras.
* **Topologia Toroidal:** Conectividade circular onde as bordas opostas da grade se unem (efeito rosca/Karnaugh), garantindo que todos os neurônios tenham o mesmo número de conexões de vizinhança.
