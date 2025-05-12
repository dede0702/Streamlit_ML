# Previsor de Propensão de Compra iFood 🛒

## 🎯 Objetivo do Projeto
Este projeto visa desenvolver um modelo de Machine Learning para prever a propensão de clientes do iFood a comprar um novo gadget em uma futura campanha de marketing direto. O objetivo final é auxiliar na tomada de decisões para otimizar o retorno sobre o investimento em campanhas, permitindo o direcionamento eficaz aos clientes com maior probabilidade de compra.

O projeto culmina em um aplicativo web interativo construído com Streamlit, onde é possível realizar predições em lote (via upload de arquivo CSV) ou para clientes individuais (inserindo dados manualmente), além de analisar as características dos diferentes segmentos de clientes.

## 🛠️ Tecnologias Utilizadas
* **Python 3.10+**
* **PyCaret:** Para automatização e simplificação do fluxo de Machine Learning (pré-processamento, treinamento, tuning e avaliação de modelos).
* **Streamlit:** Para a criação do aplicativo web interativo.
* **Pandas:** Para manipulação de dados.
* **NumPy:** Para operações numéricas.
* **Plotly Express:** Para a geração de gráficos interativos no aplicativo.
* **Pillow:** Para manipulação de imagens (logotipos).
* **Scikit-learn, LightGBM (via PyCaret):** Bibliotecas de Machine Learning subjacentes.

## 📂 Estrutura de Pastas Sugerida

![image](https://github.com/user-attachments/assets/0978792a-a20b-42bf-9670-822a5829f5e0)




## 🚀 Etapas do Projeto

1.  **Definição do Problema e Coleta de Dados:**
    * O problema é uma classificação binária: prever se um cliente irá ou não aderir a uma campanha (`Response = 1` ou `0`).
    * Foi utilizado um dataset (`data.csv`) de uma campanha piloto com 2.240 clientes, contendo informações demográficas, de relacionamento com a empresa e histórico de compras.

2.  **Pré-processamento e Engenharia de Features:**
    * **Limpeza Inicial:** Tratamento de valores ausentes (ex: imputação da mediana para a feature `Income`).
    * **Engenharia de Features Manual:** Criação de novas features para enriquecer o modelo, como:
        * `Dt_Customer_Days`: Tempo de relacionamento do cliente em dias.
        * `Age`: Idade do cliente.
        * `Spent`: Gasto total do cliente em diferentes categorias de produtos.
        * `Living_With`: Situação de moradia simplificada (Sozinho, Com Parceiro).
        * `Children`: Número total de crianças e adolescentes em casa.
        * `Family_Size`: Tamanho total da família.
        * `Is_Parent`: Flag binária indicando se o cliente tem filhos/adolescentes.
    * **Remoção de Colunas:** Exclusão de colunas irrelevantes ou com variância zero (ex: `ID`, `Z_CostContact`, `Z_Revenue`).
    * **Conversão de Tipos:** Ajuste dos tipos de dados (ex: `datetime` para `Dt_Customer`, `category` para features categóricas como `Education`, `Marital_Status`, `Living_With`, `Is_Parent`).

3.  **Treinamento do Modelo com PyCaret:**
    * **`setup()`:** Configuração do ambiente PyCaret, definindo o DataFrame, a variável alvo (`Response`), aplicando transformações como normalização (`minmax`), transformação de features para normalidade, tratamento de desbalanceamento de classes (`fix_imbalance=True`), e remoção de multicolinearidade. Uma `session_id` foi usada para garantir reprodutibilidade.
    * **`compare_models()`:** Comparação de diversos algoritmos de classificação, ordenando-os pela métrica **AUC (Area Under the ROC Curve)** para selecionar modelos com bom poder de discriminação geral. O XGBoost foi excluído para evitar problemas de compatibilidade de versão encontrados anteriormente.
    * **`create_model()` e `tune_model()`:** O melhor modelo identificado (LightGBM) foi instanciado e seus hiperparâmetros foram otimizados (`tune_model`) também com foco na métrica **AUC**.
    * **`finalize_model()`:** O modelo tunado foi finalizado (treinado no conjunto completo de dados de treino + validação).
    * **`save_model()`:** O pipeline completo (incluindo pré-processamento e o modelo LightGBM treinado) foi salvo como `modelo_LGBM.pkl` na pasta `pickle/`.

4.  **Desenvolvimento do Aplicativo Streamlit (`app_streamlit.py`):**
    * **Interface com Abas:**
        * **Predição via CSV:** Permite ao usuário carregar um arquivo CSV com múltiplos clientes. O aplicativo pré-processa os dados, aplica o modelo salvo e exibe as predições (score de probabilidade para a classe 1 e a classe predita 0 ou 1).
        * **Predição Online Individual:** Um formulário onde o usuário pode inserir manualmente os dados de um único cliente, simulando um novo cadastro. A predição é gerada e exibida de forma destacada. Um histórico dessas predições online é mantido e pode ser baixado ou limpo.
        * **Analytics Simplificado:** Após a predição via CSV, esta aba permite ao usuário explorar visualmente as diferenças entre os clientes previstos como propensos (1) e não propensos (0). O usuário pode selecionar features numéricas e categóricas para visualizar distribuições comparativas através de boxplots, histogramas e gráficos de barras.
    * **Ajuste de Threshold:** Um controle na sidebar permite ao usuário final ajustar o limiar de decisão (threshold de probabilidade) para classificar um cliente como propenso (1) ou não (0). Todas as predições e análises no aplicativo são atualizadas dinamicamente com base neste threshold.
    * **Pré-processamento Consistente:** A mesma lógica de engenharia de features manual do treinamento é aplicada aos dados de entrada no Streamlit antes de submetê-los ao pipeline do PyCaret.
    * **Carregamento do Modelo:** O modelo `modelo_LGBM.pkl` é carregado usando `pycaret.classification.load_model` e cacheado para performance.

## ⚙️ Como Usar o Projeto

**Pré-requisitos:**
* Python (idealmente a mesma versão usada no desenvolvimento, ex: 3.10).
* Git (para clonar o repositório, se aplicável).

**Configuração do Ambiente:**
1.  **Clone o Repositório (se estiver no GitHub):**
    ```bash
    git clone <url_do_seu_repositorio_github>
    cd <nome_do_repositorio>
    ```
2.  **Crie e Ative um Ambiente Virtual:**
    ```bash
    python -m venv venv 
    # No Windows:
    venv\Scripts\activate
    # No macOS/Linux:
    source venv/bin/activate
    ```
3.  **Instale as Dependências:**
    Certifique-se de que o arquivo `requirements.txt` (fornecido na seção anterior ou gerado com `pip freeze` de um ambiente limpo e funcional) está na raiz do projeto.
    ```bash
    pip install -r requirements.txt
    ```

**Executando o Aplicativo Streamlit:**
1.  Certifique-se de que o modelo treinado (`modelo_LGBM.pkl`) está na pasta `pickle/` e as imagens (`icon_ifood.jpeg`, `ifood_app.png`) estão na pasta `image/`, ambas relativas ao script `app_streamlit.py`.
2.  No terminal, com o ambiente virtual ativado e na pasta raiz do projeto, execute:
    ```bash
    streamlit run app_streamlit.py
    ```
3.  O aplicativo será aberto automaticamente no seu navegador padrão.

**Interagindo com o Aplicativo:**
* **Sidebar:** Use o "Painel de Controle" para ajustar o threshold de decisão. Informações sobre o modelo e como o threshold funciona estão disponíveis.
* **Aba "Predição via CSV":** Faça upload de um arquivo CSV (com colunas similares ao `data.csv` original) para obter predições em lote. Os resultados e um resumo serão exibidos, e você poderá baixar o CSV com as predições.
* **Aba "Predição Online Individual":** Preencha o formulário com os dados de um cliente para uma predição instantânea. As predições são adicionadas a um histórico que pode ser visualizado, baixado ou limpo.
* **Aba "Analytics Simplificado":** Após processar um CSV, explore esta aba para visualizar graficamente as diferenças nas características dos clientes classificados como propensos versus não propensos, com base no threshold definido.

## 📊 Resultados do Modelo (`modelo_LGBM.pkl`)

O modelo final escolhido foi um **Light Gradient Boosting Machine (LightGBM)**, otimizado com foco na métrica **AUC (Area Under the ROC Curve)**.

* **AUC no Conjunto de Teste (Hold-out): 0.90**
    * Este é um resultado **excelente**, indicando que o modelo possui um ótimo poder de discriminação entre os clientes que são propensos a comprar e os que não são. Uma pontuação de AUC de 0.90 significa que há 90% de chance de o modelo atribuir uma probabilidade de compra maior a um cliente aleatório que compraria do que a um cliente aleatório que não compraria.
* **Outras Métricas (com threshold padrão de 0.5, podem variar com o ajuste do usuário):**
    * **Acurácia:** ~87-88%
    * **Precisão (para classe 1 "Propenso"):** ~57-60% (Quando o modelo prevê "compra", acerta em cerca de 60% das vezes).
    * **Revocação (Recall para classe 1 "Propenso"):** ~49% (O modelo identifica cerca de metade dos clientes que realmente comprariam).
    * **F1-Score (para classe 1):** ~53-54%
* **Matriz de Confusão (Exemplo com threshold 0.5 para o modelo LGBM focado em AUC):**
    * Verdadeiros Negativos (VN): ~533
    * Falsos Positivos (FP): ~37
    * Falsos Negativos (FN): ~52
    * Verdadeiros Positivos (VP): ~50
    *(Estes valores podem variar ligeiramente dependendo da execução exata do split de teste, mas o AUC é a medida mais estável da qualidade do modelo aqui).*

**Importância da Curva ROC e do Threshold Ajustável:**
Uma curva ROC com AUC de 0.90 está bem distante da linha de chance e próxima do "canto ideal" (alta taxa de verdadeiros positivos, baixa taxa de falsos positivos). Isso significa que o modelo oferece uma **excelente gama de trade-offs possíveis entre Precisão e Revocação** ao se variar o limiar de decisão. A funcionalidade no aplicativo Streamlit que permite ao cliente final **ajustar o threshold** é crucial, pois permite adaptar a estratégia de predição do modelo aos objetivos específicos de cada campanha de marketing (maximizar alcance, minimizar custos, etc.).

**Análise de Features:**
A aba "Analytics" permite uma exploração visual de como diferentes features (como "Income", "Age", "Spent") se distribuem entre os grupos de clientes previstos como propensos ou não. Por exemplo, observou-se que clientes com "Income" (Renda) maior tendem a ser classificados como mais propensos. Para features com outliers visuais (como "Income"), a visualização nos gráficos é ajustada (mostrando do 1º ao 99º percentil) para melhor clareza da distribuição principal.

## 🚀 Próximos Passos (Sugestões)
* Implementar o deploy do aplicativo na Streamlit Community Cloud (conforme o guia já discutido).
* Aprofundar a análise de features na aba "Analytics" com mais opções de gráficos ou estatísticas descritivas.
* Explorar técnicas de Cost-Sensitive Learning se os custos de Falsos Positivos e Falsos Negativos forem bem definidos.
* Monitorar o desempenho do modelo em produção e retreiná-lo periodicamente com novos dados.
