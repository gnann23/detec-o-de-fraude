"""
Detecção de Anomalias (Fraudes) em Transações de Cartão de Crédito
====================================================================

Projeto que treina e avalia modelos de Machine Learning para identificar
transações fraudulentas em um dataset fortemente desbalanceado.

Etapas do projeto:
    01. Primeiros passos: carga e análise exploratória dos dados (EDA)
    02. Avaliação e Técnicas de Balanceamento: tratamento do desbalanceamento
        de classes (SMOTE) e métricas apropriadas para dados desbalanceados
    03. Modelos Avançados e Explicabilidade: Random Forest e XGBoost,
        comparação de desempenho e explicabilidade com SHAP

Uso:
    python deteccao_fraude.py --data creditcard.csv

Bibliotecas necessárias:
    pandas, numpy, scikit-learn, imbalanced-learn, xgboost, shap, matplotlib
    (instale com: pip install pandas numpy scikit-learn imbalanced-learn
     xgboost shap matplotlib)
"""

import argparse
import time

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # gera imagens sem precisar de interface gráfica
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_auc_score,
    roc_curve,
    average_precision_score,
    precision_recall_curve,
)

from imblearn.over_sampling import SMOTE

import xgboost as xgb
import shap


# ---------------------------------------------------------------------------
# 01. PRIMEIROS PASSOS NO PROJETO — Carga e Análise Exploratória (EDA)
# ---------------------------------------------------------------------------

def carregar_dados(caminho: str) -> pd.DataFrame:
    """Carrega o dataset de transações a partir de um arquivo CSV."""
    print(f"\n[1/5] Carregando dados de '{caminho}'...")
    df = pd.read_csv(caminho)
    print(f"  -> {df.shape[0]:,} transações e {df.shape[1]} colunas carregadas.")
    return df


def analise_exploratoria(df: pd.DataFrame) -> None:
    """Explora a distribuição das classes (fraude vs. normal) e dados gerais."""
    print("\n[2/5] Análise exploratória dos dados (EDA)")

    print("\nPrimeiras linhas:")
    print(df.head())

    print("\nValores nulos por coluna (top 5, se houver):")
    nulos = df.isnull().sum()
    print(nulos[nulos > 0].head() if nulos.sum() > 0 else "  Nenhum valor nulo encontrado.")

    contagem = df["Class"].value_counts()
    total = len(df)
    fraudes = contagem.get(1, 0)
    normais = contagem.get(0, 0)
    print("\nDistribuição das classes:")
    print(f"  Normais (0): {normais:,} ({normais / total:.4%})")
    print(f"  Fraudes (1): {fraudes:,} ({fraudes / total:.4%})")
    print("  -> Dataset extremamente desbalanceado: métricas como acurácia")
    print("     não são confiáveis aqui. Usaremos precisão, recall, F1 e AUC.")

    # Gráfico de distribuição das classes
    plt.figure(figsize=(5, 4))
    contagem.plot(kind="bar", color=["#3B82F6", "#EF4444"])
    plt.title("Distribuição de Classes (0 = Normal, 1 = Fraude)")
    plt.xlabel("Classe")
    plt.ylabel("Quantidade de transações")
    plt.yscale("log")
    plt.tight_layout()
    plt.savefig("distribuicao_classes.png", dpi=120)
    plt.close()
    print("  Gráfico salvo em 'distribuicao_classes.png'.")


# ---------------------------------------------------------------------------
# 02. AVALIAÇÃO E TÉCNICAS DE BALANCEAMENTO
# ---------------------------------------------------------------------------

def preparar_dados(df: pd.DataFrame):
    """Normaliza colunas sensíveis à escala e separa treino/teste (estratificado)."""
    print("\n[3/5] Pré-processamento e divisão treino/teste")

    df = df.copy()

    # 'Amount' e 'Time' têm escalas muito diferentes das colunas V1..V28
    # (que já vêm padronizadas por PCA), então padronizamos também.
    scaler = StandardScaler()
    df["Amount_scaled"] = scaler.fit_transform(df[["Amount"]])
    df["Time_scaled"] = scaler.fit_transform(df[["Time"]])
    df = df.drop(columns=["Amount", "Time"])

    X = df.drop(columns=["Class"])
    y = df["Class"]

    # stratify=y garante que a proporção de fraudes seja mantida em treino e teste
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    print(f"  Treino: {X_train.shape[0]:,} amostras | Teste: {X_test.shape[0]:,} amostras")
    print(f"  Fraudes no treino: {y_train.sum():,} | Fraudes no teste: {y_test.sum():,}")

    return X_train, X_test, y_train, y_test


def balancear_com_smote(X_train, y_train):
    """Aplica SMOTE (oversampling sintético) apenas no conjunto de treino.

    Importante: o balanceamento NUNCA deve ser aplicado ao conjunto de teste,
    para que a avaliação reflita a distribuição real do mundo (poucas fraudes).
    """
    print("\n[4/5] Balanceamento de classes com SMOTE")
    print(f"  Antes do SMOTE -> Normais: {(y_train == 0).sum():,} | Fraudes: {(y_train == 1).sum():,}")

    # sampling_strategy=0.3 -> a classe minoritária passa a ter 30% do tamanho
    # da majoritária (balanceamento parcial). Evita gerar dados sintéticos
    # demais, o que tornaria o modelo lento e poderia introduzir ruído.
    smote = SMOTE(sampling_strategy=0.3, random_state=42)
    X_res, y_res = smote.fit_resample(X_train, y_train)

    print(f"  Depois do SMOTE -> Normais: {(y_res == 0).sum():,} | Fraudes: {(y_res == 1).sum():,}")
    return X_res, y_res


def avaliar_modelo(nome: str, modelo, X_test, y_test) -> dict:
    """Calcula e imprime as métricas apropriadas para classificação desbalanceada."""
    y_pred = modelo.predict(X_test)
    y_proba = modelo.predict_proba(X_test)[:, 1]

    roc_auc = roc_auc_score(y_test, y_proba)
    pr_auc = average_precision_score(y_test, y_proba)

    print(f"\n--- Avaliação: {nome} ---")
    print(classification_report(y_test, y_pred, target_names=["Normal", "Fraude"], digits=4))
    print(f"  ROC-AUC : {roc_auc:.4f}")
    print(f"  PR-AUC  : {pr_auc:.4f}  (mais informativa que ROC-AUC em dados desbalanceados)")

    # Matriz de confusão
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Normal", "Fraude"])
    fig, ax = plt.subplots(figsize=(4, 4))
    disp.plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title(f"Matriz de Confusão — {nome}")
    plt.tight_layout()
    nome_arquivo = f"matriz_confusao_{nome.lower().replace(' ', '_')}.png"
    plt.savefig(nome_arquivo, dpi=120)
    plt.close()

    return {
        "nome": nome,
        "y_proba": y_proba,
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "modelo": modelo,
    }


def plotar_curvas_comparativas(resultados: list, y_test) -> None:
    """Plota curvas ROC e Precisão-Recall comparando todos os modelos treinados."""
    plt.figure(figsize=(6, 5))
    for r in resultados:
        fpr, tpr, _ = roc_curve(y_test, r["y_proba"])
        plt.plot(fpr, tpr, label=f"{r['nome']} (AUC={r['roc_auc']:.3f})")
    plt.plot([0, 1], [0, 1], "k--", alpha=0.4)
    plt.xlabel("Taxa de Falsos Positivos")
    plt.ylabel("Taxa de Verdadeiros Positivos")
    plt.title("Curva ROC — Comparação de Modelos")
    plt.legend()
    plt.tight_layout()
    plt.savefig("curva_roc.png", dpi=120)
    plt.close()

    plt.figure(figsize=(6, 5))
    for r in resultados:
        prec, rec, _ = precision_recall_curve(y_test, r["y_proba"])
        plt.plot(rec, prec, label=f"{r['nome']} (PR-AUC={r['pr_auc']:.3f})")
    plt.xlabel("Recall")
    plt.ylabel("Precisão")
    plt.title("Curva Precisão-Recall — Comparação de Modelos")
    plt.legend()
    plt.tight_layout()
    plt.savefig("curva_precisao_recall.png", dpi=120)
    plt.close()
    print("\n  Gráficos comparativos salvos: 'curva_roc.png' e 'curva_precisao_recall.png'.")


# ---------------------------------------------------------------------------
# 03. MODELOS AVANÇADOS E EXPLICABILIDADE
# ---------------------------------------------------------------------------

def treinar_modelos(X_train, y_train):
    """Treina um modelo simples (baseline) e dois modelos avançados."""
    modelos = {}

    print("\n  Treinando Regressão Logística (baseline)...")
    inicio = time.time()
    log_reg = LogisticRegression(max_iter=1000, random_state=42)
    log_reg.fit(X_train, y_train)
    modelos["Regressao Logistica"] = log_reg
    print(f"    concluído em {time.time() - inicio:.1f}s")

    print("  Treinando Random Forest...")
    inicio = time.time()
    rf = RandomForestClassifier(
        n_estimators=150, max_depth=12, n_jobs=-1, random_state=42, class_weight="balanced_subsample"
    )
    rf.fit(X_train, y_train)
    modelos["Random Forest"] = rf
    print(f"    concluído em {time.time() - inicio:.1f}s")

    print("  Treinando XGBoost...")
    inicio = time.time()
    xgb_model = xgb.XGBClassifier(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.1,
        eval_metric="aucpr",
        random_state=42,
        n_jobs=-1,
    )
    xgb_model.fit(X_train, y_train)
    modelos["XGBoost"] = xgb_model
    print(f"    concluído em {time.time() - inicio:.1f}s")

    return modelos


def explicar_com_shap(modelo, X_test, nome_modelo: str, amostra: int = 1000) -> None:
    """Gera gráficos de explicabilidade (SHAP) para o modelo treinado.

    Usa uma amostra do conjunto de teste para manter o processamento rápido,
    já que o cálculo de valores SHAP é custoso em datasets grandes.
    """
    print(f"\n[5/5] Explicabilidade (SHAP) para o modelo: {nome_modelo}")

    X_amostra = X_test.sample(n=min(amostra, len(X_test)), random_state=42)

    explainer = shap.TreeExplainer(modelo)
    shap_values = explainer.shap_values(X_amostra)

    # Para modelos com saída binária, alguns retornam uma lista [classe0, classe1]
    valores = shap_values[1] if isinstance(shap_values, list) else shap_values

    plt.figure()
    shap.summary_plot(valores, X_amostra, show=False)
    plt.tight_layout()
    plt.savefig("shap_summary_plot.png", dpi=120, bbox_inches="tight")
    plt.close()
    print("  Gráfico de importância das variáveis salvo em 'shap_summary_plot.png'.")

    # Importância média absoluta de cada variável, em formato de tabela
    importancia = pd.DataFrame({
        "variavel": X_amostra.columns,
        "importancia_media": np.abs(valores).mean(axis=0),
    }).sort_values("importancia_media", ascending=False)

    print("\n  Top 10 variáveis mais relevantes para detectar fraude:")
    print(importancia.head(10).to_string(index=False))
    importancia.to_csv("shap_importancia_variaveis.csv", index=False)
    print("  Tabela completa salva em 'shap_importancia_variaveis.csv'.")


# ---------------------------------------------------------------------------
# EXECUÇÃO PRINCIPAL
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Detecção de fraudes em transações de cartão de crédito.")
    parser.add_argument("--data", type=str, default="creditcard.csv", help="Caminho para o arquivo CSV do dataset.")
    args = parser.parse_args()

    inicio_total = time.time()

    # 01. Primeiros passos
    df = carregar_dados(args.data)
    analise_exploratoria(df)

    # 02. Avaliação e balanceamento
    X_train, X_test, y_train, y_test = preparar_dados(df)
    X_train_bal, y_train_bal = balancear_com_smote(X_train, y_train)

    # 03. Modelos avançados
    modelos = treinar_modelos(X_train_bal, y_train_bal)

    resultados = []
    for nome, modelo in modelos.items():
        resultado = avaliar_modelo(nome, modelo, X_test, y_test)
        resultados.append(resultado)

    plotar_curvas_comparativas(resultados, y_test)

    # Escolhe o melhor modelo pelo PR-AUC (mais robusta para classes raras)
    melhor = max(resultados, key=lambda r: r["pr_auc"])
    print(f"\n>>> Melhor modelo (por PR-AUC): {melhor['nome']} (PR-AUC={melhor['pr_auc']:.4f})")

    # Explicabilidade apenas para modelos baseados em árvore (Random Forest/XGBoost)
    if melhor["nome"] in ("Random Forest", "XGBoost"):
        explicar_com_shap(melhor["modelo"], X_test, melhor["nome"])
    else:
        rf_result = next(r for r in resultados if r["nome"] == "Random Forest")
        explicar_com_shap(rf_result["modelo"], X_test, "Random Forest")

    print(f"\nProjeto concluído em {time.time() - inicio_total:.1f}s.")


if __name__ == "__main__":
    main()
