#!/usr/bin/env python3

from pathlib import Path


# ==========================================================
# TAM-VLM v35
# Automated figure insertion
# ==========================================================


# -----------------------------
# methodology.tex
# -----------------------------

method_file = Path("sections/methodology.tex")

text = method_file.read_text()


fig1 = r"""

\begin{figure}[t]
    \centering
    \includegraphics[width=\linewidth]{figures/fig1_framework_v35.pdf}
    \caption{Overview of TAM-VLM for physical backdoor trigger detection in driving vision-language models. The framework uses a frozen visual encoder to extract image representations and a lightweight detector head to estimate trigger evidence. A validation-only threshold is applied during inference-time screening.}
    \label{fig:tamvlm_framework}
\end{figure}

"""


fig2 = r"""

\begin{figure}[t]
    \centering
    \includegraphics[width=\linewidth]{figures/fig2_runtime_pipeline_v35.pdf}
    \caption{Inference-time screening pipeline of TAM-VLM. An input driving image is mapped into a frozen visual representation, processed by the lightweight detector head, and classified using a validation-derived decision threshold.}
    \label{fig:runtime_pipeline}
\end{figure}

"""


marker = r"\subsection{Frozen Visual Representation Extraction}"


if "fig1_framework_v35.pdf" not in text:

    text = text.replace(
        marker,
        fig1 + fig2 + "\n" + marker
    )

    method_file.write_text(text)



# -----------------------------
# experiments.tex
# -----------------------------


exp_file = Path("sections/experiments.tex")

text = exp_file.read_text()


fig3 = r"""

\begin{figure}[t]
    \centering
    \includegraphics[width=\linewidth]{figures/fig3_benchmark_protocol_v35.pdf}
    \caption{Leakage-safe benchmark construction and evaluation protocol. Scene-grouped partitioning prevents source overlap between training and evaluation. The protocol includes controlled evaluation, held-out-trigger generalization, and fixed-budget trigger diversity analysis.}
    \label{fig:benchmark_protocol}
\end{figure}

"""


fig4 = r"""

\begin{figure}[t]
    \centering
    \includegraphics[width=\linewidth]{figures/fig4_trigger_examples_v35.pdf}
    \caption{Examples of generator-matched trigger variants. Source keyframes are transformed into trigger-edited samples and matched null controls using the same image-editing pipeline. Visualization annotations are used only for illustration and are not provided to TAM-VLM.}
    \label{fig:trigger_examples}
\end{figure}

"""


marker = r"\subsection{Data Splits and Leakage Control}"


if "fig3_benchmark_protocol_v35.pdf" not in text:

    text = text.replace(
        marker,
        fig3 + fig4 + "\n" + marker
    )

    exp_file.write_text(text)



# -----------------------------
# results.tex
# -----------------------------


res_file = Path("sections/results.tex")

text = res_file.read_text()



figures = {


r"\subsection{Strict Leave-One-Trigger-Out Generalization}":

r"""
\begin{figure}[t]
    \centering
    \includegraphics[width=\linewidth]{figures/fig5_baseline_comparison_v35.pdf}
    \caption{Comparison of TAM-VLM with representation-based and anomaly-detection baselines under the controlled evaluation protocol.}
    \label{fig:baseline_comparison}
\end{figure}

""",


r"\subsection{Fixed-Budget Trigger-Diversity Scaling}":

r"""
\begin{figure}[t]
    \centering
    \includegraphics[width=\linewidth]{figures/fig6_loto_generalization_v35.pdf}
    \caption{Leave-one-trigger-out (LOTO) generalization across held-out physical trigger categories. Each category is excluded from detector development and evaluated only during testing.}
    \label{fig:loto_generalization}
\end{figure}

""",


r"\subsection{Deployment-Oriented Hard-Negative and Temporal Evaluation}":

r"""
\begin{figure}[t]
    \centering
    \includegraphics[width=\linewidth]{figures/fig7_diversity_audit_v35.pdf}
    \caption{Effect of trigger-category diversity on held-out generalization under a fixed positive-sample budget.}
    \label{fig:diversity_audit}
\end{figure}

""",


r"\subsection{Synthetic-Degradation Robustness}":

r"""
\begin{figure}[t]
    \centering
    \includegraphics[width=\linewidth]{figures/fig8_runtime_robustness_v35.pdf}
    \caption{Runtime robustness evaluation of TAM-VLM under hard-negative conditions and temporal confirmation strategies.}
    \label{fig:runtime_robustness}
\end{figure}

""",


r"\subsection{Data Efficiency}":

r"""
\begin{figure}[t]
    \centering
    \includegraphics[width=\linewidth]{figures/fig10_data_efficiency_ablation_v35.pdf}
    \caption{Effect of positive supervision availability on detection performance. The analysis characterizes the relationship between annotation budget and detector performance.}
    \label{fig:data_efficiency}
\end{figure}

""",


r"\subsection{Exact ROC, Precision--Recall, and Low-FPR Diagnostics}":

r"""
\begin{figure}[t]
    \centering
    \includegraphics[width=\linewidth]{figures/fig9_component_ablation_v35.pdf}
    \caption{Component-level sensitivity analysis of TAM-VLM. The ablation study evaluates the contribution of detector design choices and validates the final architecture.}
    \label{fig:component_ablation}
\end{figure}

"""
}



for marker, fig in figures.items():

    if "fig" not in text:

        text = text.replace(
            marker,
            fig + "\n" + marker
        )


res_file.write_text(text)



print("===================================")
print("TAM-VLM v35 figures inserted")
print("===================================")
