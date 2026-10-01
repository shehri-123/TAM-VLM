from pathlib import Path

file = Path("sections/results.tex")

text = file.read_text()


insertions = [

(
r"\subsection{Fixed-Budget Trigger-Diversity Scaling}",

r"""
\begin{figure}[t]
    \centering
    \includegraphics[width=\linewidth]{figures/fig6_loto_generalization_v35.pdf}
    \caption{Leave-one-trigger-out (LOTO) generalization across held-out physical trigger categories. Each category is excluded from detector development and evaluated only during testing.}
    \label{fig:loto_generalization}
\end{figure}

"""
),


(
r"\subsection{Deployment-Oriented Hard-Negative and Temporal Evaluation}",

r"""
\begin{figure}[t]
    \centering
    \includegraphics[width=\linewidth]{figures/fig7_diversity_audit_v35.pdf}
    \caption{Effect of trigger-category diversity on held-out generalization under a fixed positive-sample budget.}
    \label{fig:diversity_audit}
\end{figure}

"""
),


(
r"\subsection{Synthetic-Degradation Robustness}",

r"""
\begin{figure}[t]
    \centering
    \includegraphics[width=\linewidth]{figures/fig8_runtime_robustness_v35.pdf}
    \caption{Runtime robustness evaluation of TAM-VLM under hard-negative conditions and temporal confirmation strategies.}
    \label{fig:runtime_robustness}
\end{figure}

"""
),


(
r"\subsection{Dedicated Ablation Evidence}",

r"""
\begin{figure}[t]
    \centering
    \includegraphics[width=\linewidth]{figures/fig9_component_ablation_v35.pdf}
    \caption{Component-level sensitivity analysis of TAM-VLM. The ablation study evaluates the contribution of detector design choices and validates the final architecture.}
    \label{fig:component_ablation}
\end{figure}

"""
),


(
r"\subsection{Exact ROC, Precision--Recall, and Low-FPR Diagnostics}",

r"""
\begin{figure}[t]
    \centering
    \includegraphics[width=\linewidth]{figures/fig10_data_efficiency_ablation_v35.pdf}
    \caption{Effect of positive supervision availability on detection performance. The analysis characterizes the relationship between annotation budget and detector performance.}
    \label{fig:data_efficiency}
\end{figure}

"""
)

]


for marker, figure in insertions:

    if figure.split("{figures/")[1].split("}")[0] not in text:

        text = text.replace(
            marker,
            figure + "\n" + marker
        )

        print("Inserted:", marker)

    else:
        print("Already exists:", marker)


file.write_text(text)

print("\nRemaining results figures completed.")
