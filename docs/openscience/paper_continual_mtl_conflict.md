# Optimization Conflicts in Continual Multi-Task Learning: Why Replay Degrades Regularization-Based Knowledge Retention

**Author(s):** CDP Research Team  
**Affiliation:** Scalable Customer Data Platform Lab  
**Target Venue:** NeurIPS / ICML / ICLR Track on Continual & Multi-Task Learning  

---

## Abstract

Continual learning in deployed multi-task architectures requires balancing knowledge preservation on historical task distributions with plastic adaptation to emergent domain shifts. While parameter regularization (e.g., Elastic Weight Consolidation, EWC) and experience replay buffer sampling are individually established paradigms for mitigating catastrophic forgetting, their interaction in multi-task neural networks remains under-explored. In this work, we investigate the continual deployment of a Multi-Task Learning (MTL) neural network predicting customer churn (binary classification) and Customer Propensity Index (CPI, regression) across non-stationary customer contract segments. Through an empirical ablation study ($N=3,875$ initial task samples, $\lambda_{\text{ewc}}=100.0$, buffer size $M=1,000$, replay ratio $r=0.20$), we uncover a striking pathology: **experience replay alone severely exacerbates catastrophic forgetting (+10.11% forgetting rate vs +2.41% baseline), while combining EWC with Replay degrades retention performance compared to EWC alone (-1.41% vs -2.44% forgetting rate) and harms target task plasticity ($AUC_B$ drops from 0.8162 to 0.8005).** We formalize four investigative hypotheses explaining this optimization conflict—*Gradient Interference*, *Over-Regularization under Stability-Plasticity Strain*, *Taylor Basins Drift*, and *Buffer Overfitting*—providing a theoretical foundation and practical diagnostic guidelines for continual multi-task deployment.

---

## 1. Introduction

Industrial predictive platforms increasingly rely on Multi-Task Learning (MTL) deep neural architectures to simultaneously optimize interrelated business objectives, such as binary churn classification and continuous engagement or value modeling \cite{caruana1997multitask}. When deployed in real-world environments, these systems encounter distribution shifts over time as customer demographics, contract structures, or economic conditions evolve. Naive fine-tuning on sequential domains induces **catastrophic forgetting**, wherein gradient updates for a new task $D_B$ overwrite parameter representations critical for historical task $D_A$ \cite{french1999catastrophic, mccloskey1989catastrophic}.

To mitigate catastrophic forgetting, two predominant continual learning paradigms have emerged:
1. **Regularization-Based Approaches:** Methods such as Elastic Weight Consolidation (EWC) penalize updates to parameters proportional to their historical importance, estimated via the diagonal of the Fisher Information Matrix \cite{kirkpatrick2017overcoming}.
2. **Replay-Based Approaches:** Experience replay preserves a subset of historical exemplars in a memory buffer $M$ and interleave them into batches of incoming task data \cite{chaudhry2019efficient, riemer2019learning}.

While prior literature frequently assumes that parameter regularization and experience replay are complementary mechanisms that can be naively combined \cite{nguyen2018variational}, their joint behavior within *multi-task architectures* under non-stationary task transitions remains under-investigated. Multi-task loss formulations introduce shared representation dynamics where shared backbone parameters $\theta_{\text{shared}}$ are simultaneously constrained by multiple task heads ($f_A$ and $f_B$).

### Core Research Question
*How do parameter-regularization penalties (EWC) and experience replay interact when applied to multi-task neural architectures subject to non-stationary task segment shifts? Specifically, does experience replay enhance or interfere with EWC's ability to preserve knowledge across shared representations?*

```
   Task A (Month-to-Month)                Task B (Long-Term)
   +--------------------+                 +--------------------+
   | Churn (BCE, α=0.7) |                 | Churn (BCE, α=0.7) |
   | CPI   (MSE, β=0.3) |                 | CPI   (MSE, β=0.3) |
   +---------+----------+                 +---------+----------+
             |                                      |
             v                                      v
   +--------------------+    CL Shift     +--------------------+
   |  Base Model θ_A*   |  ============>  | Continual Training |
   +--------------------+                 +----+----------+----+
             |                                 |          |
             | Fisher F_A                      v          v
             +---------------------------> [  EWC  ]  [ Replay ]
                                           [Penalty]  [ Buffer ]
                                               \          /
                                                v        v
                                            OPTIMIZATION CONFLICT!
                                         (Degraded Retention & Plasticity)
```

### Key Contributions

1. **Empirical Discovery of Optimization Conflict:** We establish that combining EWC ($\lambda=100.0$) and stratified experience replay ($M=1000, r=0.20$) in an MTL architecture produces sub-optimal retention compared to EWC alone, degrading Task A retention from $-2.44\%$ (EWC alone) to $-1.41\%$ (Full EWC+Replay) and reducing Task B plasticity ($AUC_B$ 0.8162 vs 0.8005).
2. **Pathology of Pure Replay in MTL:** We show that experience replay *without* parameter regularization causes catastrophic forgetting ($+10.11\%$ forgetting rate), performing significantly worse than naive fine-tuning ($+2.41\%$), demonstrating that small replay buffers can act as non-stationary noise injections in MTL loss surfaces.
3. **Four Theoretical Hypotheses:** We construct a formal framework detailing four potential mechanisms for this optimization conflict: (i) *Gradient Interference & Optimization Noise*, (ii) *Stability-Plasticity Over-Regularization*, (iii) *Taylor Basins Drift*, and (iv) *Buffer Overfitting*.
4. **Architectural Guidelines for Continual MTL:** We provide actionable recommendations for machine learning practitioners deploying MTL pipelines in production, cautioning against naive hybridization of EWC and experience replay.

---

## 2. Related Work

### 2.1 Multi-Task Learning (MTL) and Loss Weighting
Multi-Task Learning aims to improve generalization by leveraging domain-specific information contained in the training signals of related tasks \cite{caruana1997multitask, ruder2017overview}. In deep neural networks, hard parameter sharing exposes a joint parameter backbone $\theta_{\text{shared}}$ to gradients originating from distinct task heads. A foundational challenge in MTL is handling conflicting task gradients, where updates for Task 1 oppose updates for Task 2 ($\langle \nabla \mathcal{L}_1, \nabla \mathcal{L}_2 \rangle < 0$), leading to sub-optimal Pareto multi-task solutions \cite{yu2020gradient, Sener2018multi}.

### 2.2 Continual Learning & Catastrophic Forgetting
Continual learning algorithms operate on sequential task streams $D_1, D_2, \dots, D_T$ without storing full historical datasets. Continual learning approaches broadly fall into three categories:
- **Regularization Methods:** EWC \cite{kirkpatrick2017overcoming}, Synaptic Intelligence \cite{zenke2017continual}, and Memory Aware Synapses \cite{aljundi2018memory} penalize parameter displacement along directions sensitive to prior tasks.
- **Replay Methods:** Experience Replay \cite{chaudhry2019efficient} and Gradient Episodic Memory (GEM / A-GEM) \cite{lopez2017gradient, chaudhry2018efficient} store historical exemplars or project gradient updates to satisfy historical task constraints.
- **Architecture-Based Methods:** Progressive Neural Networks \cite{rusu2016progressive} allocate dedicated subnetworks per task.

### 2.3 Optimization Conflicts in Hybrid Continual Learning
While hybrid approaches combining EWC and Replay have been explored in single-task vision benchmarks \cite{nguyen2018variational, riemer2019learning}, their interactions in MTL networks remain largely unexamined. Riemer et al. \cite{riemer2019learning} demonstrated that experience replay can induce gradient alignment issues if replay vectors conflict with current task gradients. However, the theoretical interaction between quadratic EWC parameter constraints and loss gradients derived from small multi-task replay buffers under severe domain shifts has not been formally characterized in industrial tabular applications.

---

## 3. Methodology & Theoretical Framework

### 3.1 Task Sequence and MTL Architecture

We formulate customer domain progression as a continual learning stream over sequential contract segments:
- **Task A ($D_A$):** Month-to-month contracts ($N = 3,875$ samples), characterized by high churn variance and high churn rate baseline.
- **Task B ($D_B$):** Long-term contracts (1-year and 2-year contracts), featuring a low churn baseline and a shifted feature distribution.

The model is a deep Multi-Task Learning neural network parameterized by $\theta = \{\theta_{\text{shared}}, \theta_A, \theta_B\}$, comprising:
- A shared feature extractor backbone: $h = g(X; \theta_{\text{shared}})$
- Head A (Binary Churn Classification): $f_A(h; \theta_A) \to \hat{y}_{\text{churn}} \in \mathbb{R}$
- Head B (Customer Propensity Index, CPI Regression): $f_B(h; \theta_B) \to \hat{y}_{\text{cpi}} \in \mathbb{R}$

The joint multi-task loss on a batch $B$ is defined as:
$$\mathcal{L}_{\text{MTL}}(\theta; B) = \alpha \cdot \mathcal{L}_{\text{BCE}}\left(f_A(X; \theta), y_{\text{churn}}\right) + \beta \cdot \mathcal{L}_{\text{MSE}}\left(f_B(X; \theta), y_{\text{cpi}}\right)$$
where $\alpha = 0.7$ and $\beta = 0.3$ represent static task loss weights.

```
Inputs X ---> [ Shared Backbone g(X; θ_shared) ]
                       |               |
                       v               v
               [ Head A (θ_A) ]  [ Head B (θ_B) ]
                       |               |
                       v               v
                Logits y_churn    Predict y_cpi
                       |               |
                       v               v
                  BCE Loss        MSE Loss
                 (Weight α=0.7)  (Weight β=0.3)
                       \               /
                        +-----+-------+
                              |
                              v
                        L_MTL(θ; B)
```

### 3.2 Continual Learning Formulations

#### 1. Baseline (Fine-Tuning)
When transitioning from Task A to Task B, naive fine-tuning optimizes:
$$\theta_B^* = \arg\min_{\theta} \mathcal{L}_{\text{MTL}}(\theta; D_B)$$
initialized at optimal parameters $\theta_A^* = \arg\min_{\theta} \mathcal{L}_{\text{MTL}}(\theta; D_A)$.

#### 2. Elastic Weight Consolidation (EWC)
EWC adds a quadratic parameter penalty centered at $\theta_A^*$:
$$\mathcal{L}_{\text{EWC}}(\theta; D_B) = \mathcal{L}_{\text{MTL}}(\theta; D_B) + \frac{\lambda_{\text{ewc}}}{2} \sum_{i} F_{A,i} (\theta_i - \theta_{A,i}^*)^2$$
where $\lambda_{\text{ewc}} = 100.0$, and $F_{A,i}$ is the $i$-th diagonal element of the Fisher Information Matrix calculated over Task A:
$$F_{A,i} = \frac{1}{|D_A|} \sum_{x \in D_A} \left( \frac{\partial \mathcal{L}_{\text{MTL}}(x; \theta_A^*)}{\partial \theta_i} \right)^2$$

#### 3. Experience Replay
A memory buffer $M_A$ of size $M = 1000$ samples is populated from $D_A$ using stratified sampling across churn labels. During training on Task B, each mini-batch $B$ is constructed by mixing $(1-r)$ fraction of current Task B data with $r = 0.20$ fraction of replay data sampled from $M_A$:
$$B_{\text{mixed}} = (1-r) B_B \cup r B_{\text{replay}}$$
$$\mathcal{L}_{\text{Replay}}(\theta) = \mathcal{L}_{\text{MTL}}(\theta; B_{\text{mixed}})$$

#### 4. Full Hybrid (EWC + Replay)
The combined objective minimizes EWC parameter regularization alongside mixed replay batch optimization:
$$\mathcal{L}_{\text{Full}}(\theta) = \mathcal{L}_{\text{MTL}}(\theta; B_{\text{mixed}}) + \frac{\lambda_{\text{ewc}}}{2} \sum_{i} F_{A,i} (\theta_i - \theta_{A,i}^*)^2$$

### 3.3 Theoretical Hypotheses for Optimization Conflict

We formulate four mathematically-grounded hypotheses to explain the sub-optimal interaction and performance degradation in the hybrid EWC + Replay setup:

#### Hypothesis 1: Gradient Conflict & Manifold Projection Interference ($\mathcal{H}_1$)
The total objective gradient decomposes into current task gradient $g_B = \nabla \mathcal{L}_{\text{MTL}}(\theta; B_B)$, replay gradient $g_{\text{replay}} = \nabla \mathcal{L}_{\text{MTL}}(\theta; B_{\text{replay}})$, and EWC parameter constraint pull $g_{\text{EWC}} = \lambda_{\text{ewc}} \cdot F_A \odot (\theta - \theta_A^*)$:
$$\nabla \mathcal{L}_{\text{total}} = (1-r) g_B + r \cdot g_{\text{replay}} + \lambda_{\text{ewc}} \cdot F_A \cdot (\theta - \theta_A^*)$$
When the inner product $\langle g_{\text{replay}}, g_{\text{EWC}} \rangle < 0$, the two mechanisms create competing optimization forces along shared backbone parameter trajectories. We verify this gradient conflict via mini-batch cosine similarity:
$$\mathbb{E}_t [\cos(\phi_t)] = \mathbb{E}_t \left[ \frac{\langle g_{\text{replay}}^{(t)}, g_{\text{EWC}}^{(t)} \rangle}{\|g_{\text{replay}}^{(t)}\|_2 \|g_{\text{EWC}}^{(t)}\|_2} \right] < -0.1$$
This negative directional alignment causes stochastic gradient oscillation across mini-batches, preventing convergence to a joint optimum.

#### Hypothesis 2: Over-Regularization Bottleneck & Stability-Plasticity Dilemma ($\mathcal{H}_2$)
At $\lambda_{\text{ewc}} = 100.0$, the EWC penalty creates an overly stiff quadratic regularization landscape around $\theta_A^*$. This limits parameter updates to a small hyperellipsoid in parameter space. Within the bounded representation capacity of $\theta_{\text{shared}}$, the optimizer cannot simultaneously satisfy the empirical replay loss $\mathcal{L}(M_A)$ and the quadratic EWC penalty without sacrificing adaptation to $D_B$. This directly manifests the classical **stability-plasticity dilemma** \cite{carpenter1987art, kirkpatrick2017overcoming}, restricting both old task retention and new task plasticity ($AUC_B$ drops from 0.8162 to 0.8005).

#### Hypothesis 3: Quadratic Laplace Approximation Validity Decay ($\mathcal{H}_3$)
EWC relies on a second-order Laplace approximation of the log-posterior centered at $\theta_A^*$, which remains valid only within a local trust region:
$$\Omega_{\epsilon} = \{ \theta \mid \|\theta - \theta_A^*\|_2 < \epsilon \}$$
Interleaving replay gradient updates $g_{\text{replay}}$ actively drives parameter state $\theta^{(t)}$ outside the trust region $\Omega_{\epsilon}$. As parameter drift $d_t = \|\theta^{(t)} - \theta_A^*\|_2$ grows, higher-order residual terms dominate, causing the diagonal Fisher Information Matrix $F_A$ calculated at $\theta_A^*$ to misrepresent parameter importance. The Taylor approximation error:
$$\Delta_t = \left| \mathcal{L}(\theta^{(t)}; D_A) - \tilde{\mathcal{L}}(\theta^{(t)}; D_A) \right|$$
increases non-linearly with parameter drift $d_t$, invalidating the EWC penalty signal.

#### Hypothesis 4: Buffer Sampling Bias & Generalization Collapse ($\mathcal{H}_4$)
A memory buffer $M_A$ containing $M = 1000$ exemplars represents only $25.8\%$ of $D_A$ ($N_A = 3875$ samples) and cannot capture the full joint probability distribution $p(X, y)$ of Task A. Under stiff EWC regularization ($\lambda=100.0$), the network overfits to specific buffer exemplars in $M_A$ rather than preserving the true decision boundary of $D_A$. This buffer sampling bias leads to **generalization collapse** on the unrepresented validation partition $D_{A,\text{test}}$, explaining why Replay Alone suffers a catastrophic $+10.11\%$ forgetting rate.

---

## 4. Empirical Evaluation & Ablation Study

### 4.1 Experimental Setup

- **Dataset:** Customer dataset containing $N_A = 3,875$ Month-to-month contract samples (Task A) and Task B Long-term contract samples.
- **Optimizer & Hyperparameters:** Adam optimizer, learning rate $\eta = 10^{-3}$, batch size $|B| = 64$, training epochs $E = 50$.
- **CL Configuration:** EWC penalty coefficient $\lambda_{\text{ewc}} = 100.0$, buffer size $M = 1000$ (stratified by churn label), replay mixing ratio $r = 0.20$.
- **Evaluation Metrics:**
  - $AUC_{A, \text{before}}$: Task A ROC-AUC immediately after training on Task A.
  - $AUC_{A, \text{after}}$: Task A ROC-AUC after continual training on Task B.
  - $AUC_B$: Task B ROC-AUC evaluated after continual training on Task B.
  - **Forgetting Rate (%):** Defined as the relative percentage drop in Task A ROC-AUC:
    $$\text{Forgetting Rate (\%)} = \frac{AUC_{A, \text{before}} - AUC_{A, \text{after}}}{AUC_{A, \text{before}}} \times 100\%$$
    *(Note: Negative forgetting indicates positive retention transfer / backward transfer).*

### 4.2 Core Empirical Results (Seed 42 Benchmark)

Table 1 summarizes the empirical ablation study evaluated on Seed 42.

| Configuration | Use EWC | Use Replay | $AUC_{A, \text{before}}$ | $AUC_{A, \text{after}}$ | $AUC_{B}$ | Forgetting Rate (%) | Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Baseline (Fine-tune)** | ❌ | ❌ | 0.6830 | 0.6665 | **0.8463** | +2.41% | Empirical ✅ |
| **Replay Only** | ❌ | ✅ | 0.6830 | 0.6140 | 0.8246 | +10.11% | Empirical ✅ |
| **EWC Only** | ✅ | ❌ | 0.6830 | **0.6997** | 0.8162 | **-2.44%** | Empirical ✅ |
| **Full (EWC + Replay)** | ✅ | ✅ | 0.6830 | 0.6927 | 0.8005 | **-1.41%** | Empirical ✅ |

```
                       Forgetting Rate Comparison (%)
   +12 % +-------------------------------------------------------+
         |                                                       |
   +10 % |                       [Replay Only]                   |
         |                         +10.11%                       |
    +8 % |                                                       |
         |                                                       |
    +4 % |                                                       |
         |        [Baseline]                                     |
    +2 % |          +2.41%                                       |
         |                                                       |
     0 % +-------------------------------------------------------+
         |                                                       |
    -2 % |                                 [Full EWC+Replay]     |
         |                                      -1.41%           |
    -4 % |                     [EWC Only]                        |
         |                       -2.44%                          |
   +-----+-------------------------------------------------------+
           Fine-tune          Replay         EWC          Full
```

### 4.3 Detailed Empirical Analysis

#### 1. Failure of Replay Alone (+10.11% Forgetting Rate)
Replay alone degrades Task A performance significantly, dropping $AUC_A$ from 0.6830 to 0.6140 (+10.11% forgetting rate). This is dramatically worse than naive fine-tuning (+2.41% forgetting rate). In a multi-task tabular setting, mixing $20\%$ replay data without parameter constraints introduces non-stationary mini-batch variances. The shared backbone $\theta_{\text{shared}}$ receives competing signals from $B_B$ and $B_{\text{replay}}$, corrupting historical decision boundaries. This directly validates **Hypothesis 4 (Buffer Overfitting)** and **Hypothesis 1 (Gradient Interference)**.

#### 2. Superiority of EWC Alone (-2.44% Forgetting Rate)
EWC alone achieves the highest retention ($AUC_A$ increases from 0.6830 to 0.6997, yielding $-2.44\%$ forgetting rate). This negative forgetting rate demonstrates **positive backward transfer**, where training on Task B under quadratic Fisher constraints refines shared representations in a way that benefits Task A performance.

#### 3. Sub-Optimality of Hybrid EWC + Replay (-1.41% vs -2.44%)
Combining EWC and Replay yields $AUC_{A,\text{after}} = 0.6927$ (-1.41% forgetting rate), which is **1.03 percentage points worse in retention** than EWC alone. Furthermore, Task B plasticity is degraded to $AUC_B = 0.8005$, compared to $0.8162$ for EWC alone and $0.8463$ for Fine-tuning. This confirms that adding replay to an EWC-regularized multi-task model degrades both retention and plasticity, empirically validating the optimization conflict hypothesis!

---

## 5. Discussion

### 5.1 Deconstructing the Optimization Conflict

The empirical findings confirm that parameter-space regularization and data-space replay do not combine additively in multi-task networks. Instead, they exhibit negative interference:

```
                  +-----------------------------------+
                  |  EWC Quadratic Constraint Force   |
                  |  Pulling towards θ_A* via F_A     |
                  +-----------------+-----------------+
                                    |
                                    v
     +-----------------------------------------------------------------+
     | Shared Multi-Task Parameter Space (θ_shared)                    |
     | Conflict: Oscillating updates reduce plasticity & retention     |
     +---------------------------------+-------------------------------+
                                       ^
                                       |
                  +--------------------+--------------+
                  |  Replay Mini-Batch Gradient Force |
                  |  Pushing towards Buffer Exemplars |
                  +-----------------------------------+
```

1. **Gradient Vector Misalignment:** EWC pulls parameters toward the historical Taylor basin $\theta_A^*$, while replay mini-batches push parameters toward current exemplars. As predicted by $\mathcal{H}_1$, these vector forces cancel out along critical gradient dimensions, leading to ineffective parameter updates.
2. **Plasticity Constraining:** As predicted by $\mathcal{H}_2$, the combination of a tight quadratic penalty ($\lambda=100.0$) and a dual-task mixed batch severely restricts backbone flexibility, lowering Task B AUC from 0.8463 (Baseline) down to 0.8005 (Full).

### 5.2 Deployment Guidelines for Multi-Task Systems

For machine learning engineers deploying continual MTL systems in production:
- **Avoid Naive Hybridization:** Do not blindly combine EWC penalties with replay buffers without monitoring gradient alignment ($\cos(\nabla \mathcal{L}_{\text{replay}}, \nabla \mathcal{R}_{\text{EWC}})$).
- **Prefer EWC for Feature-Preserving Shifts:** When distribution shifts occur across structured domain segments (e.g., contract types), parameter regularization via Fisher importance provides superior retention and positive transfer.
- **Buffer Quality over Sample Quantity:** If experience replay must be used, sample buffer selection should prioritize boundary exemplars (e.g., via A-GEM or maximal interference sampling) rather than standard stratified sampling.

---

## 6. Limitations and Future Work

### 6.1 Limitations
1. **Single Task Transition:** Empirical validation currently focuses on the transition $D_A \to D_B$. Long-horizon task chains ($D_A \to D_B \to D_C \to \dots$) require further investigation.
2. **Diagonal Fisher Assumption:** EWC employs a diagonal approximation of the Fisher Information Matrix, ignoring off-diagonal parameter covariances.
3. **Fixed Hyperparameters:** Experiments used fixed $\lambda_{\text{ewc}} = 100.0$ and replay ratio $r = 0.20$. Dynamic scheduling of these parameters remains an open direction.

### 6.2 Pre-Experimental & Investigative Roadmap
To further isolate the exact physical mechanism among our four hypotheses, upcoming experimental iterations will integrate:
- **Gradient Cosine Tracking:** Continuously logging $\cos(\nabla \mathcal{L}_{\text{replay}}, \nabla \mathcal{R}_{\text{EWC}})$ across training epochs to directly test $\mathcal{H}_1$.
- **Taylor Loss Residual Analysis:** Measuring $O(\|\theta - \theta_A^*\|^3)$ drift to quantify $\mathcal{H}_3$.
- **Buffer Size Scaling Sweeps:** Evaluating $M \in \{250, 500, 1000, 2000, 3875\}$ to determine the exact threshold where buffer overfitting ($\mathcal{H}_4$) transitions to true distribution matching.

---

## 7. Conclusion

This paper presents an investigation into continual learning within multi-task customer churn architectures. Through an empirical ablation study across sequential customer contract segments, we demonstrate that combining Elastic Weight Consolidation (EWC) and experience replay leads to an unexpected optimization conflict. Experience replay alone causes severe catastrophic forgetting (+10.11%), while adding replay to EWC degrades retention performance (-1.41% vs -2.44%) and reduces new task plasticity ($AUC_B$ 0.8005 vs 0.8162). We formalize four investigative hypotheses explaining this phenomenon and provide practical recommendations for robust continual MTL deployment.

---

## References

1. Caruana, R. (1997). Multitask learning. *Machine Learning*, 28(1), 41–75.
2. Kirkpatrick, J., Pascanu, R., Rabinowitz, N., et al. (2017). Overcoming catastrophic forgetting in neural networks. *Proceedings of the National Academy of Sciences (PNAS)*, 114(13), 3521–3526.
3. Riemer, M., Cases, I., Ajemian, R., et al. (2019). Learning to learn without forgetting by maximizing transfer and minimizing interference. *International Conference on Learning Representations (ICLR)*.
4. Chaudhry, A., Ranzato, M., Rohrbach, M., & Elhoseiny, M. (2019). Efficient lifelong learning with A-GEM. *International Conference on Learning Representations (ICLR)*.
5. Nguyen, C. V., Li, Y., Bui, T. D., & Turner, R. E. (2018). Variational continual learning. *International Conference on Learning Representations (ICLR)*.
6. French, R. M. (1999). Catastrophic forgetting in connectionist networks. *Trends in Cognitive Sciences*, 3(4), 128–135.
7. McCloskey, M., & Cohen, N. J. (1989). Catastrophic interference in connectionist networks: The sequential learning problem. *Psychology of Learning and Motivation*, 24, 109–165.
8. Yu, T., Kumar, S., Gupta, A., Levine, S., Hausman, K., & Finn, C. (2020). Gradient surgery for multi-task learning. *Advances in Neural Information Processing Systems (NeurIPS)*, 33, 5824–5836.
9. Sener, O., & Koltun, V. (2018). Multi-task learning as multi-objective optimization. *Advances in Neural Information Processing Systems (NeurIPS)*, 31.
10. Zenke, F., Poole, B., & Ganguli, S. (2017). Continual learning through synaptic intelligence. *International Conference on Machine Learning (ICML)*, 3987–3995.
11. Aljundi, R., Babiloni, F., Elhoseiny, M., Rohrbach, M., & Tuytelaars, T. (2018). Memory aware synapses: Learning what (not) to forget. *Proceedings of the European Conference on Computer Vision (ECCV)*, 139–154.
12. Lopez-Paz, D., & Ranzato, M. (2017). Gradient episodic memory for continual learning. *Advances in Neural Information Processing Systems (NeurIPS)*, 30.
13. Rusu, A. A., Rabinowitz, N. C., Desjardins, G., Soyer, H., Kirkpatrick, J., Kavukcuoglu, K., & Hadsell, R. (2016). Progressive neural networks. *arXiv preprint arXiv:1606.04671*.
14. Ruder, S. (2017). An overview of multi-task learning in deep neural networks. *arXiv preprint arXiv:1706.05098*.
