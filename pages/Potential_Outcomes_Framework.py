import streamlit as st
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns


from scipy.stats import bernoulli
from navigation import make_sidebar, make_footer


def highlight_treatments(data):
    """
    Highlight the treatment cells. Note that this is not shown in the
    code blocks.
    """

    # Empty DF with no styling
    df_styled = pd.DataFrame("", index=data.index, columns=data.columns)

    treated_mask = data["T"] == 1
    untreated_mask = data["T"] == 0
    treated_format = "color: green;"
    untreated_format = "color: blue;"

    df_styled.loc[treated_mask, "T"] = treated_format
    df_styled.loc[untreated_mask, "T"] = untreated_format

    return df_styled


def highlight_potential_outcomes(data):
    """
    Highlight the counterfactual cells. Note that this is not shown in the
    code blocks.
    """

    # Empty DF with no styling
    df_styled = pd.DataFrame("", index=data.index, columns=data.columns)

    for T in (0, 1):
        factual_mask = (data["T"] == T) & (data["Y"] == data[f"Y{T}"])
        counterfactual_mask = (data["T"] != T) & (data["Y"] != data[f"Y{T}"])
        if T == 1:
            factual_format = "background-color: rgba(0, 128, 50, 0.85);"
            counterfactual_format = "background-color: rgba(0, 128, 50, 0.25);"
        else:
            factual_format = "background-color: rgba(0, 100, 256, 0.85);"
            counterfactual_format = "background-color: rgba(0, 100, 256, 0.25);"

        df_styled.loc[factual_mask, f"Y{T}"] = factual_format
        df_styled.loc[counterfactual_mask, f"Y{T}"] = counterfactual_format

    return df_styled


def format_causal_data(causal_data, formatters):
    styled_df = causal_data.style.apply(formatters[0], axis=None)
    if len(formatters) > 1:
        for formatter in formatters[1:]:
            styled_df = styled_df.apply(formatter, axis=None)

    return styled_df.format(precision=0)


def display_causal_data(causal_data, formatters):
    formatters = [formatters] if not isinstance(formatters, list) else formatters
    st.dataframe(format_causal_data(causal_data, formatters))


FORMATTERS = [highlight_treatments]

make_sidebar()

SHOW_CODE = st.sidebar.toggle("Hide/Show Python Code", value=True)


# TODO: Making these TOCs by hand does not scale well
st.sidebar.markdown(
    """
[Observed Outcomes and Treatments](#observed-outcomes-and-treatments) \\
[Potential Outcomes](#potential-outcomes) \\
[Individual Treatment Effect](#individual-treatment-effect-te) \\
[Treatment Effect on the Treated](#treatment-effect-on-treated-tet) \\
[Average Treatment Effect](#average-treatment-effect-ate) \\
[Average Treatment Effect on the Treated](#average-treatment-effect-on-the-treated-att) \\
[Causation vs Association](#causation-vs-association) \\
a. [Bias](#bias-b) \\
b. [Association](#association-a) \\
c. [When Association is Causation](#when-association-is-causation)
"""
)


"""
# The Potential Outcomes Framework & the Fundamental Problem of Causal Inference
---

### Observed Outcomes and Treatments

$Y_i$ is the **Observed Outcome** for the individual $i$. $T_i$ is the **treatment**
for the same individual. When the treatement variable is binary--i.e. the individual
can either receive the treatment or not--then it is defined as follows:

$
T_i = \\begin{cases}
        1, & \\text{if individual } i \\text{ received the treatment} \\\\
        0, & \\text{otherwise}
    \\end{cases}
$

"""


if SHOW_CODE:
    st.code(
        """
    import pandas as pd
    import numpy as np

    # Initialize a dataset for four individuals, each indexed by `i`
    causal_data = pd.DataFrame(index=(1, 2, 3, 4))
    causal_data.index.name = "i"

    # Observed outcomes for each individual
    observed_outcomes = (500, 600, 850, 750)
    causal_data["Y"] = observed_outcomes

    # Assigned Treatment for each individual
    treatment = (0, 0, 1, 1)
    causal_data["T"] = treatment

    causal_data.head()
    """
    )

# Initialize a demo causal dataset, individuals indexed by `i`
causal_data = pd.DataFrame()
causal_data = pd.DataFrame(index=(1, 2, 3, 4))
causal_data.index.name = "i"

# Observed Outcomes
observed_outcomes = (500, 600, 850, 750)
causal_data["Y"] = observed_outcomes

# Assigned Treatments
treatment = (0, 0, 1, 1)
causal_data["T"] = treatment


display_causal_data(causal_data, FORMATTERS)

"""
### Potential Outcomes

**MOTIVATION FOR POTENTIAL OUTCOMES AS A DEVICE TO THINK ABOUT CAUSALITY**

In the case of binary treatments, **Potential Outcomes** capture the state of an
individual $i$ that results from being in the presence of the treatment $T_i=1$
or the absence of the treatment $T_i=0$. One can think of the potential outcomes
as an inherent trait associated with each individual, akin to one's height or eye color,
and capture both the state of the real world, as well as the state of an alternative
universe.

Potential outcomes fall into one of two groups:

- **Factual** potential outcomes are those that are truly observed. They are what
are actualized in data that we record.
- **Counterfactual** potential outcomes on the
other hand are unrealized states that _would have happened_ in an alternative
world where the individual received the alternative treatment condition.

Formally define the potential outcomes as follows:

$Y_{1i}$ is the potential outcome of individual $i$ in the **presence of the treatment**.
  - If we _observe_ that the individual _did in fact_ receive the treatment, i.e. $Y_{1i} | T_i=1$, then we refer to $Y_{1i}$ as a **factual** potential outcome.
  - However, if _it were possible to observe_ that the individual _did not receive_ the treatment, i.e. $Y_{1i} | T_i=0$, then we refer to $Y_{1i}$ as a **counterfactual** potential outcome.

In a similar fashion, $Y_{0i}$ is the potential outcome for the same individual in the **absence of the treatment**.
  - _If it were possible to observe_ that the individual received the treatment, i.e. $Y_{0i} | T_i=1$, then we refer to $Y_{0i}$ as a **counterfactual** potential outcome.
  - However, if we _observe_ that the individual _did in fact not receive_ the treatment, i.e. $Y_{0i} | T_i=0$, then we refer to $Y_{0i}$ as a **factual** potential outcome.

In other words, factual potential outcomes occur when the state of the potential
outcome aligns with the observed treatment condition--i.e. $Y_0 | T=0$ or $Y_1 | T=1$,
while conterfactual potential outcomes are fictitiuos situations where the potential
outcome is counter to the observed treatment condition--i.e. $Y_0 | T=1$ or $Y_1 | T=0$.
"""

if SHOW_CODE:
    st.code(
        body="""

        # Assign potential outcomes for 4 the individuals
        causal_data["Y0"] = (500, 600, 700, 800)
        causal_data["Y1"] = (450, 650, 850, 750)
        causal_data.head()
        """
    )

# Assign potential outcomes for 4 individuals
causal_data["Y0"] = (500, 600, 700, 800)
causal_data["Y1"] = (450, 650, 850, 750)

FORMATTERS.append(highlight_potential_outcomes)
display_causal_data(causal_data, FORMATTERS)


"""### Individual Treatment Effect (TE)"""

st.markdown(
    """
$$
{TE}_i = Y_{1i} - Y_{0i}
$$

${TE}_1$ = :green[450] - :blue[500] = -50

${TE}_2$ = :green[650] - :blue[600] = 50

${TE}_3$ = :green[850] - :blue[700] = 150

${TE}_4$ = :green[750] - :blue[800] = -50
"""
)

if SHOW_CODE:
    st.code(
        body="""
        def TE(causal_data):
            '''
            Treatment Effect, TE
                TE = Y1 - Y0
            '''
            Y1 = causal_data["Y1"]
            Y0 = causal_data["Y0"]
            return Y1 - Y0

        causal_data["TE"] = TE(causal_data)
        causal_data.head()
        """
    )


def TE(causal_data):
    """
    Treatment Effect, TE:
        TE = Y1 - Y0
    """
    Y1 = causal_data["Y1"]
    Y0 = causal_data["Y0"]
    return Y1 - Y0


causal_data["TE"] = TE(causal_data)

display_causal_data(causal_data, FORMATTERS)

"""
### Average Treatment Effect (ATE)

$$
\\begin{align*}
ATE &= \\frac{1}{N} \sum_i^N {TE}_{i} \\\\
    &= E[Y_1 - Y_0]
\\end{align*}
$$

In the example above, the $ATE$ would be calculated as
$$
ATE = \\frac{(-50) + 50 + 150 + (-50)}{4} = 25
$$
"""

if SHOW_CODE:
    st.code(
        body="""
        def ATE(causal_data):
            '''
            Average Treatment Effect, ATE
                ATE = E[Y1 - Y0]
            '''
            return TE(causal_data).mean()

        print(ATE(causal_data))
        # 25.0
        """
    )


def ATE(causal_data):
    """
    Average Treatment Effect, ATE
        ATE = E[Y1 - Y0]
    """
    return TE(causal_data).mean()


"""
### Treatment Effect on Treated (TET)

$$
TET = Y_1 - Y_0|T = 1
$$
"""

if SHOW_CODE:
    st.code(
        body="""
        def TET(causal_data):
            '''
            Treatment Effect on Treated, TET:
                TET = (Y1 - Y0)|T=1
            '''

            T1 = causal_data["T"] == 1     # T=1
            Y1_T1 = causal_data[T1]["Y1"]  # Y1|T=1
            Y0_T1 = causal_data[T1]["Y0"]  # Y0|T=1
            return Y1_T1 - Y0_T1           # (Y1 - Y0)|T=1
        
        causal_data["TET"] = TET(causal_data)
        causal_data.head()
        """
    )


def TET(causal_data):
    """
    Treatment Effect on Treated, TET:
        TET = (Y1 - Y0) | T=1
    """
    T1 = causal_data["T"] == 1
    Y1_T1 = causal_data[T1]["Y1"]
    Y0_T1 = causal_data[T1]["Y0"]
    return Y1_T1 - Y0_T1


causal_data["TET"] = TET(causal_data)

display_causal_data(causal_data, FORMATTERS)


"""

### Average Treatment Effect on the Treated (ATT)

In the example above, the $ATT$ would be calculated as
$$
ATT = \\frac{150 + (-50)}{2} = 50
$$

"""

if SHOW_CODE:
    st.code(
        body="""
        def ATT(causal_data):
            '''
            Average Treatment Effect on Treated, ATT
                ATT = E[Y1 - Y0|T=1]
            '''
            return TET(causal_data).mean()

        print(ATT(causal_data))
        # 50
        """
    )


def ATT(causal_data):
    """
    Average Treatment Effect on Treated, ATT
        ATE = E[Y1 - Y0 | T=1]
    """
    return TET(causal_data).mean()


"""## Causation vs Association"""


"""
### Bias (B)

In the real world individuals vary on countless dimensions, not just the
dimensions that we decide to intervene on. These additional dimensions can vary
in conjunction with the treatment. We can think of **Bias** as capturing the
inherent differences in the control and variation groups that cannot be accounted
for directly by the treatment.

We can formally define bias as the difference in the average outcome between the
control group ($T=0$) and variation group ($T=1)$ in the (counterfactual) world
where neither group receives the treatment.


$$
B = E[Y_0|T=1] - E[Y_0|T=0]
$$

"""

if SHOW_CODE:
    st.code(
        body="""
        def bias(causal_data):
            '''
            Bias:
                B = E[Y0|T=1] - E[Y0|T=0]
            '''

            T0 = causal_data["T"] == 0          # T=0
            T1 = causal_data["T"] == 1          # T=1

            Y0_T1 = causal_data[T1]["Y0"]       # Y0|T=1
            Y0_T0 = causal_data[T0]["Y0"]       # Y0|T=0

            return Y0_T1.mean() - Y0_T0.mean()  # E[Y0|T=1] - E[Y0|T=0]

        print(bias(causal_data))
        200.0
        """
    )


def bias(causal_data):
    """
    E[Y0|T=1] - E[Y0|T=0]
    """
    T0 = causal_data["T"] == 0
    T1 = causal_data["T"] == 1

    Y0_T1 = causal_data[T1]["Y0"]
    Y0_T0 = causal_data[T0]["Y0"]
    return Y0_T1.mean() - Y0_T0.mean()


"""
### Association (A)

Association is the difference in expected outcomes between the treated and
untreated condition.
"""

st.latex(
    """
    \\begin{align*}
    A = E[Y|T=1] - E[Y|T=0]
    \\end{align*}
"""
)


if SHOW_CODE:
    st.code(
        body="""
        def association(causal_data):
            '''
            Association, A:
                A = E[Y|T=1] - E[Y|T=0] 
                  = E[Y1|T=1] - E[Y0|T=0]
            '''

            T0 = causal_data["T"] == 0          # T=0
            T1 = causal_data["T"] == 1          # T=1

            Y1_T1 = causal_data[T1]["Y1"]       # Y1|T=1
            Y0_T0 = causal_data[T0]["Y0"]       # Y0|T=0

            return Y1_T1.mean() - Y0_T0.mean()  # E[Y1|T=1] - E[Y0|T=0]

        print(association(causal_data))
        """
        "250"
    )


def association(causal_data):
    """E[Y|T=1] - E[Y|T=0] = E[Y1|T=1] - E[Y0|T=0]"""
    T0 = causal_data["T"] == 0
    T1 = causal_data["T"] == 1

    Y1_T1 = causal_data[T1]["Y1"]
    Y0_T0 = causal_data[T0]["Y0"]

    return Y1_T1.mean() - Y0_T0.mean()


"""### When Association is Causation

It turns out that Association can be broken down into two terms, nameley the
Bias $B$ and the Average Treatment Effect on the Treatment, $ATT$.

"""

st.markdown(
    """
    $$
    A = ATT + B
    $$
    """
)


show_proof = st.toggle("💡 Show the derivation", value=False)

if show_proof:
    st.markdown(
        """
    Starting with the formal definition of association in terms of potential
    outcomes:

    $$
    A = E[Y|T=1] - E[Y|T=0],
    $$

    we can substitute the potential outcomes with observed outcomes: For the treated,
    the observed outcome is $Y_1$, thus $E[Y|T=1] = E[Y_1|T=1]$.
    Similarly, for the untreated, the observed outcome is $Y_0$, thus $E[Y|T=0] = E[Y_0|T=0]$.
    This gives us the Association in terms of observed outcomes.

    $$
    A = E[Y_1|T=1] - E[Y_0|T=0]
    $$

    We can now perform a little trick by adding and subtracting the following term to $A$:

    $$
    \\pm \; E[Y_0|T=1]
    $$

    This term is the _counterfactual outcome_ of _would have happened_ to the treated group,
    had they not received the treatment.

    $$
    \\begin{align*}
    A &= E[Y_1|T=1] - E[Y_0|T=0] \pm E[Y_0|T=1] & \\pm \\text{ counterfactual} \\\\
    &= \color{blue}{E[Y_1|T=1]} \; \color{black}{-} \; \color{red}{E[Y_0|T=0]} \color{black} + \color{red}{E[Y_0|T=1]} \color{black}{-} \color{blue}{E[Y_0|T=1]} &  \\text{...grouping terms} \\\\
    &= \color{blue}{E[Y_1|T=1] - E[Y_0|T=1]} \; \color{black}{+} \; \color{red}{E[Y_0|T=1] - E[Y_0|T=0]} & \\text{...rearranging terms} \\\\
    &= \\color{blue}{\\underbrace{E[Y_1 - Y_0|T=1]}_\\text{ATT}} \; \color{black}{+} \; \color{red}{\\underbrace{E[Y_0|T=1] - E[Y_0|T=0])}_\\text{BIAS}} & \\text{...giving us two components} \\\\
    \\end{align*}
    $$
"""
    )


# Run the actual assertion
assert np.isclose(
    association(causal_data), ATT(causal_data).mean() + bias(causal_data), atol=0.01
)

if SHOW_CODE:

    st.code(
        """
    assert np.isclose(
        association(causal_data), ATT(causal_data) + bias(causal_data),
        atol=0.01
    )
    """
    )


"""
## Interactive Demo
"""


def generate_causal_data(
    n_points=3,
    beta_confound=1,
    beta_assignment=0,
    ate=0.5,
    outcome_variance=0.5,
    random_seed=12,
):
    def logistic(x):
        return 1 / (1 + np.exp(-x))

    np.random.seed(random_seed)

    # Some confounding variable, X
    X = np.random.randn(n_points)

    # Treatment is function of X

    log_odds = beta_assignment * X
    # T = log_odds > 0

    np.random.seed(random_seed)
    p_treatment = logistic(log_odds)
    T = bernoulli.rvs(p=p_treatment)

    # Random indvidual treatment effects distributed around ATE
    TE = ate + np.random.randn(n_points) * ate / 5
    Y0 = (beta_confound * X) + np.random.randn(n_points) * outcome_variance
    Y1 = Y0 + TE

    df = pd.DataFrame(dict(X=X, Y0=Y0, Y1=Y1, T=T, TE=TE))
    df["Y"] = df.apply(lambda x: x["Y1"] if x["T"] == 1 else x["Y0"], axis=1)

    return df


lparam_col, rparam_col = st.columns([0.5, 0.5])

with lparam_col:
    outcome_confounding = st.slider(
        label="Outcome Confounding", min_value=-0.0, max_value=3.0, value=0.0, step=0.1
    )

    ate = st.slider(
        label="Treatment Effect",
        min_value=0.0,
        max_value=2.0,
        value=0.0,
        step=0.01,
    )

with rparam_col:
    assignment_confounding = st.slider(
        label="Assignment Confounding",
        min_value=0.0,
        max_value=5.0,
        value=0.0,
        step=0.01,
    )

    outcome_variance = st.slider(
        label="Outcome Variance",
        min_value=0.0,
        max_value=1.0,
        value=0.5,
        step=0.01,
    )


Y0_COLORS = ["darkblue", "lightgreen"]
Y1_COLORS = ["lightblue", "darkgreen"]
MARKER_SIZE = 100
N_SHOW = 25
DATA_RANGE = 3.5


def plot_observed(causal_data, n_show=N_SHOW):
    treatment_means = []

    for T in (0, 1):
        treatment_data = causal_data[causal_data["T"] == T]
        treatment_means.append(treatment_data["Y"].mean())

        color = Y1_COLORS[T] if (T == 1) else Y0_COLORS[T]
        sns.scatterplot(
            treatment_data.iloc[:n_show],
            x="X",
            y=f"Y{T}",
            s=MARKER_SIZE,
            color=color,
            zorder=2,
            label=f"$Y|T={T}$",
        )

    for T in (0, 1):
        color = Y1_COLORS[T] if T == 1 else Y0_COLORS[T]
        plt.axhline(treatment_means[T], color=color, label=f"$E[Y|T={T}]$")

    ASSOCIATION = association(causal_data)
    plt.xlim((-DATA_RANGE, DATA_RANGE))
    plt.ylim((-DATA_RANGE, DATA_RANGE))
    plt.grid()
    plt.xlabel("$X$")
    plt.ylabel("$Y$")
    plt.legend(loc="lower right")
    plt.title(f"ASSOCIATION, $A$\n$E[Y|T=1] - E[Y|T=0]$ = {ASSOCIATION:0.3}")
    return ASSOCIATION


def plot_counterfactuals(causal_data, n_show=N_SHOW):
    for T in (0, 1):
        treatment_data = causal_data[causal_data["T"] == T].iloc[:n_show]
        sns.scatterplot(
            treatment_data,
            x="X",
            y="Y0",
            color=Y0_COLORS[T],
            s=MARKER_SIZE,
            zorder=3,
            label=f"$Y_0|T={T}$",
        )
        sns.scatterplot(
            treatment_data,
            x="X",
            y="Y1",
            color=Y1_COLORS[T],
            s=MARKER_SIZE,
            zorder=2,
            label=f"$Y_1|T={T}$",
        )
        for i, r in treatment_data.iterrows():
            label = None if i > 0 else "$Y_{1i} - Y_{0i}$"
            plt.plot(
                (r.X, r.X),
                (r.Y0, r.Y1),
                color="k",
                linewidth=0.5,
                zorder=1,
                label=label,
            )

    ATT = TET(causal_data).mean()

    plt.xlim((-DATA_RANGE, DATA_RANGE))
    plt.ylim((-DATA_RANGE, DATA_RANGE))
    plt.xlabel("$X$")
    plt.ylabel("$Y$")
    plt.grid()
    plt.legend(loc="lower right", title="Individual\nTreatment\nEffects")
    plt.title(f"ATT\n$E[Y_1-Y_0|T=1]$ = {ATT:0.3}")

    return ATT


def plot_bias(causal_data, n_show=N_SHOW):
    treatment_means = []
    for T in (0, 1):
        treatment_data = causal_data[causal_data["T"] == T]
        sns.scatterplot(
            treatment_data.iloc[:n_show],
            x="X",
            y="Y0",
            s=MARKER_SIZE,
            color=Y0_COLORS[T],
            zorder=2,
            label=f"$Y_0|T={T}$",
        )
        treatment_mean = treatment_data["Y0"].mean()
        treatment_means.append(treatment_mean)

    for T in (0, 1):
        plt.axhline(treatment_means[T], color=Y0_COLORS[T], label=f"$E[Y_0|T={T}]$")

    BIAS = bias(causal_data)
    plt.xlim((-DATA_RANGE, DATA_RANGE))
    plt.ylim((-DATA_RANGE, DATA_RANGE))
    plt.xlabel("$X$")
    plt.ylabel("$Y_0$")
    plt.grid()
    plt.legend(loc="lower right")

    plt.title(f"BIAS, $B$\n$E[Y_0|T=1] - E[Y_0|T=0]$ = {BIAS:0.3}")
    return BIAS


# Run simulation
causal_data = generate_causal_data(
    n_points=10000,
    beta_confound=outcome_confounding,
    beta_assignment=assignment_confounding,
    ate=ate,
    outcome_variance=outcome_variance,
    random_seed=12,
)

fig, axs = plt.subplots(1, 3, figsize=(15, 5))

plt.sca(axs[0])
BIAS = plot_bias(causal_data)

plt.sca(axs[1])
ATT = plot_counterfactuals(causal_data)

plt.sca(axs[2])
ASSOCIATION = plot_observed(causal_data)

plt.margins(tight=True)

st.pyplot(fig)

# Assert Association relationship holds for all simulations
assert np.isclose(ASSOCIATION, ATT + BIAS, atol=0.01)


"""
## Review
- Causality is defined using potential outcomes, not realized (observed) outcomesff
- Observed association is neither necessary nor sufficient to establish causation
- Estimating causal effect of a treatment generally requires understanding the assignment mechanism
"""

change_link_color = """
<style>
    a:visited{
        color:None;
        background-color: transparent;
        text-decoration: none;
    }
    a:hover{
        color:red;
        background-color: transparent;
        text-decoration: none;
    }
    
    a:link {
        color: None;
        background-color: transparent;
        text-decoration: none;
}
</style>
"""
st.markdown(change_link_color, True)

make_footer()
