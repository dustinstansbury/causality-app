import streamlit as st
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

from streamlit.components.v1 import html

from matplotlib import pyplot as plt
from scipy import stats
import seaborn as sns

SHOW_ALL_CODE = st.checkbox("Show all code cells", value=True)


"""
# Definitions & Notation


## Treatments & Observed Outcomes

$
T_i = \\begin{cases}
        1, & \\text{if individual } i \\text{ received the treatment} \\\\
        0, & \\text{otherwise}
    \\end{cases}
$

$Y_i$ is the observed outcome for the individual $i$

"""

# Initialize a demo causal dataset, individuals indexed by `i`
causal_data = pd.DataFrame()
causal_data = pd.DataFrame(index=(1, 2, 3, 4))
causal_data.index.name = "i"

# Assign Treatments
treatment = (0, 0, 1, 1)
causal_data["T"] = treatment

# Observed Outcomes
observed_outcomes = (500, 600, 850, 750)
causal_data["Y"] = observed_outcomes

st.dataframe(causal_data.head())

show_oo_code = st.checkbox(
    "Show **Treatments** and **Observed Outcomes** as code", value=SHOW_ALL_CODE
)
if show_oo_code:
    st.code(
        """
    import pandas as pd
    import numpy as np
            
    # Initialize a demo dataset, individuals indexed by `i`
    causal_data = pd.DataFrame(index=(1, 2, 3, 4))
    causal_data.index.name = "i"

    # Assign Treatments
    treatment = (0, 0, 1, 1)
    causal_data["T"] = treatment

    # Observed Outcomes
    observed_outcomes = (500, 600, 850, 750)
    causal_data["Y"] = observed_outcomes

    causal_data.head()
    """
    )

"""
## Potential Outcomes & The Fundamental Problem of Causal Inference 🪄

### Potential Outcomes


- $Y_{0i}$ is the potential outcome for unit $i$ in the absence of the treatment. If we were to observe that the unit did not receive the treatment, this is a **factual potential outcome**
- $Y_{1i}$ is the potential outcome for the same unit in the precsence of the treatment. When we observe that the unit did not receive the treatment, this is a **counterfactual potential outcome**


"""

# Assign potential outcomes for 4 individuals
causal_data["Y0"] = (500, 600, 700, 800)
causal_data["Y1"] = (450, 650, 850, 750)


def highlight_potential_outcomes(data):
    """
    Highlight the counterfactual cells
    """

    # Empty DF with no styling
    df_styled = pd.DataFrame("", index=data.index, columns=data.columns)

    for T in (0, 1):
        factual_mask = (data["T"] == T) & (data["Y"] == data[f"Y{T}"])
        counterfactual_mask = (data["T"] != T) & (data["Y"] != data[f"Y{T}"])
        if T == 1:
            factual_format = "background-color: rgba(0, 128, 50, 0.85);"
            counterfactual_format = "background-color: rgba(0, 128, 50, 0.35);"
        else:
            factual_format = "background-color: rgba(0, 100, 256, 0.85);"
            counterfactual_format = "background-color: rgba(0, 100, 256, 0.35);"

        df_styled.loc[factual_mask, f"Y{T}"] = factual_format
        df_styled.loc[counterfactual_mask, f"Y{T}"] = counterfactual_format

    return df_styled


st.dataframe(causal_data.style.apply(highlight_potential_outcomes, axis=None))


show_po_code = st.checkbox("Show **Potential_outcomes** as code", value=SHOW_ALL_CODE)
if show_po_code:
    st.code(
        body="""

        # Assign potential outcomes for 4 the individuals
        causal_data["Y0"] = (500, 600, 700, 800)
        causal_data["Y1"] = (450, 650, 850, 750)
        causal_data.head()
        """
    )


"""### Individual Treatment Effect"""

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


def TE(causal_data):
    """
    Treatment Effect, TE:
        TE = Y1 - Y0
    """
    Y1 = causal_data["Y1"]
    Y0 = causal_data["Y0"]
    return Y1 - Y0


causal_data["TE"] = TE(causal_data)

st.dataframe(causal_data.style.apply(highlight_potential_outcomes, axis=None))
# st.dataframe(causal_data)


show_te_code = st.checkbox("Show **Treatement Effect** as code", value=SHOW_ALL_CODE)
if show_te_code:
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

"""
### Treatment Effect on Treated (TET)

$$
TET = Y_1 - Y_0|T = 1
$$
"""


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
# st.dataframe(causal_data)
st.dataframe(causal_data.style.apply(highlight_potential_outcomes, axis=None))

show_tet_code = st.checkbox(
    "Show **Treatment Effect on Treated** as code", value=SHOW_ALL_CODE
)
if show_tet_code:
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


"""### Average Treatment Effect

#### WIP


### Average Treatment Effect on the Treated

#### WIP

"""


"""### Association, Bias, & Causation"""


"""
#### Association ($A$)

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


show_assoc_code = st.checkbox("Show **Association** as code", value=SHOW_ALL_CODE)
if show_assoc_code:
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


# st.write(association(causal_data))


"""
#### Bias

In the real world individuals vary on countless dimensions, not just the
dimensions that we decide to intervene on. These additional dimensions can vary
in conjunction with the treatment. We can think of **Bias** as capturing the
inherent differences in the control and variation groups that cannot be accounted
for directly by the treatment.

We can formally define bias as the difference in the average outcome between the
control group ($T=0$) and variation group ($T=1)$ in the (counterfactual) world
where neither group receives the treatment.
"""

st.latex("E[Y_0|T=1] - E[Y_0|T=0]")


show_bias_code = st.checkbox("Show **Bias** as code", value=SHOW_ALL_CODE)
if show_bias_code:
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


"""## Association vs Causation"""

# Run the actual assertion
assert np.isclose(
    association(causal_data), TET(causal_data).mean() + bias(causal_data), atol=0.01
)

show_assoc_assert_code = st.checkbox(
    "Demonstrate the Association relationship in code", value=SHOW_ALL_CODE
)
if show_assoc_assert_code:

    st.code(
        """
    assert np.isclose(
        association(causal_data),
        TET(causal_data).mean() + bias(causal_data),
        atol=0.01
    )
    """
    )


"""
### Demonstrating the interaction between Bias, Treatment Effect, and observed Association 
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
    T = stats.bernoulli.rvs(p=p_treatment)

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
        label="Outcome Confounding", min_value=0.0, max_value=3.0, value=0.0, step=0.1
    )

    ate = st.slider(
        label="Treatment Effect",
        min_value=0.0,
        max_value=2.0,
        value=1.0,
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
        min_value=0.2,
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
    plt.title(f"ASSOCIATION\n$E[Y|T=1] - E[Y|T=0]$ = {ASSOCIATION:0.3}")
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

    plt.title(f"BIAS\n$E[Y_0|T=1] - E[Y_0|T=0]$ = {BIAS:0.3}")
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
# Review
- Causality is defined using potential outcomes, not realized (observed) outcomesff
- Observed association is neither necessary nor sufficient to establish causation
- Estimating causal effect of a treatment generally requires understanding the assignment mechanism
"""

# bmc_button = """
# <script type="text/javascript" src="https://cdnjs.buymeacoffee.com/1.0.0/button.prod.min.js" data-name="bmc-button" data-slug="dustinstansbury" data-color="#FFDD00" data-emoji=""  data-font="Cookie" data-text="Buy me a coffee" data-outline-color="#000000" data-font-color="#000000" data-coffee-color="#ffffff" ></script>
# """

# html(bmc_button, height=70, width=220)
