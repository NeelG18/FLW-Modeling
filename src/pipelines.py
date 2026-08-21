"""Single source of truth for all eight regression model configurations.

Every experiment -- the single-split comparison, the year ablation, the
walk-forward validation -- builds its estimators through ``build_pipeline``
here. Nothing else in the codebase may instantiate a regressor directly, so
that a hyperparameter appears in exactly one place and cannot drift between
experiments.

Two things live here:

``TUNED_PARAMS``
    The selected configuration for each model, as chosen by the searches in
    ``ablations.run_hyperparameter_search``.

``SEARCH_SPACES``
    The grids/distributions those searches ran over, kept alongside the
    results so the selection is reproducible rather than asserted.
"""

from scipy.stats import randint, uniform
from sklearn import neighbors, svm
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, PolynomialFeatures, StandardScaler
from sklearn.tree import DecisionTreeRegressor

from data_prep import GROUP_COLS

MODEL_NAMES = [
    "KNN",
    "Neural Network",
    "SVM",
    "Linear Regression",
    "Decision Tree",
    "Random Forest",
    "Ridge",
    "Poly Ridge",
]

# --------------------------------------------------------------------------
# Categorical encoding policy
# --------------------------------------------------------------------------
# Which models drop the first level of each categorical variable.
#
# Dropping a level removes the exact linear dependence between a category's
# indicator columns and the intercept. That matters only for the models
# fitted by least squares, where the dependence makes the design matrix
# singular; those are listed below.
#
# For every other model it is not merely unnecessary but harmful. The
# encoder is configured with handle_unknown="ignore", so a category unseen
# during training encodes as all zeros -- which is exactly how the dropped
# reference level also encodes. The two become indistinguishable, and a
# country the model has never encountered is silently treated as the
# reference country rather than as unknown. Distance, tree, and kernel
# models have no intercept to alias against, so they keep the full
# indicator set and retain that distinction.
#
# The least-squares models below therefore accept the ambiguity knowingly:
# an unseen category is predicted as though it were the reference level.
# That is the cost of a non-singular design matrix, and it is bounded by
# how often unseen categories occur at prediction time.
DROP_FIRST_MODELS = frozenset({"Linear Regression", "Ridge", "Poly Ridge"})

# --------------------------------------------------------------------------
# Selected hyperparameters
# --------------------------------------------------------------------------
TUNED_PARAMS = {
    "KNN": dict(n_neighbors=5, p=1, weights="distance"),
    "Neural Network": dict(
        activation="relu",
        alpha=0.031198232171566222,
        hidden_layer_sizes=(200, 100, 50),
        learning_rate="adaptive",
        solver="sgd",
        random_state=1,
        max_iter=1183,
        early_stopping=False,
        validation_fraction=0.1,
        n_iter_no_change=10,
    ),
    "SVM": dict(kernel="poly", gamma="scale", degree=2, C=10),
    "Linear Regression": dict(fit_intercept=True),
    "Decision Tree": dict(
        criterion="squared_error",
        max_depth=None,
        min_samples_split=10,
        min_samples_leaf=1,
        ccp_alpha=0.001,
        random_state=0,
    ),
    "Random Forest": dict(
        n_estimators=100,
        max_depth=None,
        min_samples_split=5,
        min_samples_leaf=1,
        max_features="sqrt",
        bootstrap=False,
        random_state=0,
    ),
    "Ridge": dict(alpha=0.1, fit_intercept=True),
    "Poly Ridge": dict(alpha=10, fit_intercept=False, poly_degree=3),
}

# --------------------------------------------------------------------------
# Search spaces the selections above came from
# --------------------------------------------------------------------------
SEARCH_SPACES = {
    "KNN": {
        "kind": "grid",
        "cv": 5,
        "space": {
            "regressor__n_neighbors": [5, 10, 15, 20],
            "regressor__weights": ["uniform", "distance"],
            "regressor__p": [1, 2],
        },
    },
    "Linear Regression": {
        "kind": "grid",
        "cv": 5,
        "space": {
            "regressor__fit_intercept": [True, False],
            "regressor__positive": [True, False],
        },
    },
    "Ridge": {
        "kind": "grid",
        "cv": 5,
        "space": {
            "regressor__alpha": [0.1, 1.0, 10.0, 100.0],
            "regressor__fit_intercept": [True, False],
        },
    },
    "Decision Tree": {
        "kind": "grid",
        "cv": 5,
        "space": {
            "regressor__criterion": ["squared_error", "friedman_mse"],
            "regressor__max_depth": [None, 10, 20, 30],
            "regressor__min_samples_split": [2, 5, 10, 15],
            "regressor__min_samples_leaf": [1, 2, 4, 8],
            "regressor__ccp_alpha": [0.0, 0.001, 0.01],
        },
    },
    "Random Forest": {
        "kind": "grid",
        "cv": 3,
        "space": {
            "regressor__n_estimators": [100, 200],
            "regressor__max_depth": [10, 20, None],
            "regressor__min_samples_split": [2, 5],
            "regressor__min_samples_leaf": [1, 2],
        },
    },
    "Neural Network": {
        "kind": "random",
        "cv": 3,
        "n_iter": 20,
        "random_state": 42,
        "space": {
            "regressor__hidden_layer_sizes": [
                (50,),
                (100,),
                (100, 50),
                (150, 100),
                (200, 100, 50),
            ],
            "regressor__activation": ["relu", "tanh"],
            "regressor__solver": ["adam", "sgd"],
            "regressor__alpha": uniform(0.0001, 0.1),
            "regressor__learning_rate": ["constant", "invscaling", "adaptive"],
            "regressor__max_iter": randint(500, 1500),
            "regressor__early_stopping": [True, False],
        },
    },
}


# --------------------------------------------------------------------------
# Preprocessing
# --------------------------------------------------------------------------
def make_column_transformer(year_mode="scaled", drop_first=False):
    """Build the categorical encoder plus the chosen treatment of ``year``.

    Parameters
    ----------
    year_mode : {'scaled', 'raw', 'none'}
        ``'scaled'`` standardises ``year`` (fit inside the pipeline, so the
        scaler never sees the held-out fold). ``'raw'`` passes it through
        unscaled. ``'none'`` drops it, leaving a purely cross-sectional model.
    drop_first : bool
        Drop the first level of each categorical variable. See
        ``DROP_FIRST_MODELS`` for when this is appropriate; prefer
        ``make_column_transformer_for`` so the choice follows the model.
    """
    encoder = OneHotEncoder(
        drop="first" if drop_first else None, handle_unknown="ignore"
    )

    if year_mode == "none":
        return ColumnTransformer(
            transformers=[("cat", encoder, GROUP_COLS)],
            remainder="drop",
        )
    if year_mode == "raw":
        return ColumnTransformer(
            transformers=[("cat", encoder, GROUP_COLS)],
            remainder="passthrough",
        )
    if year_mode == "scaled":
        return ColumnTransformer(
            transformers=[
                ("cat", encoder, GROUP_COLS),
                ("num", StandardScaler(), ["year"]),
            ],
            remainder="drop",
        )
    raise ValueError(f"Unknown year_mode: {year_mode!r}")


def make_column_transformer_for(name, year_mode="scaled"):
    """Transformer with the encoding policy this model requires."""
    if name not in MODEL_NAMES:
        raise ValueError(f"Unknown model: {name!r}")
    return make_column_transformer(year_mode, drop_first=name in DROP_FIRST_MODELS)


# --------------------------------------------------------------------------
# Estimators
# --------------------------------------------------------------------------
def _make_regressor(name, param_overrides=None):
    params = dict(TUNED_PARAMS[name])
    if param_overrides:
        params.update(param_overrides)

    if name == "KNN":
        return neighbors.KNeighborsRegressor(**params)
    if name == "Neural Network":
        return MLPRegressor(**params)
    if name == "SVM":
        return svm.SVR(**params)
    if name == "Linear Regression":
        return LinearRegression(**params)
    if name == "Decision Tree":
        return DecisionTreeRegressor(**params)
    if name == "Random Forest":
        return RandomForestRegressor(**params)
    if name == "Ridge":
        return Ridge(**params)
    if name == "Poly Ridge":
        params.pop("poly_degree")
        return Ridge(**params)
    raise ValueError(f"Unknown model: {name!r}")


def build_pipeline(name, year_mode="scaled", column_transformer=None, param_overrides=None):
    """Build the full preprocessing + estimator pipeline for one model.

    Parameters
    ----------
    name : str
        One of ``MODEL_NAMES``.
    year_mode : {'scaled', 'raw', 'none'}
        Passed to ``make_column_transformer`` when no transformer is supplied.
    column_transformer : ColumnTransformer, optional
        Use a pre-built transformer instead. Callers that need to inspect the
        encoded matrix before fitting supply the transformer they inspected.
    param_overrides : dict, optional
        Replace individual tuned hyperparameters. Intended for ablations
        that sweep one setting; the tuned values remain the default so a
        sweep cannot quietly become the configuration used elsewhere.
    """
    if name not in MODEL_NAMES:
        raise ValueError(f"Unknown model: {name!r}")

    ct = (
        column_transformer
        if column_transformer is not None
        else make_column_transformer_for(name, year_mode)
    )
    steps = [("preprocessor", ct)]

    if name == "Poly Ridge":
        steps.append(
            (
                "poly",
                PolynomialFeatures(
                    degree=TUNED_PARAMS["Poly Ridge"]["poly_degree"],
                    include_bias=False,
                ),
            )
        )

    steps.append(("regressor", _make_regressor(name, param_overrides)))
    return Pipeline(steps)
