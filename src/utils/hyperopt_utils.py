from typing import Dict, Any
import pandas as pd
from ray import tune
from ray.tune.search.optuna import OptunaSearch
from ray.tune.schedulers import ASHAScheduler, PopulationBasedTraining
from ray.tune.search import ConcurrencyLimiter
import optuna


def setup_hyperopt_dashboard(experiment_dir: str):
    """Setup Ray dashboard with useful parameter importance analysis"""
    from ray.tune.dashboard.dashboard import Dashboard
    dashboard = Dashboard(
        experiment_dir,
        host="127.0.0.1",
        port=8265
    )
    print(f"View hyperparameter tuning dashboard at: http://127.0.0.1:8265")
    return dashboard


def create_search_algorithm(strategy: str = "optuna"):
    """Create search algorithm with smart initialization strategies"""
    if strategy == "optuna":
        # Optuna provides better sampling strategies than basic random search
        search_alg = OptunaSearch(
            metric="val_total_loss",
            mode="min",
            sampler=optuna.samplers.TPESampler(
                n_startup_trials=5,  # Number of random trials before optimization
                multivariate=True,  # Consider parameter relationships
                seed=42
            )
        )
    else:
        search_alg = tune.suggest.HyperOptSearch(
            metric="val_total_loss",
            mode="min"
        )

    # Limit concurrent trials to avoid resource exhaustion
    return ConcurrencyLimiter(search_alg, max_concurrent=4)


def create_scheduler(strategy: str = "asha"):
    """Create scheduler for efficient trial termination"""
    if strategy == "asha":
        # Asynchronous Successive Halving Algorithm
        return ASHAScheduler(
            time_attr='training_iteration',
            metric="val_total_loss",
            mode="min",
            max_t=100,
            grace_period=10,
            reduction_factor=2,
            brackets=3
        )
    elif strategy == "pbt":
        # Population Based Training
        return PopulationBasedTraining(
            time_attr="training_iteration",
            metric="val_total_loss",
            mode="min",
            perturbation_interval=10,
            hyperparam_mutations={
                "training.learning_rate": tune.loguniform(1e-4, 1e-2),
                "model.dropout": tune.uniform(0.1, 0.5)
            }
        )


def analyze_results(analysis):
    """Analyze hyperparameter optimization results"""
    # Convert results to DataFrame
    df = analysis.results_df

    # Get best trial
    best_trial = analysis.best_trial
    best_config = analysis.best_config

    # Calculate parameter importance
    importance_df = analyze_parameter_importance(df)

    # Generate parameter correlation analysis
    correlation_df = analyze_parameter_correlations(df)

    return {
        "best_trial": best_trial,
        "best_config": best_config,
        "importance": importance_df,
        "correlations": correlation_df,
        "all_results": df
    }


def train_with_hyperopt(config: Dict[str, Any], datamodule, exp_dir: str):
    """Enhanced hyperparameter optimization with better visualization and analysis"""

    # Setup search space with smart initialization
    search_space = {
        "model": {
            "num_layers": tune.randint(2, 5),
            "dropout": tune.uniform(0.1, 0.5),
            "temp_dropout_rate": tune.uniform(0.1, 0.5),
            "hidden_dim": tune.choice([64, 128, 256, 512]),
            "temp_kernels": tune.sample_from(
                lambda _: [tune.choice([4, 8, 16])] * tune.randint(2, 5).sample()
            )
        },
        "training": {
            "learning_rate": tune.loguniform(1e-4, 1e-2),
            "batch_size": tune.choice([16, 32, 64, 128]),
            "weight_decay": tune.loguniform(1e-5, 1e-3)
        }
    }

    # Setup search algorithm and scheduler
    search_alg = create_search_algorithm("optuna")
    scheduler = create_scheduler("asha")

    # Run hyperparameter optimization
    analysis = tune.run(
        train_tune_model,
        config=search_space,
        num_samples=50,
        search_alg=search_alg,
        scheduler=scheduler,
        local_dir=exp_dir,
        name="hyperopt",
        resources_per_trial={"cpu": 4, "gpu": 0.5},  # Adjust based on your resources
        checkpoint_freq=10,
        keep_checkpoints_num=2,
        checkpoint_score_attr="val_total_loss",
        progress_reporter=tune.CLIReporter(
            parameter_columns=["model.num_layers", "model.dropout", "training.learning_rate"],
            metric_columns=["val_total_loss", "val_los_prediction_msle", "training_iteration"]
        )
    )

    # Analyze results
    results = analyze_results(analysis)

    # Save analysis
    save_hyperopt_results(results, exp_dir)

    # Start dashboard
    dashboard = setup_hyperopt_dashboard(exp_dir)

    return results


def save_hyperopt_results(results: Dict, exp_dir: str):
    """Save hyperparameter optimization results"""
    results_dir = Path(exp_dir) / "hyperopt_results"
    results_dir.mkdir(exist_ok=True)

    # Save best configuration
    with open(results_dir / "best_config.yaml", "w") as f:
        yaml.dump(results["best_config"], f)

    # Save parameter importance analysis
    results["importance"].to_csv(results_dir / "parameter_importance.csv")

    # Save parameter correlations
    results["correlations"].to_csv(results_dir / "parameter_correlations.csv")

    # Save all trials data
    results["all_results"].to_csv(results_dir / "all_trials.csv")

    # Create summary visualization
    create_summary_plots(results, results_dir)


def create_summary_plots(results: Dict, save_dir: Path):
    """Create summary visualizations of hyperparameter optimization"""
    import plotly.express as px
    import plotly.graph_objects as go

    # Parameter importance plot
    fig = px.bar(
        results["importance"],
        x="importance",
        y="parameter",
        orientation="h",
        title="Parameter Importance"
    )
    fig.write_html(save_dir / "parameter_importance.html")

    # Learning curves for top trials
    fig = go.Figure()
    top_trials = results["all_results"].nsmallest(5, "val_total_loss")
    for _, trial in top_trials.iterrows():
        fig.add_trace(go.Scatter(
            y=trial["val_total_loss_history"],
            name=f"Trial {trial['trial_id']}"
        ))
    fig.update_layout(title="Learning Curves for Top Trials")
    fig.write_html(save_dir / "learning_curves.html")
