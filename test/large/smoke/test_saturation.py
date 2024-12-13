import subprocess

from os import path

def test_run_main_saturation(tmp_path, request):
    root_path = request.config.rootpath

    # Check for artifacts created within the experiment.save_path directory
    # When not specified, this is `output`
    experiment_save_path = path.join(tmp_path, "experiment", "saturation_test")

    # Check for artifacts created within hydra.run.dir directory
    # When not specified, this is `outputs/{yyyy-mm-dd}/{HH:MM:SS}``
    hydra_run_dir = path.join(tmp_path, "hydra", "saturation_test")

    # Run the script with arguments
    result = subprocess.run(
        [
            "python",
            "main.py",
            "--config-name",
            "saturation.yaml",
            "experiment.max_steps=200",
            f"hydra.run.dir={hydra_run_dir}",
            f"experiment.save_path={experiment_save_path}"
        ],
        cwd=root_path,
        capture_output=True,
        text=True
    )

    # Verify the exit code
    assert result.returncode == 0, f"Script failed with stderr: {result.stderr}"

    # Verify standard out contains training string
    test_string = "[INFO] - Starting online training..."
    assert test_string in result.stdout, f"'{test_string}' not found in stdout: {result.stdout}"

    # Check for artifacts created within experiment save_path directory
    assert path.isdir(experiment_save_path), "experiment.save_path is missing or is not a directory"

    output_dir = path.join(experiment_save_path, "delayed_saturation_debug", "param-", "seed-0")
    assert path.isdir(output_dir), "experiment.save_path output is missing or is not a directory"
    output_logs_dir = path.join(output_dir, "logs")
    assert path.isdir(output_logs_dir), "experiment.save_path logs is missing or is not a directory"
    output_plots_dir = path.join(output_dir, "plots")
    assert path.isdir(output_plots_dir), "experiment.save_path plots is missing or is not a directory"
    output_config_path = path.join(output_dir, "config.yaml")
    assert path.isfile(output_config_path), "experiment.save_path config.yaml is missing or is not a file"
    output_online_eval_path = path.join(output_dir, "Online_eval.pkl")
    assert path.isfile(output_online_eval_path), "experiment.save_path Online_eval.pkl is missing or is not a file"
    output_stats_path = path.join(output_dir, "stats.json")
    assert path.isfile(output_stats_path), "experiment.save_path stats.json is missing or is not a file"

    # TODO: verify stats.json content
    # with open(output_stats_path, "r") as f:
    #     data = json.load(f)
    # assert "key" in data, "Expected 'key' in JSON file."
    # assert isinstance(data["key"], str), "'key' should be a string."

    # Verify that the hydra output directory contains appropriate side effects
    assert path.isdir(hydra_run_dir), "hydra.run.dir path missing or is not a directory"
    logs_path = path.join(hydra_run_dir, "main.log")
    assert path.isfile(logs_path)

    with open(logs_path, "r") as f:
        main_log = f.read()

    assert test_string in main_log, f"'{test_string}' not found in main.log"
