import os

import matplotlib.pyplot as plt
import numpy as np

from heat_solver import load_config, run_simulation


def main() -> None:
    """Run the heat equation simulation from a JSON configuration."""
    config_path = "config.json"

    try:
        cfg = load_config(config_path)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return
    except ValueError as e:
        print(f"Invalid configuration: {e}")
        return

    print("Running simulation...")
    u_final = run_simulation(cfg)

    # Ensure output directory exists
    out_dir = os.path.dirname(cfg.output_csv) or "."
    os.makedirs(out_dir, exist_ok=True)

    # Save final temperature as CSV
    np.savetxt(cfg.output_csv, u_final, delimiter=",")
    print(f"Saved final temperature to {cfg.output_csv}")

    # Plot final temperature
    x = np.linspace(0.0, cfg.length, cfg.nx)
    plt.figure()
    plt.plot(x, u_final, marker="o")
    plt.xlabel("x")
    plt.ylabel("Temperature u(x, T)")
    plt.title("Final temperature distribution")

    out_plot_dir = os.path.dirname(cfg.output_plot) or "."
    os.makedirs(out_plot_dir, exist_ok=True)
    plt.savefig(cfg.output_plot)
    plt.close()
    print(f"Saved final temperature plot to {cfg.output_plot}")


if __name__ == "__main__":
    main()
