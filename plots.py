import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import PowerNorm
from scipy.ndimage import gaussian_filter1d
from pathlib import Path
from typing import List, Dict, Any


def plot_micro_single(result: Dict, out_dir: Path) -> Path:
    rid = result["robot_id"]
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    x = result["x_data"]
    y = result["y_data"]
    t = result["t_data"]

    axes[0, 0].plot(t, x, 'b-', lw=1, alpha=0.5)
    sigma = max(1, len(t) // 50)
    axes[0, 0].plot(t, gaussian_filter1d(x, sigma=sigma), 'b-', lw=2)
    axes[0, 0].set_xlabel('t')
    axes[0, 0].set_ylabel('x(t)')
    axes[0, 0].set_title(f'Robot {rid}: x(t)')

    axes[0, 1].plot(t, y, 'r-', lw=1, alpha=0.5)
    axes[0, 1].plot(t, gaussian_filter1d(y, sigma=sigma), 'r-', lw=2)
    axes[0, 1].set_xlabel('t')
    axes[0, 1].set_ylabel('y(t)')
    axes[0, 1].set_title(f'Robot {rid}: y(t)')

    axes[1, 0].plot(x, y, '-', lw=0.8, alpha=0.6, color='gray')
    axes[1, 0].plot(x[0], y[0], 'go', ms=8, label='start')
    axes[1, 0].plot(x[-1], y[-1], 'rs', ms=8, label='end')
    axes[1, 0].set_xlabel('x')
    axes[1, 0].set_ylabel('y')
    axes[1, 0].set_title(f'Robot {rid}: trajectory')
    axes[1, 0].legend()
    axes[1, 0].set_aspect('equal')

    axes[1, 1].axis('off')
    eq_text = f"Status: {result['status']}\n\n"
    if result['status'] == 'ok':
        eq_text += f"dx/dt:\n{result.get('best_x', 'N/A')}\n\n"
        eq_text += f"dy/dt:\n{result.get('best_y', 'N/A')}"
    else:
        eq_text += result.get('error', 'timeout')

    axes[1, 1].text(0.05, 0.95, eq_text, transform=axes[1, 1].transAxes,
                    fontsize=7, va='top', family='monospace',
                    bbox=dict(boxstyle='round', fc='#f0f0f0', alpha=0.9))
    axes[1, 1].set_title(f'Robot {rid}: discovered ODE')

    fig.suptitle(f'MICRO: ODE for robot {rid}', fontsize=13)
    plt.tight_layout()
    path = out_dir / f'micro_robot_{rid}.png'
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return path


def plot_micro_summary(results: List[Dict], out_dir: Path) -> Path:
    n = len(results)
    fig, axes = plt.subplots(1, min(n, 6), figsize=(4 * min(n, 6), 4))
    if n == 1:
        axes = [axes]

    for i, res in enumerate(results[:6]):
        ax = axes[i]
        x = res["x_data"]
        y = res["y_data"]
        ax.plot(x, y, '-', lw=0.6, alpha=0.7)
        ax.plot(x[0], y[0], 'go', ms=5)
        ax.set_title(f'Robot {res["robot_id"]}\nR²: {res["status"]}', fontsize=8)
        ax.set_aspect('equal')
        ax.tick_params(labelsize=6)

    fig.suptitle('MICRO: all robot trajectories', fontsize=12)
    plt.tight_layout()
    path = out_dir / 'micro_summary.png'
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return path


def plot_meso(result: Dict, out_dir: Path) -> Path:
    robot_ids = result["robot_ids"]
    ids_str = '_'.join(str(r) for r in robot_ids)

    fig = plt.figure(figsize=(16, 10))
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.4, wspace=0.35)

    ax1 = fig.add_subplot(gs[0, 0])
    if "samples" in result and "t_data" in result:
        t = result["t_data"]
        samples = result["samples"]
        cmap = plt.cm.tab10
        for i in range(0, len(samples), 2):
            idx = i // 2
            color = cmap(idx / max(1, len(robot_ids)))
            ax1.plot(t, samples[i], '-', color=color, lw=1,
                     label=f'x_{robot_ids[idx]}')
    ax1.set_xlabel('t')
    ax1.set_ylabel('x(t)')
    ax1.set_title('x-coordinates')
    ax1.legend(fontsize=6)

    ax2 = fig.add_subplot(gs[0, 1])
    if "samples" in result:
        samples = result["samples"]
        for i in range(0, len(samples), 2):
            idx = i // 2
            x = samples[i]
            y = samples[i + 1]
            color = cmap(idx / max(1, len(robot_ids)))
            ax2.plot(x, y, '-', color=color, lw=0.8, alpha=0.7,
                     label=f'Robot {robot_ids[idx]}')
            ax2.plot(x[0], y[0], 'o', color=color, ms=5)
    ax2.set_xlabel('x')
    ax2.set_ylabel('y')
    ax2.set_title('Trajectories')
    ax2.legend(fontsize=6)
    ax2.set_aspect('equal')

    ax3 = fig.add_subplot(gs[1, 0])
    ax3.axis('off')
    eq_text = f"Status: {result['status']}\n"
    eq_text += f"Robots: {robot_ids}\n\n"
    if result['status'] == 'ok':
        for eq_info in result.get('equations', []):
            if eq_info.get('level') == 0 and eq_info.get('solution') == 0:
                var = eq_info['variable']
                txt = eq_info['text_form']
                if len(txt) > 80:
                    txt = txt[:80] + '...'
                eq_text += f"{var}: {txt}\n"
    ax3.text(0.02, 0.98, eq_text, transform=ax3.transAxes,
             fontsize=7, va='top', family='monospace',
             bbox=dict(boxstyle='round', fc='#f0f0f0', alpha=0.9))
    ax3.set_title('Discovered system')

    ax4 = fig.add_subplot(gs[1, 1])
    if "coeff_table" in result and result["coeff_table"] is not None:
        df = result["coeff_table"]
        if not df.empty and df.shape[0] > 0 and df.shape[1] > 0:
            data = df.values[:min(20, df.shape[0]), :min(10, df.shape[1])]
            im = ax4.imshow(data, aspect='auto', cmap='RdBu_r')
            ax4.set_xticks(range(data.shape[1]))
            ax4.set_xticklabels(df.columns[:data.shape[1]],
                                rotation=45, ha='right', fontsize=5)
            ax4.set_ylabel('Run index')
            plt.colorbar(im, ax=ax4, shrink=0.7)
    ax4.set_title('Coefficient matrix')

    fig.suptitle(f'MESO: coupled system for robots {robot_ids}', fontsize=13)
    plt.tight_layout()
    path = out_dir / f'meso_robots_{ids_str}.png'
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return path


def plot_macro(result: Dict, out_dir: Path, robot_id: int = 0) -> Path:
    fig = plt.figure(figsize=(16, 12))
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.4, wspace=0.35)

    epde_hmaps = result.get("epde_hmaps", [])
    x_epde = result.get("x_epde", np.array([]))
    y_epde = result.get("y_epde", np.array([]))
    t_epde = result.get("t_epde", np.array([]))
    n_windows = len(epde_hmaps)

    for i in range(min(n_windows, 3)):
        ax = fig.add_subplot(gs[0, i])
        hu, hv = epde_hmaps[i]
        magnitude = np.sqrt(hu ** 2 + hv ** 2)
        if len(x_epde) > 0 and len(y_epde) > 0:
            im = ax.imshow(
                magnitude, origin='lower',
                extent=[x_epde[0], x_epde[-1], y_epde[0], y_epde[-1]],
                cmap='viridis', interpolation='bilinear',
                norm=PowerNorm(gamma=0.5),
            )
            plt.colorbar(im, ax=ax, shrink=0.7)
        else:
            ax.imshow(magnitude, origin='lower', cmap='viridis')

        t_label = f't={t_epde[i]:.2f}' if i < len(t_epde) else f'window {i}'
        ax.set_title(f'|displacement| {t_label}')
        ax.set_xlabel('x')
        ax.set_ylabel('y')

    ax_u = fig.add_subplot(gs[1, 0])
    data_u = result.get("data_u")
    if data_u is not None and data_u.ndim == 3:
        mean_u = np.mean(data_u, axis=0)
        ax_u.imshow(mean_u, origin='lower', cmap='coolwarm', interpolation='bilinear')
        ax_u.set_title('Mean u(x,y)')

    ax_v = fig.add_subplot(gs[1, 1])
    data_v = result.get("data_v")
    if data_v is not None and data_v.ndim == 3:
        mean_v = np.mean(data_v, axis=0)
        ax_v.imshow(mean_v, origin='lower', cmap='coolwarm', interpolation='bilinear')
        ax_v.set_title('Mean v(x,y)')

    ax_eq = fig.add_subplot(gs[1, 2])
    ax_eq.axis('off')
    eq_text = f"Status: {result['status']}\n\n"
    if result['status'] == 'ok':
        eq_text += f"u: {result.get('best_u', 'N/A')}\n\n"
        eq_text += f"v: {result.get('best_v', 'N/A')}"
    else:
        eq_text += result.get('error', 'timeout')

    ax_eq.text(0.02, 0.98, eq_text, transform=ax_eq.transAxes,
               fontsize=7, va='top', family='monospace',
               bbox=dict(boxstyle='round', fc='#f0f0f0', alpha=0.9))
    ax_eq.set_title('Discovered PDE')

    fig.suptitle(f'MACRO: PDE for displacement field (robot {robot_id})',
                 fontsize=13)
    plt.tight_layout()
    path = out_dir / f'macro_robot_{robot_id}.png'
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return path


def plot_micro_solver(solve_result: Dict, out_dir: Path) -> Path:
    if solve_result.get("status") != "ok":
        return None
    rid = solve_result["robot_id"]
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    t = solve_result["t_pred"]
    x_true = solve_result["x_true"]
    y_true = solve_result["y_true"]
    x_pred = solve_result["x_pred"]
    y_pred = solve_result["y_pred"]
    n = min(len(t), len(x_true))

    axes[0, 0].plot(t[:n], x_true[:n], 'b-', lw=1, alpha=0.5, label='true')
    axes[0, 0].plot(t, x_pred, 'r--', lw=1.5, label='FD solver')
    axes[0, 0].set_xlabel('t')
    axes[0, 0].set_ylabel('x(t)')
    axes[0, 0].set_title(f'Robot {rid}: x(t)')
    axes[0, 0].legend(fontsize=8)

    axes[0, 1].plot(t[:n], y_true[:n], 'b-', lw=1, alpha=0.5, label='true')
    axes[0, 1].plot(t, y_pred, 'r--', lw=1.5, label='FD solver')
    axes[0, 1].set_xlabel('t')
    axes[0, 1].set_ylabel('y(t)')
    axes[0, 1].set_title(f'Robot {rid}: y(t)')
    axes[0, 1].legend(fontsize=8)

    axes[1, 0].plot(x_true[:n], y_true[:n], 'b-', lw=0.8, alpha=0.5, label='true')
    axes[1, 0].plot(x_pred, y_pred, 'r--', lw=1, label='FD solver')
    axes[1, 0].plot(x_true[0], y_true[0], 'go', ms=8)
    axes[1, 0].set_xlabel('x')
    axes[1, 0].set_ylabel('y')
    axes[1, 0].set_title(f'Robot {rid}: trajectory')
    axes[1, 0].legend(fontsize=8)

    err_x = x_pred[:n] - x_true[:n]
    err_y = y_pred[:n] - y_true[:n]
    axes[1, 1].plot(t[:n], err_x, 'b-', lw=1, alpha=0.7, label='err_x')
    axes[1, 1].plot(t[:n], err_y, 'r-', lw=1, alpha=0.7, label='err_y')
    axes[1, 1].axhline(0, color='k', lw=0.5)
    axes[1, 1].set_xlabel('t')
    axes[1, 1].set_ylabel('error')
    axes[1, 1].set_title(f'Robot {rid}: solver error')
    axes[1, 1].legend(fontsize=8)

    fig.suptitle(f'MICRO solver: robot {rid}', fontsize=13)
    plt.tight_layout()
    path = out_dir / f'micro_solver_robot_{rid}.png'
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return path


def plot_meso_solver(solve_result: Dict, out_dir: Path) -> Path:
    if solve_result.get("status") != "ok":
        return None
    robot_ids = solve_result["robot_ids"]
    ids_str = '_'.join(str(r) for r in robot_ids)
    variable_names = solve_result["variable_names"]
    pred = solve_result["pred"]
    true = solve_result["true"]
    t = solve_result["t_data"]
    n_vars = len(variable_names)

    n_robots = n_vars // 2
    fig, axes = plt.subplots(n_robots, 2, figsize=(14, 4 * n_robots))
    if n_robots == 1:
        axes = axes[np.newaxis, :]

    for i in range(n_robots):
        xi = 2 * i
        yi = 2 * i + 1
        axes[i, 0].plot(t, true[:, xi], 'b-', lw=1, alpha=0.5, label='true')
        axes[i, 0].plot(t, pred[:, xi], 'r--', lw=1.5, label='FD solver')
        axes[i, 0].set_title(f'{variable_names[xi]}')
        axes[i, 0].legend(fontsize=7)

        axes[i, 1].plot(t, true[:, yi], 'b-', lw=1, alpha=0.5, label='true')
        axes[i, 1].plot(t, pred[:, yi], 'r--', lw=1.5, label='FD solver')
        axes[i, 1].set_title(f'{variable_names[yi]}')
        axes[i, 1].legend(fontsize=7)

    fig.suptitle(f'MESO solver: robots {robot_ids}', fontsize=13)
    plt.tight_layout()
    path = out_dir / f'meso_solver_{ids_str}.png'
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return path


def plot_macro_solver(solve_result: Dict, out_dir: Path,
                      robot_id: int = 0) -> Path:
    if solve_result.get("status") != "ok":
        return None

    sol_u = solve_result["sol_u"]
    sol_v = solve_result["sol_v"]
    x_grid = solve_result["x_grid"]
    y_grid = solve_result["y_grid"]
    n_steps = sol_u.shape[0]

    indices = [0, n_steps // 3, 2 * n_steps // 3, n_steps - 1]
    fig, axes = plt.subplots(2, len(indices), figsize=(5 * len(indices), 10))

    for col, idx in enumerate(indices):
        mag = np.sqrt(sol_u[idx] ** 2 + sol_v[idx] ** 2)
        ext = [x_grid[0], x_grid[-1], y_grid[0], y_grid[-1]] if len(x_grid) > 0 else None
        im = axes[0, col].imshow(mag, origin='lower', extent=ext,
                                  cmap='viridis', interpolation='bilinear')
        axes[0, col].set_title(f'|V| step={idx}')
        plt.colorbar(im, ax=axes[0, col], shrink=0.7)

        im2 = axes[1, col].imshow(sol_u[idx], origin='lower', extent=ext,
                                   cmap='coolwarm', interpolation='bilinear')
        axes[1, col].set_title(f'u step={idx}')
        plt.colorbar(im2, ax=axes[1, col], shrink=0.7)

    fig.suptitle(f'MACRO solver evolution: robot {robot_id}', fontsize=13)
    plt.tight_layout()
    path = out_dir / f'macro_solver_robot_{robot_id}.png'
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return path
