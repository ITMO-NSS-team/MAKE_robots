import re
import logging
import numpy as np
import torch
from pathlib import Path
from typing import Dict, List, Any, Tuple

log = logging.getLogger(__name__)


def _apply_term_1d(name: str, x: np.ndarray, y: np.ndarray,
                   t: np.ndarray, dt: float) -> np.ndarray:
    n = name.lower().strip()
    if n in ('c', 'const', '1'):
        return np.ones_like(x)
    if 'x{power: 2' in n or 'x^2' in n:
        return x ** 2
    if 'y{power: 2' in n or 'y^2' in n:
        return y ** 2
    if 'x{power: 1' in n or n == 'x':
        return x.copy()
    if 'y{power: 1' in n or n == 'y':
        return y.copy()
    if 'd^2x/dx0^2' in n:
        return np.gradient(np.gradient(x, dt), dt)
    if 'dx/dx0' in n:
        return np.gradient(x, dt)
    if 'd^2y/dx0^2' in n:
        return np.gradient(np.gradient(y, dt), dt)
    if 'dy/dx0' in n:
        return np.gradient(y, dt)
    if 't{power: 2' in n:
        return t ** 2
    if 't{power: 1' in n or n == 't':
        return t.copy()
    if 'sin' in n:
        return np.sin(t)
    if 'cos' in n:
        return np.cos(t)
    if 'x' in n and 'y' in n:
        return x * y
    return np.ones_like(x)


def _apply_term_2d(name: str, field: np.ndarray,
                   dx: float, dy: float) -> np.ndarray:
    n = name.lower().strip()
    if n in ('u', 'v', '1', 'const'):
        return field.copy()
    if 'du/dx1' in n or 'dv/dx1' in n or 'du/dx' in n or 'dv/dx' in n:
        return np.gradient(field, dx, axis=1)
    if 'du/dx2' in n or 'dv/dx2' in n or 'du/dy' in n or 'dv/dy' in n:
        return np.gradient(field, dy, axis=0)
    if 'd^2' in n and 'dx1' in n and 'dx2' not in n:
        return np.gradient(np.gradient(field, dx, axis=1), dx, axis=1)
    if 'd^2' in n and 'dx2' in n and 'dx1' not in n:
        return np.gradient(np.gradient(field, dy, axis=0), dy, axis=0)
    if 'dxdy' in n or ('dx1' in n and 'dx2' in n):
        return np.gradient(np.gradient(field, dx, axis=1), dy, axis=0)
    if 'sin' in n:
        return np.sin(field)
    if 'cos' in n:
        return np.cos(field)
    return field.copy()


def _parse_coefficients(equations: List[Dict]) -> Dict[str, Dict[str, float]]:
    coeffs = {}
    for eq_info in equations:
        if eq_info.get("level") != 0 or eq_info.get("solution") != 0:
            continue
        var = eq_info["variable"]
        coeffs[var] = {}
        for term in eq_info.get("terms", []):
            coeffs[var][term["term"]] = term["coefficient"]
    return coeffs


def solve_micro_fd(result: Dict, n_steps: int = None) -> Dict:
    if result.get("status") != "ok":
        return {"status": "no_equations"}

    x_data = result["x_data"]
    y_data = result["y_data"]
    t_data = result["t_data"]
    rid = result["robot_id"]

    coeffs = _parse_coefficients(result["equations"])
    if "x" not in coeffs or "y" not in coeffs:
        return {"status": "no_coefficients"}

    dt = float(t_data[1] - t_data[0]) if len(t_data) > 1 else 1.0
    n = len(t_data)
    if n_steps is None:
        n_steps = n - 1

    x_pred = np.zeros(n_steps + 1)
    y_pred = np.zeros(n_steps + 1)
    x_pred[0] = x_data[0]
    y_pred[0] = y_data[0]

    x_cap = 5.0 * max(np.abs(x_data).max(), 1.0)
    y_cap = 5.0 * max(np.abs(y_data).max(), 1.0)

    for step in range(n_steps):
        t_arr = np.array([t_data[min(step, n - 1)]])
        x_arr = np.array([x_pred[step]])
        y_arr = np.array([y_pred[step]])

        rhs_x = 0.0
        for term, c in coeffs["x"].items():
            rhs_x += c * _apply_term_1d(term, x_arr, y_arr, t_arr, dt)[0]
        rhs_y = 0.0
        for term, c in coeffs["y"].items():
            rhs_y += c * _apply_term_1d(term, x_arr, y_arr, t_arr, dt)[0]

        safe_dt = dt
        mx = max(abs(rhs_x), abs(rhs_y), 1e-10)
        safe_dt = min(dt, 0.1 * min(x_cap, y_cap) / mx)

        x_pred[step + 1] = np.clip(x_pred[step] + safe_dt * rhs_x, -x_cap, x_cap)
        y_pred[step + 1] = np.clip(y_pred[step] + safe_dt * rhs_y, -y_cap, y_cap)

        if not (np.isfinite(x_pred[step + 1]) and np.isfinite(y_pred[step + 1])):
            x_pred[step + 1] = x_pred[step]
            y_pred[step + 1] = y_pred[step]

    t_pred = t_data[:n_steps + 1] if n_steps + 1 <= n else np.arange(n_steps + 1) * dt

    return {
        "status": "ok",
        "robot_id": rid,
        "x_pred": x_pred,
        "y_pred": y_pred,
        "t_pred": t_pred,
        "x_true": x_data,
        "y_true": y_data,
    }


def solve_meso_fd(result: Dict, normalized_coord_data: Dict,
                  cut_number: int = 200) -> Dict:
    if result.get("status") != "ok":
        return {"status": "no_equations"}

    robot_ids = result["robot_ids"]
    coeffs = _parse_coefficients(result["equations"])
    variable_names = result["variable_names"]
    samples = result["samples"]
    t_data = result["t_data"]
    dt = float(t_data[1] - t_data[0]) if len(t_data) > 1 else 1.0
    n = len(t_data)

    n_vars = len(variable_names)
    pred = np.zeros((n, n_vars))
    for vi in range(n_vars):
        pred[0, vi] = samples[vi][0]

    cap = 5.0 * max(np.abs(np.array(samples)).max(), 1.0)

    for step in range(n - 1):
        t_arr = np.array([t_data[step]])
        state = {variable_names[vi]: np.array([pred[step, vi]])
                 for vi in range(n_vars)}

        for vi, vn in enumerate(variable_names):
            if vn not in coeffs:
                pred[step + 1, vi] = pred[step, vi]
                continue
            rhs = 0.0
            x_arr = state.get(vn, np.array([pred[step, vi]]))
            y_idx = vi + 1 if vi % 2 == 0 else vi - 1
            y_name = variable_names[min(y_idx, n_vars - 1)]
            y_arr = state.get(y_name, np.array([0.0]))
            for term, c in coeffs[vn].items():
                rhs += c * _apply_term_1d(term, x_arr, y_arr, t_arr, dt)[0]
            pred[step + 1, vi] = np.clip(pred[step, vi] + dt * rhs, -cap, cap)
            if not np.isfinite(pred[step + 1, vi]):
                pred[step + 1, vi] = pred[step, vi]

    return {
        "status": "ok",
        "robot_ids": robot_ids,
        "variable_names": variable_names,
        "pred": pred,
        "true": np.column_stack(samples),
        "t_data": t_data,
    }


def solve_macro_fd(result: Dict, n_steps: int = 50,
                   dt_max: float = 0.01) -> Dict:
    if result.get("status") != "ok":
        return {"status": "no_equations"}

    coeffs = _parse_coefficients(result["equations"])
    epde_hmaps = result["epde_hmaps"]
    x_epde = result["x_epde"]
    y_epde = result["y_epde"]

    ic_u, ic_v = epde_hmaps[0]
    dx = float(x_epde[1] - x_epde[0]) if len(x_epde) > 1 else 1.0
    dy = float(y_epde[1] - y_epde[0]) if len(y_epde) > 1 else 1.0

    u = ic_u.astype(np.float64).copy()
    v = ic_v.astype(np.float64).copy()
    sol_u = [u.copy()]
    sol_v = [v.copy()]

    u_cap = max(5.0 * np.abs(ic_u).max(), 1.0)
    v_cap = max(5.0 * np.abs(ic_v).max(), 1.0)

    coeffs_u = coeffs.get("u", {})
    coeffs_v = coeffs.get("v", {})

    if coeffs_u:
        rhs_test = np.zeros_like(u)
        for term, c in coeffs_u.items():
            rhs_test += c * _apply_term_2d(term, u, dx, dy)
        rhs_scale = np.abs(rhs_test).mean()
        ic_scale = np.abs(ic_u).mean() + 1e-12
        if rhs_scale > 1e-12:
            factor = float(np.clip(ic_scale / rhs_scale, 0.01, 100.0))
            coeffs_u = {k: v * factor for k, v in coeffs_u.items()}
            coeffs_v = {k: v * factor for k, v in coeffs_v.items()}

    for step in range(n_steps):
        du = np.zeros_like(u)
        for term, c in coeffs_u.items():
            du += c * _apply_term_2d(term, u, dx, dy)
        dv = np.zeros_like(v)
        for term, c in coeffs_v.items():
            dv += c * _apply_term_2d(term, v, dx, dy)

        mx_u = max(np.abs(du).max(), 1e-10)
        mx_v = max(np.abs(dv).max(), 1e-10)
        safe_dt = min(dt_max, 0.1 * u_cap / mx_u, 0.1 * v_cap / mx_v)

        u = np.clip(u + safe_dt * du, -u_cap, u_cap)
        v = np.clip(v + safe_dt * dv, -v_cap, v_cap)

        if not (np.isfinite(u).all() and np.isfinite(v).all()):
            u = sol_u[-1].copy()
            v = sol_v[-1].copy()

        sol_u.append(u.copy())
        sol_v.append(v.copy())

    sol_u = np.stack(sol_u).astype(np.float32)
    sol_v = np.stack(sol_v).astype(np.float32)
    t_sol = np.linspace(0, 1.0, n_steps + 1, dtype=np.float32)

    return {
        "status": "ok",
        "sol_u": sol_u,
        "sol_v": sol_v,
        "t_sol": t_sol,
        "x_grid": x_epde,
        "y_grid": y_epde,
        "ic_u": ic_u,
        "ic_v": ic_v,
    }


def save_micro_pt(solve_result: Dict, out_dir: Path):
    if solve_result.get("status") != "ok":
        return
    rid = solve_result["robot_id"]
    path = out_dir / f"micro_robot_{rid}.pt"
    torch.save({
        "x_pred": torch.from_numpy(solve_result["x_pred"].astype(np.float32)),
        "y_pred": torch.from_numpy(solve_result["y_pred"].astype(np.float32)),
        "x_true": torch.from_numpy(solve_result["x_true"].astype(np.float32)),
        "y_true": torch.from_numpy(solve_result["y_true"].astype(np.float32)),
        "t": torch.from_numpy(solve_result["t_pred"].astype(np.float32)),
    }, path)
    log.info(f"Saved {path}")


def save_meso_pt(solve_result: Dict, out_dir: Path):
    if solve_result.get("status") != "ok":
        return
    ids_str = '_'.join(str(r) for r in solve_result["robot_ids"])
    path = out_dir / f"meso_robots_{ids_str}.pt"
    torch.save({
        "pred": torch.from_numpy(solve_result["pred"].astype(np.float32)),
        "true": torch.from_numpy(solve_result["true"].astype(np.float32)),
        "t": torch.from_numpy(solve_result["t_data"].astype(np.float32)),
        "variable_names": solve_result["variable_names"],
    }, path)
    log.info(f"Saved {path}")


def save_macro_pt(solve_result: Dict, out_dir: Path, robot_id: int = 0):
    if solve_result.get("status") != "ok":
        return
    path = out_dir / f"macro_robot_{robot_id}.pt"
    torch.save({
        "u": torch.from_numpy(solve_result["sol_u"]),
        "v": torch.from_numpy(solve_result["sol_v"]),
        "t": torch.from_numpy(solve_result["t_sol"]),
        "x": torch.from_numpy(solve_result["x_grid"].astype(np.float32)),
        "y": torch.from_numpy(solve_result["y_grid"].astype(np.float32)),
        "ic_u": torch.from_numpy(solve_result["ic_u"].astype(np.float32)),
        "ic_v": torch.from_numpy(solve_result["ic_v"].astype(np.float32)),
    }, path)
    log.info(f"Saved {path}")
