import numpy as np
import logging
import threading
from typing import Dict, Any, Optional
from scipy import stats, ndimage

import epde
from epde import EpdeSearch, TrigonometricTokens

from equation_parser import extract_all_equations

log = logging.getLogger(__name__)


def bin_snapshot(frames, start, end, bins=30):
    positions = frames[start + 1:end + 1]
    displacements = np.diff(frames[start:end + 1], axis=0)

    xp = positions[:, :, 0].flatten()
    yp = positions[:, :, 1].flatten()

    results = []
    xe = ye = None
    for axis in range(2):
        mag = displacements[:, :, axis].flatten()
        hmap, xe, ye, _ = stats.binned_statistic_2d(
            xp, yp, mag, statistic='mean', bins=bins
        )
        hmap = ndimage.gaussian_filter(
            np.nan_to_num(hmap, nan=0.0), sigma=1.5
        ).T
        results.append(hmap.astype(np.float32))

    return results[0], results[1], xe, ye


class MacroDiscoverer:
    def __init__(self,
                 n_windows: int = 3,
                 bins: int = 25,
                 population_size: int = 6,
                 training_epochs: int = 5,
                 max_terms: int = 5,
                 max_factors: int = 1,
                 timeout: int = 600):
        self.n_windows = n_windows
        self.bins = bins
        self.population_size = population_size
        self.training_epochs = training_epochs
        self.max_terms = max_terms
        self.max_factors = max_factors
        self.timeout = timeout

    def discover(self, frames: np.ndarray,
                 robot_idx: int = 0) -> Dict[str, Any]:
        total_frames = frames.shape[0] - 1
        single_frames = frames[:, robot_idx:robot_idx + 1, :]

        window_size = total_frames // self.n_windows
        snap_u_list = []
        snap_v_list = []
        t_centres = []
        xe = ye = None

        for i in range(self.n_windows):
            s = i * window_size
            e = (i + 1) * window_size
            hu, hv, xe, ye = bin_snapshot(single_frames, s, e, bins=self.bins)
            snap_u_list.append(hu)
            snap_v_list.append(hv)
            t_centres.append((s + e) / 2.0 / total_frames)

        x_epde = ((xe[:-1] + xe[1:]) / 2).astype(np.float32)
        y_epde = ((ye[:-1] + ye[1:]) / 2).astype(np.float32)
        t_epde = np.array(t_centres, dtype=np.float32)

        data_u = np.stack(snap_u_list).astype(np.float32)
        data_v = np.stack(snap_v_list).astype(np.float32)

        tg, xg, yg = np.meshgrid(t_epde, x_epde, y_epde, indexing='ij')

        epde_hmaps = list(zip(snap_u_list, snap_v_list))

        result_holder = [None]
        exception_holder = [None]

        def _worker():
            try:
                obj = EpdeSearch(
                    use_solver=False,
                    coordinate_tensors=(tg, xg, yg),
                )
                obj.set_preprocessor(
                    default_preprocessor_type='FD',
                    preprocessor_kwargs={},
                )
                obj.set_moeadd_params(
                    population_size=self.population_size,
                    training_epochs=self.training_epochs,
                )

                trig_tokens = TrigonometricTokens(
                    freq=(0.5, 2.0), dimensionality=2
                )

                obj.fit(
                    data=[data_u, data_v],
                    variable_names=['u', 'v'],
                    max_deriv_order=(1, 2, 2),
                    equation_terms_max_number=self.max_terms,
                    equation_factors_max_number=self.max_factors,
                    additional_tokens=[trig_tokens],
                    eq_sparsity_interval=(1e-9, 1),
                )

                res = obj.equations(
                    only_print=False, only_str=False, num=4
                )
                result_holder[0] = res
            except Exception as exc:
                exception_holder[0] = exc

        t = threading.Thread(target=_worker, daemon=True)
        t.start()
        t.join(timeout=self.timeout)

        if t.is_alive():
            log.warning(f"MACRO robot_idx={robot_idx}: timeout {self.timeout}s")
            return {
                "robot_idx": robot_idx, "status": "timeout",
                "equations": [], "epde_hmaps": epde_hmaps,
                "x_epde": x_epde, "y_epde": y_epde,
                "t_epde": t_epde, "data_u": data_u, "data_v": data_v,
            }

        if exception_holder[0] is not None:
            log.warning(f"MACRO robot_idx={robot_idx}: {exception_holder[0]}")
            return {
                "robot_idx": robot_idx, "status": "error",
                "error": str(exception_holder[0]),
                "equations": [], "epde_hmaps": epde_hmaps,
                "x_epde": x_epde, "y_epde": y_epde,
                "t_epde": t_epde, "data_u": data_u, "data_v": data_v,
            }

        res = result_holder[0]
        equations = extract_all_equations(res, ['u', 'v'])

        best_texts = {}
        for eq_info in equations:
            if eq_info["level"] == 0 and eq_info["solution"] == 0:
                best_texts[eq_info["variable"]] = eq_info["text_form"]

        log.info(f"MACRO robot_idx={robot_idx}: OK, {len(equations)} equations")

        return {
            "robot_idx": robot_idx,
            "status": "ok",
            "equations": equations,
            "best_u": best_texts.get("u", ""),
            "best_v": best_texts.get("v", ""),
            "raw_results": res,
            "epde_hmaps": epde_hmaps,
            "x_epde": x_epde,
            "y_epde": y_epde,
            "t_epde": t_epde,
            "data_u": data_u,
            "data_v": data_v,
        }


def run_macro(frames: np.ndarray,
              robot_idx: int = 0,
              **kwargs) -> Dict[str, Any]:
    discoverer = MacroDiscoverer(**kwargs)
    log.info(f"MACRO: discovering PDE for robot index {robot_idx} ...")
    return discoverer.discover(frames, robot_idx=robot_idx)
