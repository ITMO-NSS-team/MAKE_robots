import numpy as np
import logging
import threading
from typing import List, Dict, Any, Optional

import epde
from epde import EpdeSearch, TrigonometricTokens, GridTokens

from equation_parser import extract_all_equations

log = logging.getLogger(__name__)


class MicroDiscoverer:
    def __init__(self,
                 population_size: int = 6,
                 training_epochs: int = 5,
                 poly_window: int = 7,
                 sigma: int = 2,
                 boundary: int = 4,
                 max_deriv_order: int = 2,
                 equation_terms_max_number: int = 3,
                 timeout: int = 300):
        self.population_size = population_size
        self.training_epochs = training_epochs
        self.poly_window = poly_window
        self.sigma = sigma
        self.boundary = boundary
        self.max_deriv_order = max_deriv_order
        self.equation_terms_max_number = equation_terms_max_number
        self.timeout = timeout

    def discover(self, x: np.ndarray, y: np.ndarray,
                 robot_id: int = 0, cut_number: int = 200,
                 frequency: int = 1) -> Dict[str, Any]:
        x = x[:cut_number:frequency]
        y = y[:cut_number:frequency]
        t_data = np.arange(len(x))

        samples = [x, y]
        variable_names = ['x', 'y']

        result_holder = [None]
        exception_holder = [None]

        def _worker():
            try:
                epde_obj = EpdeSearch(
                    use_solver=False,
                    boundary=self.boundary,
                    coordinate_tensors=[t_data],
                )
                epde_obj.set_moeadd_params(
                    population_size=self.population_size,
                    training_epochs=self.training_epochs,
                )
                epde_obj.set_preprocessor(
                    default_preprocessor_type='poly',
                    preprocessor_kwargs={
                        'use_smoothing': True,
                        'sigma': self.sigma,
                        'polynomial_window': self.poly_window,
                    }
                )

                trig_tokens = TrigonometricTokens(
                    freq=(0.999, 1.001), dimensionality=0
                )
                grid_tokens = GridTokens(['t'], dimensionality=0)

                factors_max_number = {
                    'factors_num': [1, 2],
                    'probas': [0.85, 0.15]
                }

                epde_obj.fit(
                    data=samples,
                    variable_names=variable_names,
                    max_deriv_order=self.max_deriv_order,
                    equation_terms_max_number=self.equation_terms_max_number,
                    data_fun_pow=1,
                    additional_tokens=[trig_tokens, grid_tokens],
                    equation_factors_max_number=factors_max_number,
                    eq_sparsity_interval=(1e-6, 1),
                )

                res = epde_obj.equations(
                    only_print=False, only_str=False, num=4
                )
                result_holder[0] = res
            except Exception as exc:
                exception_holder[0] = exc

        t = threading.Thread(target=_worker, daemon=True)
        t.start()
        t.join(timeout=self.timeout)

        if t.is_alive():
            log.warning(f"MICRO robot {robot_id}: timeout {self.timeout}s")
            return {"robot_id": robot_id, "status": "timeout", "equations": []}

        if exception_holder[0] is not None:
            log.warning(f"MICRO robot {robot_id}: {exception_holder[0]}")
            return {"robot_id": robot_id, "status": "error",
                    "error": str(exception_holder[0]), "equations": []}

        res = result_holder[0]
        equations = extract_all_equations(res, variable_names)

        best_texts = {}
        for eq_info in equations:
            if eq_info["level"] == 0 and eq_info["solution"] == 0:
                best_texts[eq_info["variable"]] = eq_info["text_form"]

        log.info(f"MICRO robot {robot_id}: OK, {len(equations)} equations found")

        return {
            "robot_id": robot_id,
            "status": "ok",
            "equations": equations,
            "best_x": best_texts.get("x", ""),
            "best_y": best_texts.get("y", ""),
            "raw_results": res,
            "t_data": t_data,
            "x_data": x,
            "y_data": y,
        }


def run_micro_batch(normalized_coord_data: Dict,
                    robot_ids: List[int],
                    cut_number: int = 200,
                    frequency: int = 1,
                    **kwargs) -> List[Dict]:
    discoverer = MicroDiscoverer(**kwargs)
    results = []
    for rid in robot_ids:
        if rid not in normalized_coord_data:
            log.warning(f"Robot {rid} not found in data")
            continue
        coords = np.array(normalized_coord_data[rid])
        x = coords[:, 0]
        y = coords[:, 1]
        log.info(f"MICRO: discovering ODE for robot {rid} ...")
        result = discoverer.discover(x, y, robot_id=rid,
                                     cut_number=cut_number,
                                     frequency=frequency)
        results.append(result)
    return results
