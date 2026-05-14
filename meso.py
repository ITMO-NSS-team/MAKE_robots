import numpy as np
import logging
import threading
from typing import List, Dict, Any

import epde
from epde import (EpdeSearch, TrigonometricTokens, GridTokens,
                  CacheStoredTokens)

from equation_parser import EquationProcessor, extract_all_equations

log = logging.getLogger(__name__)


def _build_interaction_tokens(normalized_coord_data: Dict,
                              robot_ids: List[int],
                              cut_number: int,
                              frequency: int,
                              dimensionality: int = 0) -> List:
    tokens = []
    pairs = []
    for i, id_a in enumerate(robot_ids):
        for j, id_b in enumerate(robot_ids):
            if j <= i:
                continue
            pairs.append((id_a, id_b))

    for id_a, id_b in pairs:
        coords_a = np.array(normalized_coord_data[id_a][:cut_number:frequency])
        coords_b = np.array(normalized_coord_data[id_b][:cut_number:frequency])

        dx = coords_a[:, 0] - coords_b[:, 0]
        dy = coords_a[:, 1] - coords_b[:, 1]
        r = np.sqrt(dx ** 2 + dy ** 2 + 1e-8)
        inv_r = 1.0 / (r + 1e-4)

        label_r = f'r_{id_a}_{id_b}'
        label_inv = f'inv_r_{id_a}_{id_b}'
        label_dx = f'dx_{id_a}_{id_b}'
        label_dy = f'dy_{id_a}_{id_b}'

        try:
            tok_r = CacheStoredTokens(
                token_type=f'dist_{id_a}_{id_b}',
                token_labels=[label_r, label_inv],
                token_tensors={label_r: r, label_inv: inv_r},
                params_ranges={'power': (1, 1)},
                params_equality_ranges=None,
                dimensionality=dimensionality,
                meaningful=True,
            )
            tokens.append(tok_r)
        except Exception as e:
            log.warning(f"Failed to create distance token {id_a}-{id_b}: {e}")

        try:
            tok_delta = CacheStoredTokens(
                token_type=f'delta_{id_a}_{id_b}',
                token_labels=[label_dx, label_dy],
                token_tensors={label_dx: dx, label_dy: dy},
                params_ranges={'power': (1, 1)},
                params_equality_ranges=None,
                dimensionality=dimensionality,
                meaningful=True,
            )
            tokens.append(tok_delta)
        except Exception as e:
            log.warning(f"Failed to create delta token {id_a}-{id_b}: {e}")

    return tokens


class MesoDiscoverer:
    def __init__(self,
                 population_size: int = 6,
                 training_epochs: int = 5,
                 poly_window: int = 7,
                 sigma: int = 2,
                 boundary: int = 4,
                 max_deriv_order: int = 2,
                 equation_terms_max_number: int = 3,
                 timeout: int = 600):
        self.population_size = population_size
        self.training_epochs = training_epochs
        self.poly_window = poly_window
        self.sigma = sigma
        self.boundary = boundary
        self.max_deriv_order = max_deriv_order
        self.equation_terms_max_number = equation_terms_max_number
        self.timeout = timeout

    def discover(self,
                 normalized_coord_data: Dict,
                 robot_ids: List[int],
                 cut_number: int = 200,
                 frequency: int = 1) -> Dict[str, Any]:

        n_points = min(
            cut_number,
            min(len(normalized_coord_data[rid]) for rid in robot_ids)
        )
        cut_points = np.zeros((n_points, len(robot_ids), 2))
        for i, rid in enumerate(robot_ids):
            coords = np.array(normalized_coord_data[rid][:n_points:frequency])
            if frequency > 1:
                n_points = len(coords)
            cut_points[:len(coords), i, :] = coords[:n_points]

        t_data = np.arange(n_points // frequency if frequency > 1 else n_points)
        actual_len = len(t_data)

        samples = []
        variable_names = []
        for i, rid in enumerate(robot_ids):
            x = cut_points[:actual_len, i, 0]
            y = cut_points[:actual_len, i, 1]
            samples.append(x)
            samples.append(y)
            variable_names.append(f"x{i + 1}")
            variable_names.append(f"y{i + 1}")

        interaction_tokens = _build_interaction_tokens(
            normalized_coord_data, robot_ids, cut_number, frequency,
            dimensionality=0,
        )

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

                all_tokens = interaction_tokens + [trig_tokens, grid_tokens]

                epde_obj.fit(
                    data=samples,
                    variable_names=variable_names,
                    max_deriv_order=self.max_deriv_order,
                    equation_terms_max_number=self.equation_terms_max_number,
                    data_fun_pow=1,
                    additional_tokens=all_tokens,
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

        ids_str = '_'.join(str(r) for r in robot_ids)

        if t.is_alive():
            log.warning(f"MESO robots {ids_str}: timeout {self.timeout}s")
            return {"robot_ids": robot_ids, "status": "timeout",
                    "equations": [], "variable_names": variable_names}

        if exception_holder[0] is not None:
            log.warning(f"MESO robots {ids_str}: {exception_holder[0]}")
            return {"robot_ids": robot_ids, "status": "error",
                    "error": str(exception_holder[0]),
                    "equations": [], "variable_names": variable_names}

        res = result_holder[0]
        equations = extract_all_equations(res, variable_names)

        processor = EquationProcessor()
        table_main = [{v: [{}, {}]} for v in variable_names]
        k = 0
        vals_flat = []
        for elem in res:
            vals_flat.extend(elem)
        table_main, k = processor.object_table(
            [vals_flat], variable_names, table_main, k
        )
        coeff_table = processor.preprocessing_table(variable_names, table_main, k)

        log.info(f"MESO robots {ids_str}: OK, {len(equations)} equations found")

        return {
            "robot_ids": robot_ids,
            "status": "ok",
            "equations": equations,
            "coeff_table": coeff_table,
            "raw_results": res,
            "variable_names": variable_names,
            "samples": samples,
            "t_data": t_data,
        }


def run_meso(normalized_coord_data: Dict,
             robot_groups: List[List[int]],
             cut_number: int = 200,
             frequency: int = 1,
             **kwargs) -> List[Dict]:
    discoverer = MesoDiscoverer(**kwargs)
    results = []
    for group in robot_groups:
        valid_ids = [r for r in group if r in normalized_coord_data]
        if len(valid_ids) < 2:
            log.warning(f"Group {group}: <2 valid robots, skipping")
            continue
        ids_str = '_'.join(str(r) for r in valid_ids)
        log.info(f"MESO: discovering system for robots [{ids_str}] ...")
        result = discoverer.discover(
            normalized_coord_data, valid_ids,
            cut_number=cut_number, frequency=frequency,
        )
        results.append(result)
    return results
