import re
import itertools
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Any


class EquationProcessor:
    def __init__(self):
        self.regex = re.compile(r', freq:\s\d\S\d+')

    @staticmethod
    def dict_update(d_main: Dict, term: str, coeff: float, k: int) -> Dict:
        str_t = '_r' if '_r' in term else ''
        arr_term = re.sub('_r', '', term).split(' * ')
        perm_set = list(itertools.permutations(range(len(arr_term))))
        structure_added = False

        for p_i in perm_set:
            temp = " * ".join([arr_term[i] for i in p_i]) + str_t
            if temp in d_main:
                if k - len(d_main[temp]) >= 0:
                    d_main[temp] += [0 for _ in range(k - len(d_main[temp]))] + [coeff]
                else:
                    d_main[temp][-1] += coeff
                structure_added = True

        if not structure_added:
            d_main[term] = [0 for _ in range(k)] + [coeff]

        return d_main

    def equation_table(self, k: int, equation, dict_main: Dict, dict_right: Dict) -> List[Dict]:
        equation_s = equation.structure
        equation_c = equation.weights_final
        text_form_eq = self.regex.sub('', equation.text_form)

        flag = False
        for t_eq in equation_s:
            term = self.regex.sub('', t_eq.name)
            for t in range(len(equation_c)):
                c = equation_c[t]
                if f'{c} * {term} +' in text_form_eq:
                    dict_main = self.dict_update(dict_main, term, c, k)
                    equation_c = np.delete(equation_c, t)
                    break
                elif f'+ {c} =' in text_form_eq:
                    dict_main = self.dict_update(dict_main, "C", c, k)
                    equation_c = np.delete(equation_c, t)
                    break
            if f'= {term}' == text_form_eq[text_form_eq.find('='):] and not flag:
                flag = True
                dict_main = self.dict_update(dict_main, term, -1., k)

        return [dict_main, dict_right]

    def object_table(self, res: List, variable_names: List[str],
                     table_main: List[Dict], k: int) -> Tuple[List[Dict], int]:
        for list_SoEq in res:
            for SoEq in list_SoEq:
                for n, value in enumerate(variable_names):
                    gene = SoEq.vals.chromosome.get(value)
                    if gene is not None:
                        table_main[n][value] = self.equation_table(
                            k, gene.value, *table_main[n][value]
                        )
                k += 1
        return table_main, k

    def preprocessing_table(self, variable_names: List[str],
                            table_main: List[Dict], k: int) -> pd.DataFrame:
        data_frame_total = pd.DataFrame()

        for dict_var in table_main:
            for var_name, list_structure in dict_var.items():
                general_dict = {}
                for structure in list_structure:
                    general_dict.update(structure)
                dict_var[var_name] = general_dict

        for dict_var in table_main:
            for var_name, general_dict in dict_var.items():
                for key, value in general_dict.items():
                    if len(value) < k:
                        general_dict[key] = value + [0. for _ in range(k - len(value))]

        data_frame_main = [{i: pd.DataFrame()} for i in variable_names]
        for n, dict_var in enumerate(table_main):
            for var_name, general_dict in dict_var.items():
                data_frame_main[n][var_name] = pd.DataFrame(general_dict)

        for n, var_name in enumerate(variable_names):
            data_frame_temp = data_frame_main[n].get(var_name)
            if data_frame_temp is not None and not data_frame_temp.empty:
                list_columns = [f'{col}_{var_name}' for col in data_frame_temp.columns]
                data_frame_temp = data_frame_temp.copy()
                data_frame_temp.columns = list_columns
                data_frame_total = pd.concat([data_frame_total, data_frame_temp], axis=1)

        return data_frame_total


def parse_text_form(text_form: str) -> List[Dict]:
    regex = re.compile(r', freq:\s\d\S\d+')
    clean = regex.sub('', text_form)
    terms = []
    pattern = re.compile(r'([+-]?\s*[\d.eE+-]+)\s*\*\s*([^+\-=]+)')
    for m in pattern.finditer(clean):
        coeff = float(m.group(1).replace(' ', ''))
        term_name = m.group(2).strip()
        terms.append({"term": term_name, "coefficient": coeff})

    rhs_match = re.search(r'=\s*(.+)$', clean)
    if rhs_match:
        rhs = rhs_match.group(1).strip()
        if not any(t["term"] == rhs for t in terms):
            terms.append({"term": rhs, "coefficient": -1.0})

    return terms


def extract_all_equations(epde_results, variable_names):
    all_equations = []
    for level_idx, level in enumerate(epde_results):
        for sol_idx, solution in enumerate(level):
            for var_name in variable_names:
                try:
                    eq = solution.vals[var_name]
                    all_equations.append({
                        "level": level_idx,
                        "solution": sol_idx,
                        "variable": var_name,
                        "text_form": eq.text_form,
                        "latex_form": eq.latex_form,
                        "terms": parse_text_form(eq.text_form),
                    })
                except Exception:
                    pass
    return all_equations
