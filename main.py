import argparse
import logging
import json
from pathlib import Path

from data_loader import (load_pickle, extract_coord_data,
                         normalize_all_coordinates, build_frames_array,
                         get_robot_ids)
from micro import run_micro_batch
from meso import run_meso
from macro import run_macro
from plots import (plot_micro_single, plot_micro_summary,
                   plot_meso, plot_macro)
from export_xlsx import export_all

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("pickle_path", type=str)
    p.add_argument("--out", type=str, default="./results")
    p.add_argument("--micro_robots", nargs='+', type=int, default=None)
    p.add_argument("--meso_groups", type=str, default=None,
                   help='JSON: [[78,79],[83,84]]')
    p.add_argument("--macro_robot_idx", type=int, default=0)
    p.add_argument("--cut_number", type=int, default=200)
    p.add_argument("--frequency", type=int, default=1)
    p.add_argument("--population_size", type=int, default=6)
    p.add_argument("--training_epochs", type=int, default=5)
    p.add_argument("--timeout", type=int, default=600)
    p.add_argument("--levels_json", type=str, default=None,
                   help='Path to levels_robots_ids.json')
    return p.parse_args()


def main():
    args = parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    log.info(f"Loading data from {args.pickle_path}")
    raw_data = load_pickle(args.pickle_path)
    coord_data = extract_coord_data(raw_data)
    normalized = normalize_all_coordinates(coord_data)
    frames = build_frames_array(raw_data)
    all_ids = get_robot_ids(raw_data)
    log.info(f"Loaded {len(all_ids)} robots, {frames.shape[0]} frames")

    epde_kwargs = dict(
        population_size=args.population_size,
        training_epochs=args.training_epochs,
        timeout=args.timeout,
    )

    micro_ids = args.micro_robots
    if micro_ids is None:
        micro_ids = all_ids[:3]

    meso_groups = None
    if args.meso_groups:
        meso_groups = json.loads(args.meso_groups)
    elif args.levels_json:
        with open(args.levels_json) as f:
            levels = json.load(f)
        meso_groups = levels.get('meso_cluster',
                                levels.get('meso_union',
                                [[all_ids[0], all_ids[1]]]))
    else:
        meso_groups = [micro_ids[:2]] if len(micro_ids) >= 2 else []

    log.info(f"=== MICRO: robots {micro_ids} ===")
    micro_results = run_micro_batch(
        normalized, micro_ids,
        cut_number=args.cut_number,
        frequency=args.frequency,
        **epde_kwargs,
    )
    for res in micro_results:
        plot_micro_single(res, out_dir)
    if micro_results:
        plot_micro_summary(micro_results, out_dir)

    log.info(f"=== MESO: groups {meso_groups} ===")
    meso_results = run_meso(
        normalized, meso_groups,
        cut_number=args.cut_number,
        frequency=args.frequency,
        **epde_kwargs,
    )
    for res in meso_results:
        plot_meso(res, out_dir)

    log.info(f"=== MACRO: robot_idx={args.macro_robot_idx} ===")
    macro_result = run_macro(
        frames,
        robot_idx=args.macro_robot_idx,
        **epde_kwargs,
    )
    macro_robot_id = all_ids[args.macro_robot_idx] if args.macro_robot_idx < len(all_ids) else args.macro_robot_idx
    plot_macro(macro_result, out_dir, robot_id=macro_robot_id)

    xlsx_path = out_dir / 'equations.xlsx'
    log.info(f"Saving equations to {xlsx_path}")
    export_all(micro_results, meso_results, macro_result, xlsx_path)

    log.info(f"Done. Results in {out_dir}")


if __name__ == '__main__':
    main()
