import os
import logging
import subprocess
import os.path as osp

from pathlib import Path


def run_bundle_adjuster(deep_sfm_dir, ba_dir, colmap_path):
    logging.info("Running the bundle adjuster.")

    deep_sfm_model_dir = osp.join(deep_sfm_dir, 'model')
    cmd = [
        str(colmap_path), 'bundle_adjuster',
        '--input_path', str(deep_sfm_model_dir),
        '--output_path', str(ba_dir),
        '--BundleAdjustment.max_num_iterations', '150',
        '--BundleAdjustment.max_linear_solver_iterations', '500',
        '--BundleAdjustment.function_tolerance', '0',
        '--BundleAdjustment.gradient_tolerance', '0',
        '--BundleAdjustment.parameter_tolerance', '0',
        '--BundleAdjustment.refine_focal_length', '0',
        '--BundleAdjustment.refine_principal_point', '0',
        '--BundleAdjustment.refine_extra_params', '0',
        '--BundleAdjustment.refine_extrinsics', '1'
    ]
    logging.info(' '.join(cmd))

    ret = subprocess.call(cmd)
    if ret != 0:
        logging.warning('Problem with point_triangulator, existing.')
        exit(ret)


def main(deep_sfm_dir, ba_dir, colmap_path='colmap'):
    assert Path(deep_sfm_dir).exists(), deep_sfm_dir
              
    Path(ba_dir).mkdir(parents=True, exist_ok=True)
    run_bundle_adjuster(deep_sfm_dir, ba_dir, colmap_path)