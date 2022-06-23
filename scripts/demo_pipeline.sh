#!/bin/bash
PROJECT_DIR="$(pwd)"
OBJ_NAME=$1
echo "Current work dir: $PROJECT_DIR"

echo '-------------------'
echo 'Parse scanned data:'
echo '-------------------'
# Parse scanned annotated & test sequence:
python $PROJECT_DIR/parse_scanned_data.py \
    --scanned_object_path \
    "$PROJECT_DIR/data/demo/$OBJ_NAME"

echo '--------------------------------------------------------------'
echo 'Run SfM to reconstruct object point cloud for pose estimation:'
echo '--------------------------------------------------------------'
# Run SfM to reconstruct object sparse point cloud from $OBJ_NAME-annotate sequence:
python $PROJECT_DIR/run.py \
    +preprocess="sfm_spp_spg_demo" \
    dataset.data_dir="$PROJECT_DIR/data/demo/$OBJ_NAME $OBJ_NAME-annotate" \
    dataset.outputs_dir="$PROJECT_DIR/data/demo/$OBJ_NAME/sfm_model" \

echo "-----------------------------------"
echo "Run inference and output demo video:"
echo "-----------------------------------"

WITH_TRACKING=False
while [[ "$#" -gt 0 ]]; do
    case $1 in
        -u|--WITH_TRACKING) WITH_TRACKING=True ;;
    esac
    shift
done

# Run inference on $OBJ_NAME-test and output demo video:
python $PROJECT_DIR/inference_demo.py \
    +experiment="test_demo" \
    input.data_dirs="$PROJECT_DIR/data/demo/$OBJ_NAME $OBJ_NAME-test" \
    input.sfm_model_dirs="$PROJECT_DIR/data/demo/$OBJ_NAME/sfm_model" \
    use_tracking=${WITH_TRACKING}
