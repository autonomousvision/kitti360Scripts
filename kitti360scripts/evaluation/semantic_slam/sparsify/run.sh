
input_dir=INPUT_DIR
output_dir=INPUT_DIR_DOWNSAMPLE
mkdir -p $output_dir

test_ids=("0" "1" "2" "3")

for test_id in "${test_ids[@]}"
do

## downsample pred
./build/sparsify --input_path ${input_dir}/test_${test_id}.ply \
      --input_semantic_path ${input_dir}/test_semantic_${test_id}.txt \
      --output_path ${output_dir}/test_${test_id}.ply \
      --output_semantic_path ${output_dir}/test_semantic_${test_id}.txt \
      --min_dist 0.075 \
      --output_mask_path ${output_dir}/test_mask_${test_id}.txt

done
