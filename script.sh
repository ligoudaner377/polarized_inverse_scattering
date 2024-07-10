set -ex

name="4pol_minmax_reconstruct"
dataroot="./datasets/translucent"
model="pol_shape_illu"
checkpoints_dir="./checkpoints"
results_dir="./results"
input_list="4pol min max"
python ./train.py --dataroot ${dataroot} --name ${name} --model ${model}  --checkpoints_dir ${checkpoints_dir} --input_list ${input_list} --gpu_ids 0,1,2,3
model="pol_sss"
python ./train.py --dataroot ${dataroot} --name ${name} --model ${model}  --checkpoints_dir ${checkpoints_dir} --use_reconstruction_loss --input_list ${input_list}  --gpu_ids 0,1,2,3
python ./test.py --dataroot ${dataroot} --name ${name} --model ${model} --results_dir ${results_dir} --eval --input_list ${input_list}
python ./inference_real.py --dataroot ${dataroot} --dataset_mode 'real' --name ${name} --model ${model} --results_dir ${results_dir} --input_list ${input_list} --eval 