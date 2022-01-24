# RNNP
Robust Nearest Neighbour Prototype

## Run
python3 run_rnnp_40pt_corrupt_protonet_git.py --model_class ProtoNet --backbone_class Res12 --dataset MiniImageNet --eval_way 5 --eval_shot 10 --eval_query 15 --temperature 64 --step_size 40 --init_weights <path to model> --use_euclidean  --num_eval_episodes 1000 --alpha 0.9 --nummix 4 --gpu 5
