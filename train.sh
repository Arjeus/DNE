set -e

# Unset distributed training environment variables to force single GPU mode
unset RANK
unset WORLD_SIZE
unset LOCAL_RANK
unset SLURM_PROCID

echo "Launching training on GPU..."
python main.py \
--options options/data/cifar100_50-10.yaml options/data/cifar100_order1.yaml options/model/cifar_dne.yaml \
    --name dne_mlp_dense_only_cifar100_b50_10 \
    --data-path ./data/CIFAR100/ \
    --log-path ./logs \
    --device cuda:0 \
    --output-basedir ./checkpoints --extra-dim 64 --extra-heads 1
