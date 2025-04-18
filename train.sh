set -e

echo "Launching training on GPU..."
# Run the training with explicit device assignment and 
# Note: using underscore instead of hyphen for dist_url


python -u main.py \
    --options options/data/cifar100_50-10.yaml options/data/cifar100_order1.yaml options/model/cifar_dne.yaml \
    --name dne_64dim_digits \
    --data-path ./data/CIFAR100/ \
    --log-path ./logs \
    --device cuda:0 \
    --output-basedir ./checkpoints \
    --extra-dim 32 \
    --extra-heads 1 \
    --dist_url 'env://' \
    --no-distributed \
    --base-epochs 10 \
    --epochs 10
