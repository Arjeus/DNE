set -e

echo "Launching training on GPU..."
# Run the training with explicit device assignment and 
# Note: using underscore instead of hyphen for dist_url


python main.py \
    --options options/data/cifar100_50-10.yaml options/data/cifar100_order1.yaml options/model/cifar_dne.yaml \
    --name dne_64dim_digits \
    --data-path ./data/CIFAR100/ \
    --log-path ./logs \
    --device cuda:0 \
    --output-basedir ./checkpoints \
    --extra-dim 64 \
    --extra-heads 2 \
    --dist_url 'env://' \
    --no-distributed
