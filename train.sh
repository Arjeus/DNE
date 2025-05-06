set -e

echo "Launching training on GPU..."
# Run the training with explicit device assignment and 
# Note: using underscore instead of hyphen for dist_url
python main.py \
    --options options/data/iris_3.yaml options/data/iris_order1.yaml options/model/cifar_dne.yaml \
    --name dne_64dim_digits \
    --data-path ./data/DIGITS/ \
    --log-path ./logs \
    --device cuda:0 \
    --output-basedir ./checkpoints \
    --extra-dim 32 \
    --extra-heads 1 \
    --dist_url 'env://' \
    --no-distributed \
    --base-epochs 100 \
    --epochs 100 \
    --memory-size 0 \
    --seed 0
