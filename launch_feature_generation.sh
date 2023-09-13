echo $CUDA_VISIBLE_DEVICES
echo $CUDA_DEVICE_ORDER

ROOT_DATASET="data/CLEVR_CoGenT_v1.0/"

extract_features()
{
  name=$1
  python3 main.py \
  --run gen_features \
  --batch_size 64 \
  --path_dataset "${ROOT_DATASET}/${name}"
}

for name in {"images",}
do
  echo "Extracting features for ${name}"
  extract_features $name
done

