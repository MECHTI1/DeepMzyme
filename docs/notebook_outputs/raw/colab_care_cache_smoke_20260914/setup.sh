set -eu
cd /content/DeepMzyme
python -m pip install uv
uv python install 3.12
uv venv --python 3.12 /content/deepmzyme-env
uv pip install --python /content/deepmzyme-env/bin/python torch==2.11.0 --index-url https://download.pytorch.org/whl/cu128
uv pip install --python /content/deepmzyme-env/bin/python -c /content/torch-constraint.txt -r requirements/colab-overlay.txt esm==3.2.3
curl -L --fail --retry 3 'https://huggingface.co/datasets/GMBioinformatics/DeepMzyme/resolve/4c36d3bfbf5e0c892fb165cbf101184e4151dda2/DeepMzyme_Data_v11_manifest_exact_common70_nonoverlap_clean30_care30_esm_ring_external.tar.gz' -o /content/data.tar.gz
echo '8f869b8aa78dd2dc2af5efb856d137c01ba289b8fe6327b66a8139b0166975c7  /content/data.tar.gz' | sha256sum -c -
tar -xzf /content/data.tar.gz -C /content/DeepMzyme
touch /content/setup_complete
