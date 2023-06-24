export PYTHONPATH=/home/penguin/Documents/fairseq/
python fairseq_cli/hydra_train.py --config-dir examples/data2vec/config/v2/ --config-name base_audio_only_task task.data=/home/penguin/Documents/fairseq/examples/wav2vec/
# python wav2vec_manifest.py ~/Documents/data/cv-corpus-10.0-delta-2022-07-04-en/cv-corpus-10.0-delta-2022-07-04/en/clips/ --dest ./ --ext mp3 --valid-percent 0.1