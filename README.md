# Zephyr-Embedding

Reference implementation of **Zephyr-Embedding**, a position-aware index
structure for approximate nearest-neighbour search on drifting embeddings.

## Layout

- `src/zephyr/index.py` - segment layout and search.
- `src/zephyr/maintenance.py` - compaction and maintenance policy.
- `data/README.md` - notes on the public embedding streams.
- `experiments/drift_sweep.yaml` - sweep configuration used in the paper.

## Quick start

```bash
pip install -r requirements.txt
python -m zephyr.index --config experiments/drift_sweep.yaml
```

## License

MIT.
