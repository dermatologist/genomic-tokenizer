
## :battery: Potential uses (Not tested)
* Try this instead of the default BPE tokenizer for pre-training [DNABERT_2](https://github.com/MAGICS-LAB/DNABERT_2) (*see section 5*)
* Pre-train [DNAGPT](https://github.com/maris205/dnagpt) replacing BPE with this. (*No need to train_bpe.ipynb. Replace the `tokenizer` with this.*)
* Replace the [default tokenizer](https://github.com/songlab-cal/gpn/blob/05b23c54c572813810c094b31031901f7109575b/gpn/data.py#L511) in [GPN (Genomic Pre-trained Network)](https://github.com/songlab-cal/gpn) with this.