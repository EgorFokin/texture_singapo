## Setup

Make sure git-lsf and libgraphviz-dev is installed

```shell
cd texture_singapo
conda create -n texture_singapo python=3.10
conda activate texture_singapo
./install.sh
```

## Evaluation

For evaluation place desired dataset into /eval_data. Right now it contains a small subset(10 objects) from ACD dataset. You can use the current structure as a reference.

Then run:

```shell
python evaluate.py --use_cached --add_no_easitex
```

- _use_cached_ makes the script use previously generated results stored in the output folder.
- _add_no_easitex_ adds results without applying easi-tex texture to the evaluation results
