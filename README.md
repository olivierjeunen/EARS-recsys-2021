# Top-K Contextual Bandits with Equity of Exposure
Source code for our paper "Top-K Contextual Bandits with Equity of Exposure" published at RecSys 2021.


## Reproducibility
- Download the data under `src/data/`, following instructions in the [original repository](https://github.com/deezer/carousel_bandits).
- Run the following commands to reproduce the experimental results reported in the manuscript: `cd src/; python3 main.py --policies ts-lin-pessimistic-reg-2000,ts-lin-pessimistic-reg-2000-personalised-shuffle-99,ts-lin-pessimistic-reg-2000-personalised-shuffle-975,ts-lin-pessimistic-reg-2000-personalised-shuffle-95,ts-lin-pessimistic-reg-2000-personalised-shuffle-90,ts-lin-pessimistic-reg-2000-shuffle-6 --n_rounds 50 --print_every 1 --gamma 0.9 --output_path results.json`.

## Paper
If you use our code in your research, please remember to cite our paper:

```BibTeX
    @inproceedings{JeunenRecSys2021_B,
      author = {Jeunen, Olivier and Goethals, Bart},
      title = {Top-K Contextual Bandits with Equity of Exposure},
      booktitle = {Proceedings of the 15th ACM Conference on Recommender Systems},
      series = {RecSys '21},
      year = {2021},
      publisher = {ACM},
    }
