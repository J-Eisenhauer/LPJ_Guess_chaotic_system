# Twin runs of a deterministic LPJ-GUESS: how long does a small perturbation stay predictable?

LPJ-GUESS 4.1.1 (cohort mode, one patch, constant climate, no random numbers) is a deterministic dynamical system.
At one tropical-African grid cell (P1400, about 1400 mm rain per year, forest-savanna band) we perturb the model slightly
and follow the difference to an unperturbed control run, with and without fire.

| notebook | content | methods |
|---|---|---|
| `00_overview_data_FAIR.ipynb` | problem, workflow, data description, FAIR reflection, licence | - |
| `01_short_kick_experiment.ipynb` | 100 runs per mode, ten simultaneous +-1 % kicks, 512 years, 8 state variables | multi-task LASSO, cross-validation, train/test split, bootstrap |
| `02_long_rain_experiment.ipynb` | 30 runs per mode, one year of rain x (1+eps), 4580 years, NEE | delay embedding, hierarchical clustering, running correlation |

The project has two halves. The model half (LPJ-GUESS patches and run scripts) is not published here; LPJ-GUESS itself is available from
its developers (https://web.nateko.lu.se/lpj-guess/) and with frozen stochastics it can generate the dataset. The model half ends in
one file, `data/p1400_twin_runs.csv`. This folder is the analysis half: everything from that csv to the figures.


## Files

```
00_*.ipynb 01_*.ipynb 02_*.ipynb   the analysis, with outputs; settings at the top of each notebook
plotlib.py                         the plot functions used by the notebooks
data/p1400_twin_runs.csv           the data (one row = one model year of one run)
data/columns.csv                   every column with unit and meaning
CITATION.cff  LICENSE  LICENSE-DATA.md  requirements.txt
```

`figures/` is created when the notebooks run.

Provenance of the data, the FAIR reflection and the known limitations are in `00_overview_data_FAIR.ipynb` and in
the caveat sections of notebooks 01 and 02.

## Licence and citation

Code: MIT (`LICENSE`). Data: CC BY 4.0 (`LICENSE-DATA.md`). Please cite as in `CITATION.cff`.
LPJ-GUESS itself (Smith et al. 2014, Biogeosciences 11, 2027-2054) has its own licence and is not redistributed here.
