# "Interactive multi-objective path planning algorithm for accessible sidewalks"
## Bachelor Thesis by Alisa Todorova

## Context
My Bachelor thesis and research has been done in the context of a research internship at the Urban Innovation and R&D department of the City of Amsterdam.
The motivation behind my research is to make Amsterdam more accessible. The end goal of the big project that my research is part of, is to create a highly user-dependent app for Amsterdam citizens with mobility issues and those who use assistive devices (i.e., wheelchairs, strollers, canes). 
This app should find the user the most accessible sidewalks, with respect to their specific needs. 

I propose a new method that leverages the power of Gaussian Processes and an interactive approach to quickly generate likely preferred solutions (i.e., paths) without computing the entire Pareto Front.

## Installation
1. Download and extract the zip file.
2. Install all dependencies:
```bash
 pip install -r requirements.txt
```
3. Run[`experiments`](./experiments.py) 

## How it works
1. [`outer-loop`](./outer_loop.py): Selects a target region, in which we search for new paths that have likely preferred value vectors
2. [`inner-loop`](./dfs_lower.py): Finds paths with value vectors in the target region. My approach is depth-first search (DFS) algorithm, guided by the lower bounds for each objective
for each node, obtained from the [`single-objective value iteration`](./single_vi_iter.py).
4. [`multi-objective value iteration`](./multi_vi_iter.py): Computes the full Pareto front


## Usage
1. To run [`multi-objective value iteration`](./multi_vi_iter.py):
```bash
pvi_result = pvi(G, T, ('length', 'crossing'))
```
2. [`experiments`](./experiments.py): Running different experiments of my proposed algorithm.
3. [`full map`](./Sidewalk_width_crossings.geojson): Full map with radius of 800m, centered around the Rijksmuseum (11401 nodes)
4. [`small map`](./Sidewalk_width_crossings_small.geojson): Small map with radius of 250m, centered around the Rijksmuseum (1006 nodes)


## Acknowledgements
1. In [`outer-loop`](./outer_loop.py), we use the Gaussian process and Acquisition function as blackboxes from
"Ordered Preference Elicitation Strategies for Supporting Multi-Objective Decision Making" by Luisa M. Zintgraf, Diederik M. Roijers, Sjoerd Linders, Catholijn M. Jonker, and Ann Nowé, which was published at AAMAS (Autonomous Agents and Multi-Agent Systems), Stockholm 2018.
[GitHub link](https://github.com/lmzintgraf/gp_pref_elicit).
2. In [`multi-objective value iteration`](./multi_vi_iter.py), for Prune we follow the methods pareto_dominates, p_prune and pvi from
Roijers, D. M., Röpke, W., Nowe, A., & Radulescu, R. (2021).
On Following Pareto-Optimal Policies in Multi-Objective Planning and Reinforcement Learning.
Paper presented at Multi-Objective Decision Making Workshop 2021.
[Link to paper](http://modem2021.cs.nuigalway.ie/papers/MODeM_2021_paper_3.pdf) and
[GitHub link](https://github.com/rradules/POP-following/tree/main).
