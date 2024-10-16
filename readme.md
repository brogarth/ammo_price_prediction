# Overview

This is code I used to explore an ammunition price dataset as well as build a slew of supervised models to predict said prices using Facebook Prophet. A writeup describing the findings can be found [here](klott.xyz/ammo_model/article.html).

Usage is as follows:

- exploration.py: gathers basic statistics and plots of the data.
- main.py: builds all the models used in the study.
- utils.py: a library that helps automate some of the functionality in the above scripts. Not meant to be used directly.
- environment.yml: requirements to replicate the libraries used in the conda environment.

Both exploration.py and main.py will place statistical summaries, plots, and trained models into a local folder called "output." All scripts assumed a local folder "data" is present. The data itself is not hosted on Github - you will have to reach out to me directly if you would like to replicate the study (it should all be publicly available in the sources described in the article regardless).
