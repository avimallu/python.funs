# python.funs
A handy list of functions that I've developed over time to make my work easier. Details on what they contain below.

## Clustering
This is a folder for any more clustering functions or solutions I come up with, current only has what's mentioned below.

### KPOD clustering
KPOD is a method to use K-Means on data that has largely missing values. The official Python package is [here](https://pypi.org/project/kPOD/), with a CRAN package by a different author [here](https://cran.r-project.org/web/packages/kpodclustr/). Both of these are based on the original paper [here](https://arxiv.org/abs/1411.7013).

What have I changed from the original implementation?
1. Created a single class instead of multiple functions, since most of the code relies on `scikit-learn`.
2. Added a method to obtain the "best" _k_ value based on the silouette score.

## Fuzzy joins in Python
Python lacks broad string based fuzzy matching support, unlike R with its [`stringdist`](https://cran.r-project.org/web/packages/stringdist/index.html) package. If you have flexibility in the tool you want to use, please, for the love of God, use R in this instance. If you absolutely have to use Python, please head over to [my repository](https://github.com/avimallu/python.funs/tree/main/fuzzy_joins) for a fast and efficient solution that uses tf-idf values, with documentation from where it was created.
