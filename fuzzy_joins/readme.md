# Fast Fuzzy Joins in Python

This is a culmination of finding out an efficient way to perform fuzzy matching in Python at scale (on a single machine at least). Fuzzy matching in Python does not seem to be anywhere close to the level in R that packages like [`stringdist`](https://cran.r-project.org/web/packages/stringdist/) provide - multi-threaded string distance matching across a variety of distance metrics.

The primary source that this file refers for its queries is [this Medium blog](https://towardsdatascience.com/fuzzy-matching-at-scale-84f2bfd0c536?gi=9bd6bb8ccfd5), which, like most Medium blogs that claim to have focused data science content, is just a hodgepodge of pathetic code borrowed from multiple sources. The original source for performing TF-IDF driven fuzzy join at scale is likely by [https://github.com/bergvca](https://bergvca.github.io/2017/10/14/super-fast-string-matching.html) who has their own package, [StringGrouper](https://github.com/Bergvca/string_grouper), although the optimizations that leverage the NMSLIB library is likely the Medium blog author's.

The Medium blog author's code is particular poor in documentation, and here is slightly cleaned up version that I may revisit to add more information to. The approach recommended for using this class is:

1. Prepare the list of strings that you would like compared, along with the source strings that you want compared to. 
2. Send these two to the class by calling `fastfuzzy()`, with relevant arguments. Both `source_strings` and `compare_strings` must be lists of strings, and can be identical.
3. Call the class variable with the `query` method, and specify the value of `k` that interests you.
