import polars as pl
from polars import col, when

def cut(col_name, edges, labels):
    """
    Assume edges are left-inclusive, right-exclusive, i.e. each bin = [x_min, x_max), except for first and last bin,
    which include everything to the left or everything to the right.

    Inputs:
      * col_name - name of column
      * edges    - list of edges. With N bins, we have N-1 edges.
      * labels   - name of each bin. With N bins, we have N labels.
    """
    c = col(col_name)
    labels = [pl.lit(x, pl.Categorical) for x in labels]
    expr = when(c < edges[0]).then(labels[0])
    for edge, label in zip(edges[1:], labels[1:-1]):
        expr = expr.when(c < edge).then(label)
    expr = expr.otherwise(labels[-1])

    return expr


df = pl.DataFrame({
    'a': range(10)
})

pl.Config.set_tbl_rows(99)
print(
    df.select(
        'a',
        cut(
            col_name='a',
            edges = [4, 6, 8],
            labels = ["< 4", "4-5.99", "6-7.99", ">= 8"]
        ).alias("binned")
    )
)
