-- Relative price gap on gold pairs where both sides have a price.
-- Filter first (both prices present), then aggregate: the join touches ~1k rows, not 1M.
WITH priced AS (
    SELECT m.idAbt, m.idBuy,
           abs(a.price - b.price) / greatest(a.price, b.price) AS rel_gap
    FROM mapping m
    JOIN abt a ON a.id = m.idAbt
    JOIN buy b ON b.id = m.idBuy
    WHERE a.price IS NOT NULL AND b.price IS NOT NULL
)
SELECT count(*)                          AS priced_pairs,
       round(quantile_cont(rel_gap, 0.5), 3)  AS p50_gap,
       round(quantile_cont(rel_gap, 0.9), 3)  AS p90_gap,
       round(quantile_cont(rel_gap, 0.99), 3) AS p99_gap,
       count_if(rel_gap > 0.5)           AS pairs_gap_over_50pct
FROM priced;
