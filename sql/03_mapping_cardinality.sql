-- Cardinality of the gold mapping: how many Abt products link to more than one Buy product
-- and vice versa. A perfect 1:1 mapping would give zero in both rows.
SELECT 'abt -> many buy' AS direction, count(*) AS ids
FROM (SELECT idAbt FROM mapping GROUP BY 1 HAVING count(DISTINCT idBuy) > 1)
UNION ALL
SELECT 'buy -> many abt', count(*)
FROM (SELECT idBuy FROM mapping GROUP BY 1 HAVING count(DISTINCT idAbt) > 1)
UNION ALL
SELECT 'abt products without any match', count(*)
FROM abt WHERE id NOT IN (SELECT idAbt FROM mapping)
UNION ALL
SELECT 'buy products without any match', count(*)
FROM buy WHERE id NOT IN (SELECT idBuy FROM mapping);
