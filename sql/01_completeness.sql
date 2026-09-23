-- Completeness per catalog: one aggregate pass per table, no row-level output.
SELECT 'abt' AS catalog,
       count(*)                                        AS rows,
       round(100.0 * count_if(price IS NULL) / count(*), 1)        AS missing_price_pct,
       round(100.0 * count_if(description = '') / count(*), 1)     AS missing_description_pct,
       round(100.0 * count_if(manufacturer = '') / count(*), 1)    AS missing_manufacturer_pct,
       round(median(price), 2)                          AS median_price
FROM abt
UNION ALL
SELECT 'buy',
       count(*),
       round(100.0 * count_if(price IS NULL) / count(*), 1),
       round(100.0 * count_if(description = '') / count(*), 1),
       round(100.0 * count_if(manufacturer = '') / count(*), 1),
       round(median(price), 2)
FROM buy;
