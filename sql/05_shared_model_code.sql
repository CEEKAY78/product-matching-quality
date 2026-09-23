-- Share of gold pairs whose names share at least one model code (token with letters AND digits).
-- This is the single strongest signal, so it is worth knowing its coverage before modeling.
WITH codes AS (
    SELECT id, 'abt' AS side,
           list_filter(regexp_extract_all(lower(name), '[a-z0-9]+'),
                       t -> regexp_matches(t, '^[a-z0-9]{4,}$') AND regexp_matches(t, '[a-z]') AND regexp_matches(t, '[0-9]')) AS codes
    FROM abt
    UNION ALL
    SELECT id, 'buy',
           list_filter(regexp_extract_all(lower(name), '[a-z0-9]+'),
                       t -> regexp_matches(t, '^[a-z0-9]{4,}$') AND regexp_matches(t, '[a-z]') AND regexp_matches(t, '[0-9]'))
    FROM buy
)
SELECT count(*) AS gold_pairs,
       count_if(len(list_intersect(ca.codes, cb.codes)) > 0) AS pairs_sharing_code,
       round(100.0 * count_if(len(list_intersect(ca.codes, cb.codes)) > 0) / count(*), 1) AS pct,
       count_if(len(ca.codes) = 0 OR len(cb.codes) = 0) AS pairs_with_side_lacking_code
FROM mapping m
JOIN codes ca ON ca.id = m.idAbt AND ca.side = 'abt'
JOIN codes cb ON cb.id = m.idBuy AND cb.side = 'buy';
