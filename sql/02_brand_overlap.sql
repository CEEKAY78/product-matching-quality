-- Top brands on each side and how many products of that brand the other catalog has.
-- Brand vocabularies differ ("sony" vs "sony corp"), which is why brand match is a soft feature.
WITH a AS (SELECT manufacturer, count(*) AS n_abt FROM abt GROUP BY 1),
     b AS (SELECT manufacturer, count(*) AS n_buy FROM buy GROUP BY 1)
SELECT coalesce(a.manufacturer, b.manufacturer) AS brand,
       coalesce(n_abt, 0) AS n_abt,
       coalesce(n_buy, 0) AS n_buy
FROM a FULL OUTER JOIN b USING (manufacturer)
ORDER BY n_abt + n_buy DESC, brand
LIMIT 15;
