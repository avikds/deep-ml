SELECT user_id
FROM purchases
WHERE purchase_date >= DATE '2024-01-01'
  AND purchase_date < DATE '2025-01-01'
GROUP BY user_id
HAVING COUNT(DISTINCT EXTRACT(MONTH FROM purchase_date)) = 12
   AND COUNT(*) FILTER (
       WHERE EXTRACT(MONTH FROM purchase_date) = 1
   ) >= 2
   AND COUNT(*) FILTER (
       WHERE EXTRACT(MONTH FROM purchase_date) = 2
   ) >= 2
   AND COUNT(*) FILTER (
       WHERE EXTRACT(MONTH FROM purchase_date) = 3
   ) >= 2
   AND COUNT(*) FILTER (
       WHERE EXTRACT(MONTH FROM purchase_date) = 4
   ) >= 2
   AND COUNT(*) FILTER (
       WHERE EXTRACT(MONTH FROM purchase_date) = 5
   ) >= 2
   AND COUNT(*) FILTER (
       WHERE EXTRACT(MONTH FROM purchase_date) = 6
   ) >= 2
   AND COUNT(*) FILTER (
       WHERE EXTRACT(MONTH FROM purchase_date) = 7
   ) >= 2
   AND COUNT(*) FILTER (
       WHERE EXTRACT(MONTH FROM purchase_date) = 8
   ) >= 2
   AND COUNT(*) FILTER (
       WHERE EXTRACT(MONTH FROM purchase_date) = 9
   ) >= 2
   AND COUNT(*) FILTER (
       WHERE EXTRACT(MONTH FROM purchase_date) = 10
   ) >= 2
   AND COUNT(*) FILTER (
       WHERE EXTRACT(MONTH FROM purchase_date) = 11
   ) >= 2
   AND COUNT(*) FILTER (
       WHERE EXTRACT(MONTH FROM purchase_date) = 12
   ) >= 2
ORDER BY user_id ASC;