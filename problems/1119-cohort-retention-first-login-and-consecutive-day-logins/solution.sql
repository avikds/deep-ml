SELECT l.user_id,
       MIN(l.login_date) AS first_login
FROM logins l
WHERE EXISTS (
    SELECT 1
    FROM logins a
    JOIN logins b
      ON b.user_id = a.user_id
     AND b.login_date = a.login_date + INTERVAL '1 day'
    WHERE a.user_id = l.user_id
)
GROUP BY l.user_id
ORDER BY l.user_id ASC;