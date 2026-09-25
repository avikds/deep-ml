WITH numbered AS (
    SELECT
        user_id,
        visit_date,
        visit_date - (
            ROW_NUMBER() OVER (
                PARTITION BY user_id
                ORDER BY visit_date
            )::int
        ) AS streak_group
    FROM visits
    WHERE visit_date <= CURRENT_DATE - 1
),
streaks AS (
    SELECT
        user_id,
        streak_group,
        COUNT(*) AS streak_length
    FROM numbered
    GROUP BY user_id, streak_group
)
SELECT
    user_id,
    MAX(streak_length) AS longest_streak
FROM streaks
GROUP BY user_id
ORDER BY longest_streak DESC, user_id ASC
LIMIT 100;