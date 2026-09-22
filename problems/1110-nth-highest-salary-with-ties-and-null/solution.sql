SELECT MAX(salary) AS nth_salary
FROM (
    SELECT DISTINCT salary,
           DENSE_RANK() OVER (ORDER BY salary DESC) AS rnk
    FROM employee
    WHERE salary IS NOT NULL
) t
WHERE rnk = 3;