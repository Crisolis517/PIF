-- =============================================================================
-- 04_analiticas.sql — Consultas analíticas (JOIN + agregación) que responden P1.
-- load.py las ejecuta y guarda la salida en results/sql_<id>.csv
-- =============================================================================

-- name: A-01 Nota media por asignatura y período (excluye no evaluados, R-08)
SELECT a.codigo AS asignatura, p.codigo AS periodo,
       COUNT(*) FILTER (WHERE NOT c.no_evaluado)                        AS n_evaluados,
       ROUND(AVG(c.nota) FILTER (WHERE NOT c.no_evaluado), 2)           AS media,
       ROUND(STDDEV_SAMP(c.nota) FILTER (WHERE NOT c.no_evaluado), 2)   AS desv_est,
       PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY c.nota)
           FILTER (WHERE NOT c.no_evaluado)                             AS mediana,
       COUNT(*) FILTER (WHERE c.no_evaluado)                            AS n_no_evaluados
FROM academico.calificacion c
JOIN academico.asignatura a ON a.asignatura_id = c.asignatura_id
JOIN academico.periodo    p ON p.periodo_id    = c.periodo_id
GROUP BY a.codigo, p.codigo, p.orden
ORDER BY a.codigo, p.orden;

-- name: A-02 Tasa de reprobación final (G3 < 10, incluye no evaluados) por escuela y asignatura
SELECT e.codigo AS escuela, a.codigo AS asignatura,
       COUNT(*)                                     AS n_matriculas,
       SUM(CASE WHEN c.nota < 10 THEN 1 ELSE 0 END) AS n_reprobados,
       ROUND(AVG(CASE WHEN c.nota < 10 THEN 1.0 ELSE 0 END), 4) AS tasa_reprobacion,
       SUM(CASE WHEN c.no_evaluado THEN 1 ELSE 0 END) AS n_no_evaluados
FROM academico.calificacion c
JOIN academico.periodo    p ON p.periodo_id    = c.periodo_id AND p.codigo = 'P3'
JOIN academico.matricula  m ON m.estudiante_id = c.estudiante_id AND m.asignatura_id = c.asignatura_id
JOIN academico.estudiante s ON s.estudiante_id = m.estudiante_id
JOIN academico.escuela    e ON e.escuela_id    = s.escuela_id
JOIN academico.asignatura a ON a.asignatura_id = m.asignatura_id
GROUP BY e.codigo, a.codigo
ORDER BY tasa_reprobacion DESC;

-- name: A-03 Transición de estado aprobado/reprobado entre P1 y P3 por asignatura
SELECT a.codigo AS asignatura,
       CASE WHEN c1.nota >= 10 THEN 'aprueba_P1' ELSE 'reprueba_P1' END AS estado_p1,
       CASE WHEN c3.nota >= 10 THEN 'aprueba_P3' ELSE 'reprueba_P3' END AS estado_p3,
       COUNT(*) AS n
FROM academico.calificacion c1
JOIN academico.calificacion c3
  ON c3.estudiante_id = c1.estudiante_id AND c3.asignatura_id = c1.asignatura_id AND c3.periodo_id = 3
JOIN academico.asignatura a ON a.asignatura_id = c1.asignatura_id
WHERE c1.periodo_id = 1
GROUP BY 1, 2, 3
ORDER BY 1, 2, 3;

-- name: A-04 Reprobación final según reprobaciones previas (todas las matrículas)
SELECT m.reprobaciones_previas,
       COUNT(*) AS n_matriculas,
       ROUND(AVG(CASE WHEN c.nota < 10 THEN 1.0 ELSE 0 END), 4) AS tasa_reprobacion
FROM academico.matricula m
JOIN academico.calificacion c
  ON c.estudiante_id = m.estudiante_id AND c.asignatura_id = m.asignatura_id AND c.periodo_id = 3
GROUP BY 1 ORDER BY 1;
