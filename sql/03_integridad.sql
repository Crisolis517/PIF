-- =============================================================================
-- 03_integridad.sql — Verificaciones de integridad posteriores a la carga.
-- Cada bloque "-- name:" devuelve una fila (control, valor, esperado).
-- load.py las ejecuta, compara valor = esperado y guarda results/sql_integridad.csv
-- =============================================================================

-- name: I-01 Conteo de estudiantes
SELECT 'I-01 estudiantes' AS control, COUNT(*)::int AS valor, 674 AS esperado FROM academico.estudiante;

-- name: I-02 Conteo de matrículas (= filas UCI 395 + 649)
SELECT 'I-02 matriculas' AS control, COUNT(*)::int AS valor, 1044 AS esperado FROM academico.matricula;

-- name: I-03 Conteo de calificaciones (= matrículas x 3 períodos)
SELECT 'I-03 calificaciones' AS control, COUNT(*)::int AS valor, 3132 AS esperado FROM academico.calificacion;

-- name: I-04 Matrículas sin exactamente 3 calificaciones
SELECT 'I-04 matriculas_sin_3_notas' AS control, COUNT(*)::int AS valor, 0 AS esperado
FROM (SELECT m.estudiante_id, m.asignatura_id, COUNT(c.periodo_id) AS n
      FROM academico.matricula m
      LEFT JOIN academico.calificacion c USING (estudiante_id, asignatura_id)
      GROUP BY 1, 2 HAVING COUNT(c.periodo_id) <> 3) t;

-- name: I-05 Estudiantes huérfanos (sin matrícula)
SELECT 'I-05 estudiantes_sin_matricula' AS control, COUNT(*)::int AS valor, 0 AS esperado
FROM academico.estudiante s
WHERE NOT EXISTS (SELECT 1 FROM academico.matricula m WHERE m.estudiante_id = s.estudiante_id);

-- name: I-06 Matrículas con estudiante inexistente (FK anti-join)
SELECT 'I-06 matriculas_huerfanas' AS control, COUNT(*)::int AS valor, 0 AS esperado
FROM academico.matricula m LEFT JOIN academico.estudiante s USING (estudiante_id)
WHERE s.estudiante_id IS NULL;

-- name: I-07 Estudiantes con matrícula en ambas asignaturas (pares R-12)
SELECT 'I-07 estudiantes_en_ambas' AS control, COUNT(*)::int AS valor, 370 AS esperado
FROM (SELECT estudiante_id FROM academico.matricula GROUP BY 1 HAVING COUNT(*) = 2) t;

-- name: I-08 Notas fuera de dominio 0–20
SELECT 'I-08 notas_fuera_dominio' AS control, COUNT(*)::int AS valor, 0 AS esperado
FROM academico.calificacion WHERE nota NOT BETWEEN 0 AND 20;

-- name: I-09 Calificaciones marcadas no evaluadas (R-08)
SELECT 'I-09 calificaciones_no_evaluadas' AS control, COUNT(*)::int AS valor, 74 AS esperado
FROM academico.calificacion WHERE no_evaluado;

-- name: I-10 Trazabilidad: filas de origen únicas por archivo
SELECT 'I-10 filas_origen_duplicadas' AS control, COUNT(*)::int AS valor, 0 AS esperado
FROM (SELECT archivo_origen, fila_origen FROM academico.matricula GROUP BY 1, 2 HAVING COUNT(*) > 1) t;

-- name: I-11 Vista de integración: una fila por matrícula
SELECT 'I-11 filas_vista_integracion' AS control, COUNT(*)::int AS valor, 1044 AS esperado
FROM academico.v_matricula_ancha;
