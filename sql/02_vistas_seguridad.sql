-- =============================================================================
-- 02_vistas_seguridad.sql — Vista de integración y rol de solo lectura
-- =============================================================================
SET search_path TO academico;

-- Vista ancha: una fila por matrícula (unidad de análisis del dataset final).
-- Pivota las tres calificaciones y une atributos de estudiante y escuela.
CREATE OR REPLACE VIEW v_matricula_ancha AS
SELECT
    m.estudiante_id,
    a.codigo                                   AS asignatura,
    e.codigo                                   AS escuela,
    s.sexo, s.edad, s.zona, s.tamano_familia, s.estado_padres,
    s.educ_madre, s.educ_padre, s.trabajo_madre, s.trabajo_padre, s.motivo_eleccion,
    s.tutor_legal, s.tiempo_viaje, s.tiempo_estudio, s.apoyo_escolar, s.apoyo_familiar,
    s.actividades, s.guarderia, s.desea_superior, s.internet, s.relacion_romantica,
    s.relacion_familiar, s.tiempo_libre, s.salidas, s.alcohol_semana, s.alcohol_finde, s.salud,
    m.reprobaciones_previas, m.clases_pagadas, m.faltas, m.faltas_confiable,
    m.archivo_origen, m.fila_origen,
    MAX(c.nota)        FILTER (WHERE p.codigo = 'P1') AS nota_p1,
    MAX(c.nota)        FILTER (WHERE p.codigo = 'P2') AS nota_p2,
    MAX(c.nota)        FILTER (WHERE p.codigo = 'P3') AS nota_p3,
    BOOL_OR(c.no_evaluado) FILTER (WHERE p.codigo = 'P1') AS no_evaluado_p1,
    BOOL_OR(c.no_evaluado) FILTER (WHERE p.codigo = 'P2') AS no_evaluado_p2,
    BOOL_OR(c.no_evaluado) FILTER (WHERE p.codigo = 'P3') AS no_evaluado_p3,
    (SELECT COUNT(*) FROM matricula m2 WHERE m2.estudiante_id = m.estudiante_id) = 2
                                               AS en_ambas_asignaturas
FROM matricula m
JOIN asignatura   a ON a.asignatura_id = m.asignatura_id
JOIN estudiante   s ON s.estudiante_id = m.estudiante_id
JOIN escuela      e ON e.escuela_id    = s.escuela_id
JOIN calificacion c ON c.estudiante_id = m.estudiante_id AND c.asignatura_id = m.asignatura_id
JOIN periodo      p ON p.periodo_id    = c.periodo_id
GROUP BY m.estudiante_id, a.codigo, e.codigo, s.estudiante_id, m.asignatura_id;

-- Rol de solo lectura para analistas (principio de mínimo privilegio).
DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'analista_lectura') THEN
        CREATE ROLE analista_lectura NOLOGIN;
    END IF;
END $$;
GRANT USAGE ON SCHEMA academico TO analista_lectura;
GRANT SELECT ON ALL TABLES IN SCHEMA academico TO analista_lectura;
