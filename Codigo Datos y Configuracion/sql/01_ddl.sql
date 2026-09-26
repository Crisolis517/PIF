-- =============================================================================
-- 01_ddl.sql — Modelo relacional 3FN del registro académico (fuente F1, UCI)
-- Compatible con PostgreSQL 16 (probado también en PostgreSQL 18).
-- Esquema: academico. Seis tablas: escuela, asignatura, periodo, estudiante,
-- matricula (resuelve N:M estudiante–asignatura) y calificacion (1FN de G1–G3).
-- =============================================================================
DROP SCHEMA IF EXISTS academico CASCADE;
CREATE SCHEMA academico;
SET search_path TO academico;

-- Catálogo de centros --------------------------------------------------------
CREATE TABLE escuela (
    escuela_id   SMALLINT     PRIMARY KEY,
    codigo       CHAR(2)      NOT NULL UNIQUE CHECK (codigo IN ('GP', 'MS')),
    nombre       VARCHAR(40)  NOT NULL
);

-- Catálogo de asignaturas ----------------------------------------------------
CREATE TABLE asignatura (
    asignatura_id SMALLINT    PRIMARY KEY,
    codigo        CHAR(3)     NOT NULL UNIQUE CHECK (codigo IN ('MAT', 'POR')),
    nombre        VARCHAR(40) NOT NULL
);

-- Períodos lectivos 2005–2006: P1<->G1, P2<->G2, P3<->G3 (nota final) --------
CREATE TABLE periodo (
    periodo_id   SMALLINT    PRIMARY KEY,
    codigo       CHAR(2)     NOT NULL UNIQUE CHECK (codigo IN ('P1', 'P2', 'P3')),
    nombre       VARCHAR(60) NOT NULL,
    orden        SMALLINT    NOT NULL UNIQUE CHECK (orden BETWEEN 1 AND 3),
    nota_uci     CHAR(2)     NOT NULL UNIQUE CHECK (nota_uci IN ('G1', 'G2', 'G3')),
    fecha_inicio DATE        NOT NULL,
    fecha_fin    DATE        NOT NULL,
    CHECK (fecha_fin > fecha_inicio)
);

-- Estudiante: atributos personales (no dependen de la asignatura) -------------
CREATE TABLE estudiante (
    estudiante_id      INTEGER     PRIMARY KEY,
    escuela_id         SMALLINT    NOT NULL REFERENCES escuela (escuela_id),
    sexo               CHAR(1)     NOT NULL CHECK (sexo IN ('F', 'M')),
    edad               SMALLINT    NOT NULL CHECK (edad BETWEEN 15 AND 22),
    zona               VARCHAR(6)  NOT NULL CHECK (zona IN ('Urbana', 'Rural')),
    tamano_familia     VARCHAR(3)  NOT NULL CHECK (tamano_familia IN ('<=3', '>3')),
    estado_padres      VARCHAR(9)  NOT NULL CHECK (estado_padres IN ('Juntos', 'Separados')),
    educ_madre         SMALLINT    NOT NULL CHECK (educ_madre BETWEEN 0 AND 4),
    educ_padre         SMALLINT    NOT NULL CHECK (educ_padre BETWEEN 0 AND 4),
    trabajo_madre      VARCHAR(10) NOT NULL CHECK (trabajo_madre IN ('docente','salud','servicios','hogar','otro')),
    trabajo_padre      VARCHAR(10) NOT NULL CHECK (trabajo_padre IN ('docente','salud','servicios','hogar','otro')),
    motivo_eleccion    VARCHAR(10) NOT NULL CHECK (motivo_eleccion IN ('cercania','reputacion','oferta','otro')),
    tutor_legal        VARCHAR(5)  NOT NULL CHECK (tutor_legal IN ('madre','padre','otro')),
    tiempo_viaje       SMALLINT    NOT NULL CHECK (tiempo_viaje BETWEEN 1 AND 4),
    tiempo_estudio     SMALLINT    NOT NULL CHECK (tiempo_estudio BETWEEN 1 AND 4),
    apoyo_escolar      BOOLEAN     NOT NULL,
    apoyo_familiar     BOOLEAN     NOT NULL,
    actividades        BOOLEAN     NOT NULL,
    guarderia          BOOLEAN     NOT NULL,
    desea_superior     BOOLEAN     NOT NULL,
    internet           BOOLEAN     NOT NULL,
    relacion_romantica BOOLEAN     NOT NULL,
    relacion_familiar  SMALLINT    NOT NULL CHECK (relacion_familiar BETWEEN 1 AND 5),
    tiempo_libre       SMALLINT    NOT NULL CHECK (tiempo_libre BETWEEN 1 AND 5),
    salidas            SMALLINT    NOT NULL CHECK (salidas BETWEEN 1 AND 5),
    alcohol_semana     SMALLINT    NOT NULL CHECK (alcohol_semana BETWEEN 1 AND 5),
    alcohol_finde      SMALLINT    NOT NULL CHECK (alcohol_finde BETWEEN 1 AND 5),
    salud              SMALLINT    NOT NULL CHECK (salud BETWEEN 1 AND 5)
);

-- Matrícula: resuelve N:M estudiante–asignatura; atributos propios de la asignatura
CREATE TABLE matricula (
    estudiante_id          INTEGER     NOT NULL REFERENCES estudiante (estudiante_id) ON DELETE CASCADE,
    asignatura_id          SMALLINT    NOT NULL REFERENCES asignatura (asignatura_id),
    reprobaciones_previas  SMALLINT    NOT NULL CHECK (reprobaciones_previas BETWEEN 0 AND 3), -- 3 = "3 o más" (R-07)
    clases_pagadas         BOOLEAN     NOT NULL,
    faltas                 SMALLINT    NOT NULL CHECK (faltas BETWEEN 0 AND 93),
    faltas_confiable       BOOLEAN     NOT NULL,          -- anotación de la regla R-09
    archivo_origen         VARCHAR(20) NOT NULL,          -- linaje: archivo UCI
    fila_origen            INTEGER     NOT NULL CHECK (fila_origen >= 2), -- linaje: línea del CSV
    PRIMARY KEY (estudiante_id, asignatura_id),
    UNIQUE (archivo_origen, fila_origen)
);

-- Calificación: una fila por matrícula y período (1FN de G1, G2, G3) ---------
CREATE TABLE calificacion (
    estudiante_id  INTEGER  NOT NULL,
    asignatura_id  SMALLINT NOT NULL,
    periodo_id     SMALLINT NOT NULL REFERENCES periodo (periodo_id),
    nota           SMALLINT NOT NULL CHECK (nota BETWEEN 0 AND 20), -- valor original UCI
    no_evaluado    BOOLEAN  NOT NULL,                              -- anotación de la regla R-08
    PRIMARY KEY (estudiante_id, asignatura_id, periodo_id),
    FOREIGN KEY (estudiante_id, asignatura_id)
        REFERENCES matricula (estudiante_id, asignatura_id) ON DELETE CASCADE,
    CHECK (NOT no_evaluado OR nota = 0)
);

-- Índices justificados --------------------------------------------------------
-- Las PK ya indexan (estudiante_id, ...). Se añaden índices para los JOIN y
-- filtros de las consultas analíticas (P1: agregación por asignatura y período;
-- tasas por escuela).
CREATE INDEX ix_estudiante_escuela      ON estudiante (escuela_id);
CREATE INDEX ix_matricula_asignatura    ON matricula (asignatura_id);
CREATE INDEX ix_calificacion_asig_per   ON calificacion (asignatura_id, periodo_id);
CREATE INDEX ix_calificacion_periodo    ON calificacion (periodo_id) INCLUDE (nota, no_evaluado);

COMMENT ON TABLE calificacion IS 'Notas G1-G3 despivotadas (R-14). nota=0 con no_evaluado=true se excluye de medias (R-08).';
COMMENT ON COLUMN matricula.reprobaciones_previas IS 'failures UCI; el valor 3 representa 3 o más (R-07).';
