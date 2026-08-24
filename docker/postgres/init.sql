DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'perseo_app') THEN
        CREATE ROLE perseo_app
            LOGIN
            PASSWORD 'perseo_app'
            NOSUPERUSER
            NOCREATEDB
            NOCREATEROLE
            NOINHERIT
            NOREPLICATION
            NOBYPASSRLS;
    END IF;
END
$$;

GRANT CONNECT ON DATABASE perseo_rag TO perseo_app;
GRANT USAGE ON SCHEMA public TO perseo_app;

ALTER DEFAULT PRIVILEGES IN SCHEMA public
    GRANT SELECT, INSERT, UPDATE, DELETE ON TABLES TO perseo_app;
