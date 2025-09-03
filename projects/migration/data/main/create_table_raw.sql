--
-- PostgreSQL database dump
--

-- Dumped from database version 16.6
-- Dumped by pg_dump version 16.9 (Ubuntu 16.9-0ubuntu0.24.04.1)

SET statement_timeout = 0;
SET lock_timeout = 0;
SET idle_in_transaction_session_timeout = 0;
SET client_encoding = 'UTF8';
SET standard_conforming_strings = on;
SELECT pg_catalog.set_config('search_path', '', false);
SET check_function_bodies = false;
SET xmloption = content;
SET client_min_messages = warning;
SET row_security = off;

SET default_tablespace = '';

SET default_table_access_method = heap;

--
-- Name: main; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.main (
    "time" timestamp with time zone NOT NULL,
    host text,
    id text,
    name text,
    process text,
    fields jsonb
);


ALTER TABLE public.main OWNER TO postgres;

--
-- Name: COLUMN main.host; Type: COMMENT; Schema: public; Owner: postgres
--

