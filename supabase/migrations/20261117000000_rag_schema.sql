-- Applied to the Supabase project "enterprise-rag-assistant" on 2026-11-17.
create extension if not exists vector with schema extensions;

create table if not exists public.documents (
    doc_id       text primary key,
    title        text not null,
    publisher    text not null default '',
    published    text not null default '',
    url          text not null default '',
    retrieved    text not null default '',
    file_name    text not null default '',
    access_level integer not null default 0,
    created_at   timestamptz not null default now()
);

create table if not exists public.chunks (
    chunk_id     text primary key,
    doc_id       text not null references public.documents(doc_id) on delete cascade,
    page         integer not null,
    idx          integer not null,
    text         text not null,
    text_hash    text not null,
    access_level integer not null default 0,
    embedding    extensions.vector(768) not null,
    tsv          tsvector generated always as (to_tsvector('english', text)) stored
);

create index if not exists chunks_tsv_idx  on public.chunks using gin (tsv);
create index if not exists chunks_vec_idx  on public.chunks using hnsw (embedding extensions.vector_cosine_ops);
create index if not exists chunks_hash_idx on public.chunks (text_hash);
create index if not exists chunks_doc_idx  on public.chunks (doc_id);

-- The app connects directly with the database role. Row level security with no policies means the
-- public REST API (anon and authenticated keys) can read and write nothing in these tables.
alter table public.documents enable row level security;
alter table public.chunks    enable row level security;
