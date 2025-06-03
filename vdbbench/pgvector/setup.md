### Install postgresql-17.0 from source code

#### Get from source
The PostgreSQL source code for released versions can be obtained from website: https://www.postgresql.org/ftp/source/. Download the postgresql-17.0.tar.gz , then unpack it:

```
tar xf postgresql-17.0.tar.gz
```

This will create a directory postgresql-17.0 under the current directory with the PostgreSQL sources. Change into that directory for the rest of the installation procedure.

#### Building and Installation with Autoconf and Make

```
cd postgresql-17.0

./configure

make

su

make install
```

#### Install pgvector from source code

```
git clone https://github.com/pgvector/pgvector.git -b v0.8.0

cd pgvector

make PG_CONFIG=/usr/local/pgsql/bin/pg_config

make PG_CONFIG=/usr/local/pgsql/bin/pg_config install

```
