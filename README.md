# boti-data

`boti-data` is the **data access and data transformation layer** of the Boti ecosystem.

It builds on top of `boti` and gives teams a reusable interface for working with structured data across databases, parquet datasets, schema-controlled transformations, and distributed or partitioned loading workflows.

## What `boti-data` is for

Many teams have the same recurring problem: business logic depends on data that lives in multiple places, arrives in slightly different shapes, and is loaded through a mix of notebooks, scripts, ad hoc SQL, and one-off helpers.

`boti-data` helps turn that into a more coherent data access layer.

It is designed for codebases that need to:

- connect to named data sources consistently
- reflect or model database tables without hand-writing everything up front
- load data through a gateway instead of bespoke query snippets everywhere
- normalise and validate schemas before downstream use
- combine parquet and database workflows in one library
- scale from simple local reads to partitioned or distributed loading

## Problems `boti-data` solves

`boti-data` is useful when data code is suffering from issues like:

- repeated connection boilerplate across notebooks and services
- slow, fragile query code copied from place to place
- inconsistent schema assumptions between producers and consumers
- difficult transitions from exploratory analysis to reusable pipelines
- manual join and field-mapping logic repeated in many modules
- no common abstraction for loading data from SQL and parquet sources

By centralising those patterns, `boti-data` reduces duplicated plumbing and makes transformations easier to reason about.

## Why `boti-data` can make a huge difference

The biggest benefit of `boti-data` is that it creates a **shared data interface** between infrastructure and business logic.

That means teams can spend less time rewriting access code and more time working on actual transformations, validation rules, and downstream decisions.

It can make a major difference when:

- analysts and engineers share the same source systems
- a notebook prototype needs to become production code
- multiple data products depend on the same tables or parquet layouts
- schema drift is a recurring source of errors
- large extracts need partitioning or distributed execution
- teams want a clean boundary between connection details and transformation logic

## Domain areas where it is especially valuable

`boti-data` is intentionally general-purpose, but it is especially strong in domains where structured operational data must be transformed into reliable analytical or decision-ready datasets.

Examples include:

- **analytics engineering**: building reusable source loaders, schema maps, and standardised transformations
- **business operations**: consolidating data from transactional systems, planning tools, and operational databases
- **finance and controlling**: reconciling structured data with explicit schema expectations and repeatable joins
- **risk, compliance, and audit**: validating input shape, tracing transformations, and standardising access patterns
- **customer and product analytics**: joining behavioural and operational datasets with less custom plumbing
- **supply chain and logistics**: unifying inventory, movement, order, and status data from several systems
- **data platform and internal tooling**: giving teams a common gateway layer instead of ad hoc connectors
- **ML feature preparation**: building reliable dataset assembly steps from SQL and parquet sources

In those settings, the gains are not just convenience. They show up as better reuse, fewer integration bugs, and faster movement from exploration to production.

## Core capabilities

- SQL database resources
- async and sync database access helpers
- SQLAlchemy model reflection and registries
- connection catalogues
- parquet resources and readers
- gateway-style loading APIs
- filter expressions
- schema normalisation and validation helpers
- field mapping and join helpers
- partitioned and distributed data workflows

## Installation

Install directly:

```bash
pip install boti-data
```

Or install through the core package extra:

```bash
pip install "boti[data]"
```

## Imports

`boti-data` uses the top-level Python package `boti_data`:

```python
from boti_data import (
    ConnectionCatalog,
    DataGateway,
    DataHelper,
    FieldMap,
    ParquetDataConfig,
    ParquetDataResource,
    SqlAlchemyModelBuilder,
    SqlDatabaseConfig,
    SqlDatabaseResource,
)
```

Lower-level modules are also available:

```python
from boti_data.db import SqlDatabaseConfig, SqlDatabaseResource
from boti_data.gateway import DataGateway
from boti_data.parquet import ParquetDataConfig, ParquetDataResource
from boti_data.schema import validate_schema
```

## Examples

### SQL resource

```python
from boti_data import SqlDatabaseConfig, SqlDatabaseResource

config = SqlDatabaseConfig(connection_url="sqlite:///example.db", query_only=True)

with SqlDatabaseResource(config) as db:
    with db.session() as session:
        rows = session.execute(...)
```

### Gateway

```python
from boti_data import DataGateway, SqlDatabaseConfig

gateway = DataGateway(
    backend="sqlalchemy",
    config=SqlDatabaseConfig(connection_url="sqlite:///example.db", query_only=True),
)
```

## Relationship to `boti`

`boti-data` depends on `boti`, and reuses:

- logging
- resource lifecycle
- secure I/O helpers
- project/environment utilities

If you only need the runtime primitives, install `boti`.  
If you need a stronger data access and transformation layer, install `boti-data` or `boti[data]`.
