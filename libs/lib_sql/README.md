# lib_sql

`lib_sql` provides a collection of shared SQL utilities for database interactions. It includes a composable system of "SQL writers" for writing data points to a SQL database, designed to be layered to combine functionalities.

## SQL Writers

The SQL writers are designed for composability, allowing different functionalities to be layered in a specific order. A core `SqlWriter` protocol defines the interface, and concrete implementations or wrappers provide capabilities like buffering, schema management, and data transformation. Each component is focused on a single responsibility.

The composition follows a strict hierarchy: Collectors (top), Transforms (middle), and Core (bottom).

### 1. Collectors (Top Layer)

Collectors sit at the top of the composition stack. They can wrap other writers (Transforms or Core writers) but cannot be wrapped by other components. They are responsible for reshaping data before it is passed down the chain.

- **`PointCollectingSqlWriter`**: Collects individual metric-value pairs from multiple `write_point` calls into an internal dictionary. When `collect_row()` is called, it uses a `row_factory` function to assemble the collected points into a single row, which is then passed to the next writer.

### 2. Transforms (Middle Layer)

Transforms are the middle layer, responsible for processing rows that are passed to them. They can wrap Core writers or other Transforms, and can be wrapped by Collectors.

- **`BufferedSqlWriter`**: An in-memory, watermark-based buffer that collects rows and flushes them to an underlying core writer. It uses a single background thread to perform writes. A flush is triggered asynchronously when the buffer size exceeds a `low_watermark`, and it blocks synchronously if a `high_watermark` is reached to prevent excessive memory usage.

### 3. Core Writers (Bottom Layer)

The core writer is at the bottom of the composition stack and is responsible for direct communication with the database. It cannot wrap other writers.

- **`DynamicSchemaSqlWriter`**: This writer inspects the keys of incoming Python `dict` objects and ensures corresponding columns exist in the target database table. If a key does not have a matching column, it issues an `ALTER TABLE ADD COLUMN` statement. This is useful for development and scenarios where the data schema may evolve, but it carries performance overhead due to the continuous schema checks.
- **`StaticSchemaSqlWriter`**: A writer for when the table schema is fixed and known ahead of time. It does not perform any schema inspection or modification, resulting in better performance than its dynamic counterpart.

## Composition Examples

### Example 1: Collector -> Transform -> Core

A typical setup involves a `PointCollectingSqlWriter` (Collector) that sends its output to a `BufferedSqlWriter` (Transform), which in turn wraps a `DynamicSchemaSqlWriter` (Core). This is useful for collecting metrics where the schema might change.

```python
# Pseudocode for writer composition
engine = create_engine(...)
core_writer = DynamicSchemaSqlWriter(engine, "my_table")
transform = BufferedSqlWriter(core_writer)
collector = PointCollectingSqlWriter(transform, row_factory=dict)

# Application code can now write individual data points
collector.write_point("metric_a", 1.0)
collector.write_point("metric_b", 42)
collector.collect_row() # Assembles and sends the row to the transform
```

In this example, the data flows as follows:
1. The application calls `collector.write_point()` multiple times.
2. `collector.collect_row()` assembles the points into a dictionary and passes it to `transform.write()`.
3. The `transform` (a `BufferedSqlWriter`) holds the row in its internal buffer until a flush is triggered.
4. Upon flushing, the `transform` passes the batch of rows to `core_writer.write_many()`.
5. The `core_writer` checks if the table schema matches the data and then inserts the rows into the database.

### Example 2: Transform -> Core

In some cases, a Collector is not needed. Data can be written directly to a Transform that wraps a Core writer. This pattern is suitable for writing to tables with a "tall" format and a predefined schema, such as an `evals` table.

```python
# Pseudocode for writer composition
core_writer = StaticSchemaSqlWriter(engine, "evals_table")
transform = BufferedSqlWriter(core_writer)

# Application can now write rows directly to the transform
transform.write({"episode": 1, "metric_name": "return", "value": 123.4})
transform.write({"episode": 1, "metric_name": "length", "value": 50})
```

In this simpler pipeline:
1. The application calls `transform.write()` with complete row data.
2. The `transform` buffers the rows.
3. Upon flushing, the `transform` passes the batch to the `core_writer`, which inserts them into the database without performing schema checks.
