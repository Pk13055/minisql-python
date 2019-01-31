# Mini SQL Engine

## Introduction

Supports a basic SQL query setup which uses files for its storage. 
Record type is basic integer and basic queries are **only** supported.

## Types of queries

- Select records

```
select * from table_name;
select col1 from table_name;
```

- Aggregate functions: functions on a **single** column.
    - Sum
    - Average
    - Max
    - Min

```
select max(col1) from table1;
```

## Examples

1. Select * from table2;

```bash
table1.B,table2.D
158,11191
773,14421
85,5117
811,13393
311,16116
646,5403
335,6309
803,12262
718,10226
731,13021
```

2.Select A from table1;

```bash
table1.A
922
640
775
-551
-952
-354
-497
411
-900
858
```

