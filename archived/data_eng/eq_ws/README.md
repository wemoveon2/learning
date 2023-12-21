# README

Data engineering work samples for EQ Works. Utilizes Apache Spark, Apache Airflow, and SQLAlchemy to interact with a Postgres database. 

Solutions for the common questions can be found [here](https://github.com/wemoveon2/data_eng_work_sample/blob/main/common_problems.ipynb).
The DAG file for Airflow can be found [here](https://github.com/wemoveon2/data_eng_work_sample/blob/main/dag_candidate_final.py).

## Common Problems

**All** records with **identical** `geoinfo` **and** `TimeSt` are removed from `~/data/DataSample.csv`. Records with duplicated `TimeSt` is kept. Records with `TimeSt` differences in the range of miliseconds are not considered as identical even with identical `geoinfo` due to velocity of requests. 

A PySpark DataFrame containing the `_ID` of non-suspicious records and their `POIID` assigned based on shortest haversine distance (km) is created as the final output for downstream analysis.

## Data Track Problems

The DAG file requires the `postgres` connection at port 5432 as seen [here](https://github.com/wemoveon2/data_eng_work_sample/blob/main/ws-data-pipeline/docker-compose.yml). The created DAG runs every minute, logging its progress and results in a `postgres` database at each run. All explicit database operations utilizes `SQLAlchemy`.

- DAG/Pipeline Config
	- [x] Scheduled to run every minute.
	- [x] Pipeline automatically scheduled to run without manual triggering when placed in `./dag` folder.
	- [x] No backfilling.
	- [x] No retrying if one run fails.
	- [x] Scheduled to run even if previous fails (don't set `depends_on_past=True` for tasks).
- Pipeline Tasks
	- [x] 1. Create tables if they don't exist.
	- [x] 2. Insert new records into tables.
	- [x] 3. Generate given number of digits from 0 to 9 and get count within string.
	- [x] 4. Runs only if task 3 fails, updates records created by task 2, ends pipeline.
	- [x] 5. Runs if 3 successful, branches depending on if value from task 3 is G/L than threshold.
	- [x] 6. Update records created by 2 to indicate count is greater, ends pipeline.
	- [x] 7. Updates records created by 2 to indicate count is lesser, ends pipeline.
- Feedback
	- Clean Up & Label:
		- [x] Remove extra de-dupe on timestamp.
	- Correctness: 
		- [x] Ensure [get_hit_count] is implemented as expected.
	- Cleanness & Maintainability:
		- [ ] Refactor to shorten lines and to ensure adherence DRY.
			- Should replace explicit queries with function.
	- Readability:
		- [x] Add type hints for variables defined in functions. 
	- Robustness:
		- [x] Add validation for task inputs.
	- Efficiency:
		- [x] Determine bottlenecks in code.
			- The issue was in using a list in `get_hit_count`, fixed by using `set`


