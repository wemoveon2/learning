import logging
import os
import random
from datetime import datetime, timedelta
from typing import List, Set

import sqlalchemy
from airflow import DAG
from airflow.exceptions import AirflowFailException, AirflowSkipException
from airflow.operators.python import BranchPythonOperator, PythonOperator

MODULE_NAME = os.path.splitext(os.path.basename(__file__))[0]
DAG_ID = MODULE_NAME

DB_HOST = "postgres"
DB_PORT = "5432"
DB_USERNAME = "airflow"
DB_PASSWORD = "airflow"
DB_DATABASE = "airflow"

DB_URI = sqlalchemy.engine.url.URL(
    "postgresql+psycopg2",
    username=DB_USERNAME,
    password=DB_PASSWORD,
    host=DB_HOST,
    port=DB_PORT,
    database=DB_DATABASE,
)

JOB_TABLE_NAME = "dummy_job"
JOB_RESULT_TABLE_NAME = "dummy_job_result"

QUERY_JOB_TABLE_CREATION = f"""
    CREATE TABLE IF NOT EXISTS {JOB_TABLE_NAME} (
        id SERIAL PRIMARY KEY,
        is_active BOOLEAN NOT NULL DEFAULT TRUE
    );
"""
QUERY_JOB_RESULT_TABLE_CREATION = f"""
    CREATE TABLE IF NOT EXISTS {JOB_RESULT_TABLE_NAME} (
        job_id INT UNIQUE NOT NULL REFERENCES {JOB_TABLE_NAME}(id),
        is_gt_th BOOLEAN,
        is_successful BOOLEAN
    );
"""

# The string may be changed to another one with length ranging from 0 to
# 1 million in different test cases.
NUM_STR = "1867-07-01T00:00:00.000"

RAND_DIGIT_AMOUNT = 10**6
HIT_COUNT_THRESHOLD = RAND_DIGIT_AMOUNT // 2

# IDs of all Airflow tasks in the DAG
TASK_ID_CREATE_TABLES = "create_tables"
TASK_ID_INSERT_RECS = "insert_recs"
TASK_ID_GET_HIT_COUNT = "get_hit_count"
TASK_ID_BRANCHING = "branching"
TASK_ID_ACTION_ON_LTE_THRESHOLD = "action_on_lte_threshold"
TASK_ID_ACTION_ON_GT_THRESHOLD = "action_on_gt_threshold"
TASK_ID_ACTION_ON_ERROR = "action_on_error"

random.seed()

# Helper functions/classes ####################################################


def get_rand_digit() -> int:
    """Return a random digit between 0 and 9 (inclusive)."""
    return random.randint(0, 9)


def get_engine() -> sqlalchemy.engine.Engine:
    return sqlalchemy.create_engine(DB_URI)


def is_raise_error() -> bool:
    """Returns True with a probability of 0.3."""
    return get_rand_digit() < 3


# Functions for the tasks in the DAG ##########################################


def create_tables(**kwargs) -> None:
    """Create table 'dummy_job' and table 'dummy_job_result' if they do not
    exist.

    This function will be run in a DAG task with task ID
    [TASK_ID_CREATE_TABLES].

    The query for the table creation can be found in variable
    [QUERY_JOB_TABLE_CREATION] and variable [QUERY_JOB_RESULT_TABLE_CREATION].

    The tables can be created either by executing the provided raw query,
    or by using ORM (based on the information in the provided query).
    """

    with get_engine().begin() as conn:
        try:
            conn.execute(sqlalchemy.text(QUERY_JOB_TABLE_CREATION))
            conn.execute(sqlalchemy.text(QUERY_JOB_RESULT_TABLE_CREATION))
        except Exception as e:
            logging.error(e)
            raise AirflowFailException


def insert_recs(ti, **kwargs) -> None:
    """Insert a new record into table 'dummy_job' and table
    'dummy_job_result' respectively.

    This function will be run in a DAG task with task ID
    [TASK_ID_INSERT_RECS].

    Actions:
        - insert a record into table [JOB_TABLE_NAME]. The DB will
            automatically generate an integer for column 'id'. We regard
            this ID as the job ID for the current pipeline run. Store that
            value in a way that other DAG task in the same run can access.
            No need to set values for any other columns.
        - insert a record into table [JOB_RESULT_TABLE_NAME], with the value
            in column 'job_id' being the same with the 'id' value obtained
            in the previous task. No need to set values for any other columns.
        - add other thing you think necessary.
    """

    with get_engine().begin() as conn:
        try:
            job_id = conn.execute(
                sqlalchemy.text(
                    f"INSERT INTO {JOB_TABLE_NAME} (is_active) VALUES (:is_active) RETURNING id"
                ),
                **{"is_active": True},
            ).fetchone()[0]
            conn.execute(
                sqlalchemy.text(
                    f"INSERT INTO {JOB_RESULT_TABLE_NAME} (job_id) VALUES (:job_id)"
                ),
                **{"job_id": job_id},
            )
            print(f"Job ID#{job_id}")
            ti.xcom_push(key="job_id", value=job_id)
            logging.info(
                f"""insert_recs(): Sucessfully inserted job #{job_id} 
                        into {JOB_TABLE_NAME} and {JOB_RESULT_TABLE_NAME}"""
            )
        except Exception as e:
            logging.error(e)
            raise AirflowFailException


def get_hit_count(num_str: str, rand_digit_amount: int, ti, **kwargs) -> None:
    """
    Args:
        num_str: a string that contains ASCII characters.
        rand_digit_amount: total amount of random digits to generate.

    This function will be run in a DAG task with task ID
    [TASK_ID_GET_HIT_COUNT].

    Actions:
      - randomly generate a random digit (from 0 to 9) and tell whether
        it can be found in string [num_str].
      - repeat the above step for [rand_digit_amount] times.
      - store the amount of times that a random digit is found in the given
          string, such that the value can be accessed by other DAG tasks
          in the same run.
      - add other thing you think necessary.
    """
    # Simulate a scenario that this task fails for unknown reasons
    if is_raise_error():
        raise ValueError

    assert isinstance(num_str, str) and isinstance(rand_digit_amount, int)

    hits: int = 0
    num_set: Set[str] = set(num_str)
    for _ in range(rand_digit_amount):
        hits += 1 if str(get_rand_digit()) in num_set else 0
    ti.xcom_push(key="hit_count", value=hits)
    logging.info(f"get_hit_count(): Got {hits} hits")


def branching(ti, **kwargs) -> List[str]:
    """Returns the list of task IDs for the next tasks.

    This function will be run in a DAG task with task ID [TASK_ID_BRANCHING].
    That DAG task is based on 'BranchPythonOperator'. Please refer to the
    official documentation for 'BranchPythonOperator'.

    Actions:
        - Get the hit count in task [TASK_ID_GET_HIT_COUNT].
        - If the hit count in [TASK_ID_GET_HIT_COUNT] is greater than
            [HIT_COUNT_THRESHOLD], the next task ID would be
            [TASK_ID_ACTION_ON_GT_THRESHOLD].
        - Otherwise, the next task ID would be
            [TASK_ID_ACTION_ON_LTE_THRESHOLD].
        - add other thing you think necessary.
    """

    count: int = ti.xcom_pull(key="hit_count", task_ids=TASK_ID_GET_HIT_COUNT)

    assert isinstance(HIT_COUNT_THRESHOLD, int)
    assert isinstance(count, int)

    if count > HIT_COUNT_THRESHOLD:
        logging.info(f"""branching():{count} is greater than {HIT_COUNT_THRESHOLD}""")
        return [TASK_ID_ACTION_ON_GT_THRESHOLD]
    elif count <= HIT_COUNT_THRESHOLD:
        logging.info(
            f"branching():{count} is less than or equals to {HIT_COUNT_THRESHOLD}"
        )
        return [TASK_ID_ACTION_ON_LTE_THRESHOLD]
    else:
        raise AirflowFailException


def action_on_gt_threshold(ti, **kwargs) -> None:
    """Actions to be taken when the hit count is greater than the given
    threshold.

    This function will be run in a DAG task with task ID
    [TASK_ID_ACTION_ON_GT_THRESHOLD].

    Actions:
        - Get the job ID generated in task [TASK_ID_INSERT_RECS].
        - Update the record created in task [TASK_ID_INSERT_RECS] in table
            [JOB_TABLE_NAME]:
            - set column 'is_active' to be 'false'
        - Update the record created in task [TASK_ID_INSERT_RECS] in table
            [JOB_TABLE_RESULT_NAME]:
            - set column 'is_successful' to be 'true'
            - set column 'is_gt_th' to be 'true'
        - add other thing you think necessary.
    """

    job_id: str = ti.xcom_pull(key="job_id", task_ids=TASK_ID_INSERT_RECS)
    with get_engine().begin() as conn:
        try:
            conn.execute(
                sqlalchemy.text(
                    f"UPDATE {JOB_TABLE_NAME} SET is_active=:is_active WHERE id=:job_id"
                ),
                **{"is_active": False, "job_id": job_id},
            )
            conn.execute(
                sqlalchemy.text(
                    f"UPDATE {JOB_RESULT_TABLE_NAME} SET is_successful=:is_successful, is_gt_th=:is_gt_th WHERE job_id=:job_id"
                ),
                **{"job_id": job_id, "is_successful": True, "is_gt_th": True},
            )
            logging.info(f"action_on_gt_threshold(): Job #{job_id} recorded")
        except:
            raise AirflowFailException


def action_on_lte_threshold(ti, **kwargs) -> None:
    """Actions to be taken when the hit count is less than or equal to the
    given threshold.

    This function will be run in a DAG task with task ID
    [TASK_ID_ACTION_ON_LTE_THRESHOLD].

    Actions:
        - Get the job ID generated in task [TASK_ID_INSERT_RECS].
        - Update the record created in task [TASK_ID_INSERT_RECS] in table
            [JOB_TABLE_NAME]:
            - set column 'is_active' to be 'false'
        - Update the record created in task [TASK_ID_INSERT_RECS] in table
            [JOB_TABLE_RESULT_NAME]:
            - set column 'is_successful' to be 'true'
            - set column 'is_gt_th' to be 'false'
        - add other thing you think necessary.
    """

    job_id: str = ti.xcom_pull(key="job_id", task_ids=TASK_ID_INSERT_RECS)
    with get_engine().begin() as conn:
        try:
            conn.execute(
                sqlalchemy.text(
                    f"UPDATE {JOB_TABLE_NAME} SET is_active=:is_active WHERE id=:job_id"
                ),
                **{"is_active": False, "job_id": job_id},
            )
            conn.execute(
                sqlalchemy.text(
                    f"UPDATE {JOB_RESULT_TABLE_NAME} SET is_successful=:is_successful, is_gt_th=:is_gt_th WHERE job_id=:job_id"
                ),
                **{"job_id": job_id, "is_successful": True, "is_gt_th": False},
            )
            logging.info(f"action_on_lte_threshold(): Job #{job_id} recorded")
        except:
            raise AirflowFailException


def action_on_error(ti, **kwargs) -> None:
    """Actions to be taken when an exception is raised.

    This function will be run in a DAG task with task ID
    [TASK_ID_ACTION_ON_ERROR].

    Actions:
        - Get the job ID generated in task [TASK_ID_INSERT_RECS].
        - Update the record created in task [TASK_ID_INSERT_RECS] in table
            [JOB_TABLE_NAME]:
            - set column 'is_active' to be 'false'
        - Update the record created in task [TASK_ID_INSERT_RECS] in table
            [JOB_TABLE_RESULT_NAME]:
            - set column 'is_successful' to be 'false'
            - the value in column 'is_gt_th' doesn't matter
        - add other thing you think necessary.
    """

    job_id: str = ti.xcom_pull(key="job_id", task_ids=TASK_ID_INSERT_RECS)
    with get_engine().begin() as conn:
        try:
            conn.execute(
                sqlalchemy.text(
                    f"UPDATE {JOB_TABLE_NAME} SET is_active=:is_active WHERE id=:job_id"
                ),
                **{"is_active": False, "job_id": job_id},
            )
            conn.execute(
                sqlalchemy.text(
                    f"UPDATE {JOB_RESULT_TABLE_NAME} SET is_successful=:is_successful, is_gt_th=:is_gt_th WHERE job_id=:job_id"
                ),
                **{"job_id": job_id, "is_successful": False, "is_gt_th": None},
            )
            logging.error(f"action_on_error(): Encountered error on job {job_id}")
        except:
            raise AirflowFailException


# DAG creation ##########################################

default_args = {"owner": DAG_ID[4:], "end_date": "2100-01-01", "depends_on_past": False}

dag_kwargs = {
    "dag_id": DAG_ID,
    "default_args": default_args,
    "description": "EQ Work Sample",
    "schedule_interval": timedelta(minutes=1),
    "start_date": datetime(2022, 1, 1),
    "max_active_runs": 1,
    "catchup": False,
    "is_paused_upon_creation": False,
}

with DAG(**dag_kwargs) as dag:
    task_create_tables = PythonOperator(
        task_id=TASK_ID_CREATE_TABLES,
        python_callable=create_tables,
    )
    task_insert_recs = PythonOperator(
        task_id=TASK_ID_INSERT_RECS,
        python_callable=insert_recs,
    )
    task_get_hit_count = PythonOperator(
        task_id=TASK_ID_GET_HIT_COUNT,
        python_callable=get_hit_count,
        op_kwargs={"num_str": NUM_STR, "rand_digit_amount": RAND_DIGIT_AMOUNT},
    )
    task_branching = BranchPythonOperator(
        task_id=TASK_ID_BRANCHING,
        python_callable=branching,
    )
    task_action_on_lte_threshold = PythonOperator(
        task_id=TASK_ID_ACTION_ON_LTE_THRESHOLD,
        python_callable=action_on_lte_threshold,
    )
    task_action_on_gt_threshold = PythonOperator(
        task_id=TASK_ID_ACTION_ON_GT_THRESHOLD,
        python_callable=action_on_gt_threshold,
    )
    task_action_on_error = PythonOperator(
        task_id=TASK_ID_ACTION_ON_ERROR,
        python_callable=action_on_error,
        trigger_rule="one_failed",
    )

    (
        task_create_tables
        >> task_insert_recs
        >> task_get_hit_count
        >> [task_branching, task_action_on_error]
    )
    task_branching >> [task_action_on_lte_threshold, task_action_on_gt_threshold]
