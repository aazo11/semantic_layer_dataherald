import logging
import os
from enum import Enum
from typing import List, Optional, Tuple

from overrides import override
from pydantic import BaseModel, Field

from dataherald.config import System
from dataherald.context_store import ContextStore
from dataherald.db_scanner.models.types import ColumnDetail, TableDescriptionStatus
from dataherald.db_scanner.repository.base import TableDescriptionRepository
from dataherald.model.chat_model import ChatModel
from dataherald.repositories.golden_sqls import GoldenSQLRepository
from dataherald.repositories.instructions import InstructionRepository
from dataherald.types import GoldenSQL, GoldenSQLRequest, Prompt

# we want to fetch semantic models back. Semantic models model an entity
# s.m. has dimensions (extracted from DDL), measures (can be inferred from DDL), metrics (we would probably construct this), join paths (from query history)

# Ideally, it selects what dimensions/measures it wants from semantic models. Then, the semantic model can compile down to the SQL to run it.
#

GENERATE_MEASURES_PROMPT = """
You are a proficient data scientist and you job is to help a company to generate some candidate aggregation functions that can extract valuable information from a given table.

You are given a table ddl command and you job is to generate a list of candidate measures that can be applied to the table.

Generate all of the possible measures that can be applied to the table. Aggregation functions include count, sum, average, min, max, and distinct count.

Here is the table schema:
{TABLE_SCHEMA}

Your response should be a valid JSON object as follow without any other information:
[
    {{
        "name": "employee_count",
        "column": "employee_id",
        "aggregation": "count",
        "description": "The number of employees."
    }},
    {{
        "name": "total_salary",
        "column": "salary",
        "aggregation": "sum",
        "description": "The total salary of all employees."
    }}
    ...
]
"""


logger = logging.getLogger(__name__)


class Join(BaseModel):
    path: str


class Measure(BaseModel):
    name: str
    sql_operation: str


# class View(BaseModel):
#     name: str
#     required_joins: list[str]
#     description: str
#     sql_operation: str


class SemanticModelType(str, Enum):
    table = "table"
    view = "view"


class SemanticModel(BaseModel):
    """Semantically represents an entity in the database, which is a view or table."""

    obj_name: str
    schema_name: str
    # obj_type: SemanticModelType # Need to add to the DB Scanner first
    dimensions: list[ColumnDetail] | None = None
    measures: list[Measure] | None = None
    join_paths: list[Join] | None = None


class SemanticContextStore(ContextStore):
    def __init__(self, system: System):
        super().__init__(system)  # ignore the golden sql connection
        self.llm = ChatModel(self.system)
        self.semantic_collection = os.environ.get("SEMANTIC_COLLECTION", "semantic-stage")

    @override
    def retrieve_context_for_question(self, prompt: Prompt, number_of_samples: int = 3) -> Tuple[List[dict] | None, List[dict] | None]:
        # search for related semantic models here
        self.db_connection_id = prompt.db_connection_id  # TODO: Safe to pull this?
        logger.info(f"Getting semantic layers as context for {prompt.text}")
        models = self.vector_store.query(
            query_texts=[prompt.text],
            db_connection_id=self.db_connection_id,
            collection=self.semantic_collection,
            num_results=number_of_samples,
            convert_pinecone=False,
        )

        return models, None

    @override
    def generate_semantic_models(self, db_connection_id: str):
        # can go to history and try to create views. ultimately end up in mongodb
        # bring in all tables and views from mongodb -> make into models
        tables_desc_repository = TableDescriptionRepository(self.db)
        db_scan = tables_desc_repository.get_all_tables_by_db(
            {
                "db_connection_id": db_connection_id,
                "status": TableDescriptionStatus.SCANNED.value,
            }
        )
        if not db_scan:
            raise ValueError("No scanned tables found for database")

        semantic_models = [
            SemanticModel(obj_name=table.table_name, schema_name=table.schema_name, dimensions=table.columns) for table in db_scan
        ]

        logger.info(semantic_models[0].dict())

        self.vector_store.add_record(
            documents=[model.dict() for model in semantic_models],
            ids=[model.schema_name + "." + model.obj_name for model in semantic_models],
            metadata=None,
            db_connection_id=None,  # unused
            collection=self.semantic_collection,
        )

    @override
    def add_golden_sqls(self, golden_sqls: List[GoldenSQLRequest]) -> List[GoldenSQL]:
        raise NotImplementedError

    @override
    def remove_golden_sqls(self, ids: List) -> bool:
        raise NotImplementedError
