import logging
import os
from enum import Enum
from typing import List, Optional, Tuple

from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
from overrides import override
from pydantic import BaseModel, Field

from dataherald.config import System
from dataherald.context_store import ContextStore
from dataherald.db import DB
from dataherald.db_scanner.models.types import ColumnDetail, TableDescriptionStatus
from dataherald.db_scanner.repository.base import TableDescriptionRepository
from dataherald.model.chat_model import ChatModel
from dataherald.repositories.golden_sqls import GoldenSQLRepository
from dataherald.repositories.instructions import InstructionRepository
from dataherald.types import GoldenSQL, GoldenSQLRequest, LLMConfig, Prompt

# we want to fetch semantic models back. Semantic models model an entity
# s.m. has dimensions (extracted from DDL), measures (can be inferred from DDL), metrics (we would probably construct this), join paths (from query history)

# Ideally, it selects what dimensions/measures it wants from semantic models. Then, the semantic model can compile down to the SQL to run it.
#

GENERATE_MEASURES_PROMPT = """
You are a proficient data scientist. You are given a table DDL command.

Your job is to generate a list of measures that can be applied to the table to extract valuable information. Aggregation functions include count, sum, average, min, max, and distinct count. The list should be concise and without redundant measures that would give the same information.

Here is the table schema:
{TABLE_SCHEMA}

Answer in the following format:
{OUTPUT_FORMAT}
"""

GENERATE_VIEWS_PROMPT = """
TODO
"""


logger = logging.getLogger(__name__)


class Join(BaseModel):
    path: str


class Measure(BaseModel):
    name: str = Field(description="Name of the measure")
    sql_operation: str = Field(description="The SQL operation to perform")
    description: str = Field(description="Simple description of the measure")


class CandidateMeasures(BaseModel):
    measures: list[Measure] = Field(description="List of candidate measures")
    reasoning: list[str] = Field(description="Reasoning for each measure")


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

    store_id: str = Field(description="unique id of corresponding TableDescription in the DB")
    relation_name: str = Field(description="name of the table/view")
    schema_name: str = Field(description="name of the schema where the table/view resides")
    dimensions: list[ColumnDetail] | None = Field(description="list of dimensions")
    measures: list[Measure] | None = None
    join_paths: list[Join] | None = None
    # obj_type: SemanticModelType # Need to add to the DB Scanner first

    def get_full_name(self):
        return f"{self.schema_name}.{self.relation_name}"

    def get_relation_schema(self, store: DB) -> str:
        table_desc_repo = TableDescriptionRepository(store)
        table_desc = table_desc_repo.find_by_id(self.store_id)
        if table_desc is None:
            raise ValueError(f"TableDescription with id {self.store_id} for {self.get_full_name()} not found")
        return table_desc.table_schema


class SemanticContextStore(ContextStore):
    def __init__(self, system: System):
        super().__init__(system)  # ignore the golden sql connection
        self.model_factory = ChatModel(self.system)
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

    def _generate_candidate_measures(self, semantic_model: SemanticModel) -> CandidateMeasures:
        # these are per relation, so we can use the semantic model
        prompt = ChatPromptTemplate.from_template(GENERATE_MEASURES_PROMPT)
        parser = JsonOutputParser(pydantic_object=CandidateMeasures)

        chain = prompt | self.llm | parser
        output = chain.invoke(
            {"TABLE_SCHEMA": semantic_model.get_relation_schema(self.db), "OUTPUT_FORMAT": parser.get_format_instructions()}
        )
        logger.info(output)
        return output

    def generate_semantic_models(self, db_connection_id: str, model: any):
        # can go to history and try to create views. ultimately end up in mongodb
        # bring in all tables and views from mongodb -> make into models
        # self.llm_config = llm_config

        # self.llm = self.model_factory.get_model(
        #     database_connection=db_connection_id, temperature=0, model_name=self.llm_config.llm_name, api_base=self.llm_config.api_base
        # )
        self.llm = model

        logger.info(f"Generating Semantic Models for db conn {db_connection_id}")

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
            SemanticModel(store_id=table.id, relation_name=table.table_name, schema_name=table.schema_name, dimensions=table.columns)
            for table in db_scan
        ]

        logger.info(f'DB Scan finished. Example Semantic Model: {semantic_models[0].dict()}')

        for i, model in enumerate(semantic_models):
            candidate_measures = self._generate_candidate_measures(model)
            if i == 0:
                logger.info(candidate_measures.dict())
            model.measures = candidate_measures.measures

        logger.info(f'Measure generation finished. Example Semantic Model: {semantic_models[0].dict()}')

        # to generate views, the proper way is to get all possible tables that can be joined.
        # That could be through foreign key relations, asking LLM to compare column names (maybe weighting inside schema more), etc.
        # alternatively, we can insert all the current semantic models in the context store and similarity search for each one to generate
        # some neighbors. create a view using that. perhaps there will be duplicate views, to solve then.

        self.vector_store.add_record(
            documents=[model.dict() for model in semantic_models],
            ids=[model.get_full_name() for model in semantic_models],
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
