import datetime
from typing import Annotated, Literal
import uuid
from sqlmodel import TEXT, SQLModel, Field, Column, select
from pgvector.sqlalchemy import Vector
from api.utils import utc_now, embed_text


class Negative(SQLModel):
    result: Literal["negative"] = Field(
        "negative",
        description="A determination that the user should *NOT* move forward with the decisions",
    )
    reasoning: str = Field(
        description="A detailed reasoning on why the user should *NOT* move forward with the decision"
    )


class Affirmative(SQLModel):
    result: Literal["affirmative"] = Field(
        "affirmative",
        description="A determination that the user should move forward with the decision",
    )
    name: str = Field(
        description="The name of the option the user should move forward with in this decision"
    )
    reasoning: str = Field(
        description="A detailed reasoning on why the user should move forward with the decision"
    )


class ConsiderationTopic(SQLModel):
    name: str = Field(
        description="The name of the topic to consider more deeply, or context needed"
    )
    description: str = Field(
        description="A detailed description of the topic to consider more deeply, or context needed"
    )


class RequiresConsideration(SQLModel):
    result: Literal["requires_deeper_consideration"] = Field(
        "requires_deeper_consideration",
        description="A determination that the decision requires deeper consideration, or more context to make a decision",
    )
    reasoning: str = Field(
        description="A detailed reasoning on why the product requires deeper consideration"
    )
    consideration_topics: list[ConsiderationTopic] = Field(
        default_factory=list,
        description="Topics or context that are required to clarify the decision",
    )


determination = Annotated[
    Affirmative | Negative | RequiresConsideration,
    Field(
        description="The final determination of the decision based on the location in the flow chart",
        discriminator="result",
    ),
]


class FlowChartNode(SQLModel):
    question: str = Field(
        description="A boolean (yes or no) question to determine the next step in the flow chart, to be answered by the users"
    )
    if_yes: "FlowChartNode" | determination = Field(
        description="The next step in the flowchart if the answer to the question is yes"
    )
    if_no: "FlowChartNode" | determination = Field(
        description="The next step in the flowchart if the answer to the question is no"
    )


class DecisionRequest(SQLModel):
    prompt: str
    created_at: datetime.datetime = Field(default_factory=utc_now)
    updated_at: datetime.datetime = Field(
        default_factory=utc_now, sa_column_kwargs={"onupdate": utc_now}
    )


class DecisionFlowChart(SQLModel):
    flow_chart: FlowChartNode = Field(
        description="The flowchart which the user should follow to make the decision"
    )

    def to_sql(self, prompt: str, user_id: uuid.UUID) -> "Decision":
        return Decision(
            prompt=prompt,
            # embedding=embed_text(prompt), # Uncomment and adjust dimension as needed when using embeddings
            flow_chart=self.flow_chart.model_dump_json(indent=1),
            user_id=user_id,
        )


class Decision(DecisionRequest, table=True):
    id: uuid.UUID = Field(default_factory=uuid.uuid4, primary_key=True)
    # embedding: list[float] = Field(sa_column=Column(Vector(1536))) # Uncomment and adjust dimension as needed when using embeddings
    flow_chart: str = Field(sa_type=TEXT)
    user_id: uuid.UUID

    def to_model(self) -> "DecisionFullRead":
        return DecisionFullRead(
            **self.model_dump(exclude=["embedding", "flow_chart"]),
            flow_chart=FlowChartNode.model_validate_json(self.flow_chart)
        )


class DecisionListRead(DecisionRequest):
    id: uuid.UUID


class DecisionFullRead(DecisionListRead):
    flow_chart: FlowChartNode
