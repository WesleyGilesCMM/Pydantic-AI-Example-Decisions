import uuid
from fastapi import APIRouter, HTTPException
from sqlmodel import select

from api.auth.dependencies import UserId
from api.models import DatabaseConnection
from api.models.decision import Decision, DecisionFullRead, DecisionListRead
from api.agents.decision_making_agent import decision_making_agent

router = APIRouter(prefix="/decision", tags=["Decision"])


@router.get("/")
async def get_decision_history(
    db: DatabaseConnection, user_id: UserId
) -> list[DecisionListRead]:
    statement = select(Decision).where(Decision.user_id == user_id)
    decisions = db.exec(statement).all()
    return decisions


@router.post("/")
async def create_new_decision(
    db: DatabaseConnection, user_id: UserId, prompt: str
) -> DecisionFullRead:
    res = await decision_making_agent.run(
        f"Please generate a flowchart for the following decision:\n{prompt}"
    )
    decision = res.output.to_sql(prompt, user_id)
    db.add(decision)
    db.commit()
    db.refresh(decision)
    decision = decision.to_model()
    return decision


@router.get("/{id}")
async def get_decision_by_id(
    id: uuid.UUID, db: DatabaseConnection, user_id: UserId
) -> DecisionFullRead:
    decision = db.get(Decision, id)
    if not decision:
        raise HTTPException(404, "Decision not found")
    if decision.user_id != user_id:
        raise HTTPException(403, "This decision is not yours")
    decision = decision.to_model()
    return decision
