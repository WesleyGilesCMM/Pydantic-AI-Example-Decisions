from dotenv import load_dotenv

from api.agents.decision_making_agent import decision_making_agent

load_dotenv(override=True)
from api import app

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run the decision making agent")
    parser.add_argument("-d", "--decision", type=str, default="Should I invest in CoverMyMeds stock~?", help="The decision to evaluate")
    args = parser.parse_args()
    flowchart = decision_making_agent.run_sync(args.decision).output
    print(flowchart.model_dump_json(indent=2))