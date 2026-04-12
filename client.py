from typing import Dict

from openenv.core import EnvClient
from openenv.core.client_types import StepResult
from openenv.core.env_server.types import State

from server.models import MyAction, MyObservation

class myEnv(EnvClient[MyAction, MyObservation, State]):

    def _step_payload(self, action: MyAction) -> Dict:
        return {
            "action": action.dict()
        }
    
    def _parse_result(self, payload: Dict) -> StepResult[MyObservation]:
        obs_data = payload.get("observation", {})

        observation = MyObservation(
            time_step=obs_data.get("time_step", 0),
            active_incidents=obs_data.get("active_incidents", []),
            resources=obs_data.get("resources", []),
            total_people_affected=obs_data.get("total_people_affected", 0),
            resolved_incidents=obs_data.get("resolved_incidents", 0),
            system_load=obs_data.get("system_load", 0),
            response_efficiency=obs_data.get("response_efficiency", 0),
        )

        return StepResult(
            observation=observation,
            reward=payload.get("reward", 0),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: Dict) -> State:
        return State(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
        )