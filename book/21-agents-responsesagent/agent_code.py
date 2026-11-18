"""Models-from-code agent with the ResponsesAgent interface (OpenAI Responses-style
input and output). MLflow stores this file plus the code_paths it imports."""
import uuid
from collections.abc import Generator

import mlflow
from mlflow.pyfunc import ResponsesAgent
from mlflow.types.responses import (
    ResponsesAgentRequest,
    ResponsesAgentResponse,
    ResponsesAgentStreamEvent,
)

from support_bot import answer


class SupportAgent(ResponsesAgent):
    def _question(self, request: ResponsesAgentRequest) -> str:
        user_messages = [item for item in request.input if getattr(item, "role", None) == "user"]
        return user_messages[-1].content if user_messages else ""

    def predict(self, request: ResponsesAgentRequest) -> ResponsesAgentResponse:
        text = answer(self._question(request))
        return ResponsesAgentResponse(output=[self.create_text_output_item(text=text, id=str(uuid.uuid4()))])

    def predict_stream(self, request: ResponsesAgentRequest) -> Generator[ResponsesAgentStreamEvent, None, None]:
        item_id = str(uuid.uuid4())
        text = answer(self._question(request))
        for word in text.split(" "):
            yield ResponsesAgentStreamEvent(**self.create_text_delta(delta=word + " ", item_id=item_id))
        yield ResponsesAgentStreamEvent(type="response.output_item.done", item=self.create_text_output_item(text=text, id=item_id))


mlflow.models.set_model(SupportAgent())
