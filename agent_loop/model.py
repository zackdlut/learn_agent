from langchain_core.tools import BaseTool
from langchain_openai import ChatOpenAI

from .env import getenv
from .http_logger import make_async_client, make_sync_client


def create_model(tools: list[BaseTool]):
    model = ChatOpenAI(
        model=getenv("OPENAI_MODEL"),
        api_key=getenv("OPENAI_API_KEY"),
        base_url=getenv("OPENAI_BASE_URL"),
        http_client=make_sync_client(),
        http_async_client=make_async_client(),
    )
    return model.bind_tools(tools)
