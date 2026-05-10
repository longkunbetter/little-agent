from __future__ import annotations

from .agent import AgentConfig, StockAgent
from .llm import OpenAIChatCompletionsClient
from .tools import build_default_registry


def main() -> None:
    try:
        client = OpenAIChatCompletionsClient.from_env()
    except RuntimeError as exc:
        print(exc)
        return

    agent = StockAgent(
        client=client,
        tools=build_default_registry(),
        config=AgentConfig(),
    )
    agent.run_cli()


if __name__ == "__main__":
    main()
