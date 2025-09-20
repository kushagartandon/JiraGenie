# main.py
import asyncio
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai.types import Content, Part

from jira_genie.agent import root_agent

#  Assuming the agent code is in a file named 'agent.py'

async def main():
    # Initialize the session service. InMemorySessionService is for local testing.
    session_service = InMemorySessionService()

    APP_NAME = "jira_genie"
    USER_ID = "user_123"
    SESSION_ID = "session_123"

    # Create a new conversation session
    session = await session_service.create_session(
        app_name=APP_NAME,
        user_id=USER_ID,
        session_id=SESSION_ID
    )

    print(session)

    # Initialize the Runner with your agent and the session service
    runner = Runner(
        agent=root_agent,
        app_name=APP_NAME,
        session_service=session_service
    )

    # Define the user's query
    user_query = "What are thr jira projects you have access to?"
    print(f"User: {user_query}")

    # Format the query into an ADK Content object
    content = Content(role='user', parts=[Part(text=user_query)])

    # Run the agent asynchronously and iterate through the events
    async for event in runner.run_async(user_id=USER_ID, session_id=SESSION_ID, new_message=content):
        if event:
            print(f"Agent: {event.content} by {event.author}")

# Run the main function
if __name__ == "__main__":
    asyncio.run(main())