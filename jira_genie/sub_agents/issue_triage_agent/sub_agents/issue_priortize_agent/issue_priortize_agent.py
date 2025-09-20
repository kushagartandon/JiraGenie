import re
from transformers import pipeline

from google.adk.agents import LlmAgent
from .....utils.jira_connector import JiraConnector

jira = JiraConnector.getConnection()

def get_issue_details(issue_key: str) -> dict:
  """
  Fetch details of a Jira issue given its key.
  Args:
    issue_key (str): The key of the Jira issue (e.g., "PROJ-123").
  Returns:
    dict: A dictionary containing issue details such as summary, description, priority, and status etc.
  """
  issue = jira.issue(issue_key)
  return {
    "issue_key": issue.key,
    "summary": issue.fields.summary,
    "description": issue.fields.description or "",
    "priority": issue.fields.priority.name if issue.fields.priority else "None",
    "status": issue.fields.status.name,
    "created": issue.fields.created,
    "updated": issue.fields.updated,
    "assignee": issue.fields.assignee.displayName if issue.fields.assignee else "Unassigned",
    "reporter": issue.fields.reporter.displayName if issue.fields.reporter else "Unknown",
  }


def set_issue_priority(issue_key: str, agent_suggested_priority: str) -> dict:
  """
  Set the priority of a Jira issue based on agent's suggestion and .
  Args:
    issue_key (str): The key of the Jira issue (e.g., "PROJ-123").
    agent_suggested_priority (str): The priority level suggested by the agent (e.g., "High", "Medium", "Low").
  Returns:
    dict: A dictionary confirming the priority update containing issue key and new priority.

  """
  issue = jira.issue(issue_key)
  issue.update(fields={"priority": {"name": agent_suggested_priority}})  # Example: Set priority to High
  print(f"\n-----------Priority for issue {issue.key} set to {agent_suggested_priority}.------------\n")
  return {
    "issue_key": issue.key,
    "suggested_priority": agent_suggested_priority,
    "message": f"Priority for issue {issue.key} set to {agent_suggested_priority}."
  }


def analyze_issue_priority(issue_key: str) -> str:
  """
    Analyze a Jira issue and suggest a priority using NLP.
    Heuristics:
    - Look at keywords in summary/description
    - Use sentiment polarity (negative urgency signals high priority)
    - Consider story points / custom fields if available
  """
  sentiment_analyzer = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english") 

  issue = jira.issue(issue_key)
  description = issue.fields.description or ""
  summary = issue.fields.summary or ""
  text_to_analyze = f"{summary}\n\n{description}"

  if not text_to_analyze.strip():
    return "Medium"  # fallback if no text available

  # Run sentiment analysis
  result = sentiment_analyzer(text_to_analyze[:512])[0]  # truncate to 512 tokens
  sentiment = result["label"]
  score = result["score"]

  # Simple heuristic for priority mapping
  if sentiment == "NEGATIVE" and score > 0.75:
    return "High"
  elif sentiment == "NEGATIVE":
    return "Medium"
  elif sentiment == "POSITIVE" and score > 0.80:
    return "Low"
  else:
    return "Medium"
  

issue_priortize_agent = LlmAgent(
  model="gemini-2.0-flash",
  name="issue_priortize_agent",
  description="An agent that intelligently prioritizes Jira issues based on various factors.",
  instruction=(
    """Take a Jira issue, fetch its details, and assign a priority level using followinf tools.
    <tools>
    1. get_issue_details(issue_key: str) -> str
    2. set_issue_priority(issue_key: str) -> str
    </tools>

    But before assigning consider multiple factors:
    - the issue's impact on the project
    - ths issue's summary and description
    - the issue's current status
    - the issue's comments and history
    - dependencies with other tasks
    - the availability of resources
    - any deadlines associated with the issue.
    Analyze these aspects and assign a priority level (e.g., High, Medium, Low) to the issue.
    Provide reasoning for your prioritization decision and 
    also consider the given <analyze_issue_priority> tool to suggest priority based on NLP analysis of the issue's content.
    """
  ),
  tools=[get_issue_details, set_issue_priority, analyze_issue_priority]
)
  
