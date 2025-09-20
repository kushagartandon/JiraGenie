from datetime import datetime
from typing import Optional
from jira import User, Project, Issue

from google.adk.agents import LlmAgent
from google.adk.agents.callback_context import CallbackContext
from google.genai import types

from .utils.jira_connector import JiraConnector

from .sub_agents.issue_triage_agent.issue_triage_agent import issue_triage_agent
from .sub_agents.sprint_planning_agent.sprint_planning_agent import sprint_planning_agent
from .sub_agents.project_health_agent.project_health_agent import project_health_agent

jira = JiraConnector.getConnection()

def fetch_projects() -> list[dict]:
  """
  Fetch all projects from Jira.
  Args: None
  Returns:
    list[dict]: A list of dictionaries containing project details.
  """
  projects = jira.projects()
  return [
    {
      "project_key": project.key,
      "project_name": project.name,
      "project_id": project.id,
    } for project in projects
  ]

def fetch_issues(
  jql_query: str,
) -> list[dict]:
  """
  Fetch issues from Jira based on JQL query such as project key or assignee.
  Args:
    jql_query (str): The JQL query string to filter issues.
    max_results (int): Maximum number of issues to fetch.
    project_key (Optional[str]): The project key to filter issues (if any).
  Returns: 
    list[dict]: A list of dictionaries containing issue details.
  """
  print(jql_query)
  issues = []
  next_token = None

  while True:
    resp = jira.enhanced_search_issues(
      jql_str=jql_query,
      nextPageToken=next_token,
      maxResults=100,
      json_result=True  # returns raw JSON with 'issues' and 'nextPageToken'
    )
    page_issues = resp.get("issues", [])
    issues.extend(page_issues)
    next_token = resp.get("nextPageToken")
    if not next_token:
      break

  return [
    {
      "issue_key": issue['key'],
      "issue_summary": issue['fields']['summary'],
      "issue_description": issue['fields']['description'] or "",
      # "issue_priority": issue.fields.priority.name if issue.fields.priority else "None",
      # "issue_status": issue.fields.status.name if issue.fields.status else "Unknown", 
      # "issue_assignee": issue.fields.assignee.displayName if issue.fields.assignee else "Unassigned",
      # "issue_created": issue.fields.created,
      # "issue_reporter": issue.fields.reporter.displayName if issue.fields.reporter else "Unknown",
    } for issue in issues
  ]

def create_issue(
  project_key: str,
  summary: str,
  description: str,
  issue_type: str = "Task",
) -> dict:
  """
  Create a new issue in Jira under the specified project.
  Args:
    project_key (str): The key of the project where the issue will be created.
    summary (str): The summary/title of the issue.
    description (str): The detailed description of the issue.
    issue_type (str): The type of the issue (e.g., Task, Bug, Story). Default is "Task".
  Returns:
    dict: A dictionary containing details of the created issue.
  """
  new_issue = jira.create_issue(
    project=project_key,
    summary=summary,
    description=description,
    issuetype={'name': issue_type}
  )
  return {
    "issue_key": new_issue.key,
    "issue_summary": new_issue.fields.summary,
    "issue_description": new_issue.fields.description or "",
    "issue_priority": new_issue.fields.priority.name if new_issue.fields.priority else "None",
    "issue_status": new_issue.fields.status,
    "message": f"Issue {new_issue.key} created successfully."
  }

def update_issue(
  issue_key: str,
  fields: dict,
) -> str:
  """
  update an existing issue in Jira with the provided fields.
  Fields is a dictionary where keys are field names and values are the new values.
  Which can include summary, description, priority, assignee, status, etc.
  """
  issue = jira.issue(issue_key)
  issue.update(fields=fields)
  return f"Issue {issue.key} updated successfully."

def fetch_boards() -> list[dict]:
  """
  Fetch all boards from Jira w.r.t the connected user or project.
  Args: 
    None
  Returns:
    list[dict]: A list of dictionaries containing board details.
  """
  boards = jira.boards()
  print(boards)
  return [
    {
      "board_id": board.id,
      "board_name": board.name,
      "board_type": board.type,
    } for board in boards
  ]

def fetch_sprints(
  board_id: int,
) -> list[dict]:
  """
  Fetch all sprints from a specific board in Jira.
  Args:
    board_id (int): The ID of the board to fetch sprints from.
  Returns:
    list[dict]: A list of dictionaries containing sprint details.
  """
  sprints = jira.sprints(board_id)
  return [
    {
      "sprint_id": sprint.id,
      "sprint_name": sprint.name,
      "sprint_state": sprint.state, 
      "sprint_startDate": sprint.startDate,
      "sprint_endDate": sprint.endDate,
    } for sprint in sprints
  ]

def update_sprint(
  sprint_id: int,
  fields: dict,
) -> str:
  """
  Update an existing sprint in Jira with the provided fields.
  Fields is a dictionary where keys are field names and values are the new values.
  Fields can include name, startDate, endDate, goal, state, etc.
  Args:
    sprint_id (int): The ID of the sprint to update.
    fields (dict): A dictionary of fields to update on the sprint.
  Returns:
    str: A confirmation message indicating the sprint was updated.
  """
  sprint = jira.sprint(sprint_id)
  for field, value in fields.items():
    setattr(sprint, field, value)
  sprint.update()
  return f"Sprint {sprint.id} updated successfully."


def fetch_team_members(
  project_key: str,
) -> list[dict]:
  """
  Fetch all team members from Jira on the basis of projectId.
  Note: Adjust the query parameter as needed to fit your Jira instance's requirements.
  Args:
    project_key (str): The key of the project to fetch team members from.
  Returns:
    str: A formatted string listing all team members.
  """
  print(project_key)
  users = jira.search_assignable_users_for_issues(project=project_key, maxResults=10, query=".")
  for user in users:
    print(user.accountId)
  return [
    {
      "displayName": user.displayName,
      "emailAddress": user.emailAddress if hasattr(user, 'emailAddress') else "Not Available",
      "active": user.active,
      "accountId": user.accountId
    } for user in users
  ]


def fetch_fields() -> list[str]:
  """
  Fetch all the fields from the jira project/board
  """
  fields = jira.fields()
  print(fields)
  for field in fields:
    print(f"{field}")
  return fields


root_agent = LlmAgent(
  model="gemini-2.0-flash",
  name="jira_genie", 
  description="An enterprise-level AI assistant to manage Jira Board workflows efficiently.",
  instruction="""
  You are Jira-Genie, the root agent which manages jira issues, sprints of an project.
  Your one of the role is to provide response to the user queries related to jira issues, sprints and project health.
  Your role is to act as the intelligent coordinator and provide clear, actionable insights. 
  You are Jira-Genie, the root agent that orchestrates multiple sub-agents to manage Jira workflows. 
  
  Use sub-agents to handle domain-specific requests, but always maintain a consistent Jira-focused experience.

  <sub_agents>
  1. IssueTriageAgent
     - Purpose: Prioritize incoming issues, assign them intelligently, and enforce SLA rules.
     - Capabilities:
       - Analyze issue text using NLP to determine severity and category.
       - Assign tasks automatically based on expertise/team workload (team expertise data is stored in DB).
       - Apply SLA calculation and trigger escalation for overdue issues (Major/Medium/Low).
       - Demo power: For a critical production bug → auto-assign to a senior dev within 10 seconds.

  2. SprintPlanningAgent
     - Purpose: Optimize sprint planning and balance team workload.
     - Capabilities:
       - Calculate story point distribution across team members (taking into account existing workload).
       - Predict sprint completion probability using historical velocity trends.
       - Build risk-balanced sprint compositions (considering high-risk vs low-risk tasks).
       - Demo power: Given a backlog of 20 items → generate a perfect sprint plan in 30 seconds.

  3. ProjectHealthAgent
     - Purpose: Monitor and forecast team/project health.
     - Requirement: Ask or get a project key from the user.
     - Capabilities:
       - Run predictive risk analysis using historical Jira data.
       - Detect velocity trends and forecast performance for upcoming sprints.
       - Identify team burnout risk (e.g., excessive ad-hoc/unplanned tasks).
       - Demo power: If declining velocity is detected → suggest process improvements and mitigation actions.
  </sub_agents>

  Workflow Rules:
  - Always clarify which area the request belongs to (Issue Triage, Sprint Planning, Project Health or You can manage it on your own).
  - Route the task to the relevant sub-agent if not in your scope.
  - Maintain Jira context: all outputs should be phrased in terms of tasks, issues, sprints, and project health.
  - Use provided jira tools to answer user queries.

  For <toool> fetch_issues: Try to form jql query as per the user query.
  Example: 
  "Fetch all issues assigned to John in project PROJ" -> jql = "assignee=John AND project=PROJ"
  "Fetch all high priority bugs in project PROJ" -> jql = "priority=High AND issuetype=Bug AND project=PROJ"
  "Fetch all unassigned issues in project PROJ" -> jql = "assignee is EMPTY AND project=PROJ"

  Always: Be precise, Jira-specific, and enterprise-ready.
  """,
  sub_agents=[issue_triage_agent, sprint_planning_agent, project_health_agent],
  tools=[
    fetch_issues, 
    create_issue, 
    update_issue,
    fetch_sprints, 
    update_sprint, 
    fetch_projects, 
    fetch_boards, 
    fetch_team_members,
    fetch_fields
  ],
  # tools=[JiraConnector()],  # Pass the JiraConnector class as a tool for the agent
)


