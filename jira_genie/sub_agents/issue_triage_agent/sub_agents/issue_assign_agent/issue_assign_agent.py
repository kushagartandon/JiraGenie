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

def get_team_member_workload_details(project_key: str) -> list[dict]:
  """
  Fetch workload details of team members in a given project.
  Args:
    project_key (str): The key of the Jira project (e.g., "PROJ").
  Returns:
    list[dict]: A list of dictionaries, each containing details of a team member such as username, display name, email, and active status.
  """
  users = jira.search_assignable_users_for_issues(project=project_key, maxResults=10, query=".")
  return [
    {
      "accountId": user.accountId,
      "displayName": user.displayName,
      "emailAddress": user.emailAddress if hasattr(user, 'emailAddress') else "Not Available",
      "active": user.active,
      "issues_assigned": len(jira.search_issues(f'assignee={user.accountId} AND project={project_key}', maxResults=100))
    } for user in users
  ]

def get_team_member_skills_details() -> list[dict]:
  return [
    {
      "username": "ankur yadav",
      "skills": ["Java", "SpringBoot", "C++", "Python", "SQL"],
      "role": "Junior Developer"
    },
    {
      "username": "kushagar tandon",
      "skills": ["Java", "SpringBoot", "Spring", "Python", "SQL", "Mongo"],
      "role": "SDE 1"
    },
    {
      "username": "nikhil joshi",
      "skills": ["Kafka", "SpringBoot", "Java", "Python", "elasticSearch", "AWS", "GCP", "GOLanf"],
      "role": "Team lead"
    },
    {
      "username": "ashutosh sahu",
      "skills": ["Kafka", "SpringBoot", "Java", "Python", "elasticSearch", "AWS"],
      "role": "SDE 3"
    },
    {
      "username": "ankit bansal",
      "skills": ["Kafka", "SpringBoot", "Java", "Python"],
      "role": "SDE 2"
    }
  ]

def assign_issue_to_user(issue_key: str, accountId: str, username: str) -> dict:
  """
  Assign a Jira issue to a specific user.
  Args:
    issue_key (str): The key of the Jira issue (e.g., "PROJ-123").
    accountId (str): 
  Returns:
    dict: A dictionary confirming the assignment containing issue key and assignee username.
  """
  print(accountId)
  issue = jira.issue(issue_key)
  issue.update(fields={"assignee": {"accountId": accountId}})
  return {
    "issue_key": issue.key,
    "assignee": username,
    "accountId": accountId,
    "message": f"Issue {issue.key} assigned to {username}."
  }

issue_assign_agent = LlmAgent(
  model="gemini-2.0-flash",
  name="issue_assign_agent",
  description="""
  An agent that intelligently assigns Jira issues to the most suitable team member for efficient workload distribution and timely resolution.
  """,
  instruction=(
    """
    This agent is responsible for assigning a Jira issue to the most appropriate team member for efficient resolution.
    Consider the following factors when making the assignment:
    - their experience
    - specific skills
    - topics they are familiar with
    - current workload
    - the complexity of the task
    - the urgency of the task. 


    Evaluate these aspects for each member and assign the issue to the person who best matches the requirements, 
    ensuring a balanced workload and timely resolution.
     
    Provide reasoning for your assignment decision and fetch details for users and issues from the given tools.
    Use the following tools in order to get the necessary information:
    <tools>
    1. get_issue_details(issue_key: str) -> dict
    2. get_team_member_workload_details(project_key: str) -> list[dict]
    3. get_team_member_skills_details() -> list[dict] 
    4. assign_issue_to_user(issue_key: str, accountId: str, username: str) -> dict
    """

  ),
  tools=[get_issue_details, get_team_member_workload_details, get_team_member_skills_details, assign_issue_to_user]
)
