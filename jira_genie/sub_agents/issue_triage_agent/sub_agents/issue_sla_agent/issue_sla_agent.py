from google.adk.agents import LlmAgent

from .....utils.jira_connector import JiraConnector
from datetime import datetime, timedelta

jira = JiraConnector.getConnection()

def calculate_sla(issue_key: str) -> dict:
  """
  Calculate SLA details for a given Jira issue based on its priority and type.
  Args:
    issue_key (str): The key of the Jira issue (e.g., "PROJ-123").
  Returns:
    dict: A dictionary containing SLA details such as due date, escalation thresholds, and current status
  """
  try:
    # SLA matrix (hours)
    sla_matrix = {
      'critical': {'bug': 4, 'security': 2, 'feature': 8, 'task': 6},
      'high': {'bug': 8, 'security': 4, 'feature': 24, 'task': 16},
      'medium': {'bug': 24, 'security': 8, 'feature': 72, 'task': 48},
      'low': {'bug': 72, 'security': 24, 'feature': 168, 'task': 120}
    }

    issue = jira.issue(issue_key)
    priority = issue.fields.priority.name.lower() if issue.fields.priority else 'medium'
    issue_type = issue.fields.issuetype.name.lower() if issue.fields.issuetype else 'task'

    sla_hours = sla_matrix.get(priority, {}).get(issue_type.lower(), 24)
    due_date = datetime.now() + timedelta(hours=sla_hours)

    # Calculate escalation thresholds
    escalation_warning = due_date - timedelta(hours=sla_hours * 0.3)  # 30% before due
    escalation_critical = due_date - timedelta(hours=sla_hours * 0.1)  # 10% before due

    return {
      'sla_hours': sla_hours,
      'due_date': due_date.isoformat(),
      'escalation_warning': escalation_warning.isoformat(),
      'escalation_critical': escalation_critical.isoformat(),
      'status': 'on track' if datetime.now() < escalation_warning else 'at risk' if datetime.now() < escalation_critical else 'breached'
    }

  except Exception as e:
    return {
      'sla_hours': 24,
      'error': str(e),
      'status': 'error'
    }
  
def set_sla_to_issue(issue_key: str, sla_details: dict) -> dict:
  """
  Set SLA details to a Jira issue as custom fields.
  Args:
    issue_key (str): The key of the Jira issue (e.g., "PROJ-123").
    sla_details (dict): A dictionary containing SLA details such as due date, escalation thresholds, and current status.
  Returns:
    dict: A dictionary confirming the SLA update containing issue key and SLA details.
  """
  issue = jira.issue(issue_key)
  fields = jira.fields()
  print(sla_details)
  print(f"fields: {fields}" )
  # Ensure custom fields for SLA details exist in Jira instance and provide default values if missing
  custom_fields = {
    'duedate': sla_details.get('due_date', ''),  # SLA Due Date
    'customfield_10060': sla_details.get('escalation_warning', ''),  # Escalation Warning
    'customfield_10061': sla_details.get('escalation_critical', ''),  # Escalation Critical
    'customfield_10059': sla_details.get('status', 'unknown')  # SLA Status
  }
  try:
    issue.update(fields=custom_fields)
    return {
      "issue_key": issue.key,
      "sla_details": sla_details,
      "message": f"SLA details updated for issue {issue.key}."
    }
  except Exception as e:
    return {
      "issue_key": issue.key,
      "message": "Unable to set fields in JIRA Issue",
      "error": f"Fields are missing: {e}"
    }

issue_sla_agent = LlmAgent(
  name="issue_sla_agent",
  model="gemini-2.0-flash",
  description="An agent that manages and enforces SLA policies for Jira issues based on their priority and type.",
  instruction=
  """
    You are an expert SLA management agent for Jira issues. Your expertise includes:
    1. SLA CALCULATION: Calculate SLA based on issue priority and type using a predefined SLA matrix.
    2. ESCALATION MANAGEMENT: Identify escalation thresholds and current status of the issue.
    3. AUTOMATED UPDATES: Update Jira issues with calculated SLA details in custom fields.

    Key capabilities:
    - Analyze issue details to determine appropriate SLA
    - Calculate due dates and escalation thresholds
    - Update Jira issues with SLA information
    - Provide clear status updates on SLA adherence

    Always ensure SLA details are accurate and provide reasoning behind SLA calculations.
  """,
  tools=[calculate_sla, set_sla_to_issue],
)