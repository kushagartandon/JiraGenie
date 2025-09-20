from google.adk.agents import LlmAgent, SequentialAgent


from .sub_agents.issue_priortize_agent.issue_priortize_agent import issue_priortize_agent
from .sub_agents.issue_assign_agent.issue_assign_agent import issue_assign_agent
from .sub_agents.issue_sla_agent.issue_sla_agent import issue_sla_agent

issue_triage_agent = SequentialAgent(
  name="issue_triage_agent",
  description="A pipeline that prioritizes, assigns Jira issues and provides a summary report.",
  sub_agents=[issue_priortize_agent, issue_assign_agent, issue_sla_agent],
)