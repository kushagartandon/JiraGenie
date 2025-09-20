import os
from jira import JIRA
from dotenv import load_dotenv

class JiraConnector(staticmethod):

  jiraConnection = None

  @staticmethod
  def connect():
    try:
      load_dotenv()
      return JIRA(
        server=os.getenv("JIRA_SERVER"),
        basic_auth=(os.getenv("JIRA_USERNAME"), os.getenv("JIRA_API_TOKEN"))
      )
    except Exception as e:
      print(f"Error connecting to Jira: {e}")
      return None

  @staticmethod
  def getConnection():
    if JiraConnector.jiraConnection is None:
      JiraConnector.jiraConnection = JiraConnector.connect()
    return JiraConnector.jiraConnection

  
if __name__ == "__main__":
  jira = JiraConnector.getConnection()
  if jira:
    print("Successfully connected to Jira")
  else:
    print("Failed to connect to Jira")