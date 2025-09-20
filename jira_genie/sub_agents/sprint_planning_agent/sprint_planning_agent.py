from google.adk.agents import LlmAgent
from google.adk.tools import FunctionTool
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
import json
from typing import Optional, List, Dict, TypedDict

from ...utils.jira_connector import JiraConnector
 
 
class SprintPlanningAgent:
    def fetch_jira_backlog_items(self, project_key: str) -> list:
        """
        Fetch all backlog tasks from JIRA for a given project and convert them to the required format for sprint planning.
        project_key: JIRA project key (e.g., SCRUM)
        To calculate the number of Backlogs issues for a given project:
        'project = {project_key} AND sprint IS EMPTY AND status IN ("To Do", "Open" , "IN REVIEW" ,"ON HOLD")'

        sprint should be Empty and status should be one of these ("To Do", "Open" , "IN REVIEW" ,"ON HOLD")'
        Returns: List of dicts with keys: story_points, type, priority_score, complexity, dependencies
        """
        if not self.jira:
            return []
       
        try:
            jql_query = f'project = {project_key} AND sprint IS EMPTY AND status IN ("To Do", "Open" , "IN REVIEW" ,"ON HOLD")'
            # issues = self.jira.search_issues(jql_query, maxResults=1000)
            issues = []
            next_token = None

            while True:
                resp = self.jira.enhanced_search_issues(
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

            backlog_items = []
            print(f"Count of Backlog issues: {len(issues)}")

            for issue in issues:
              fields = issue.get("fields") or {}   # ensure dict, even if None

              # Story points
              story_points = fields.get("customfield_10016") or 0

              # Issue type
              issue_type = (fields.get("issuetype") or {}).get("name", "task").lower()

              # Priority (null-safe)
              priority = fields.get("priority") or {}
              priority_score = int(priority.get("id", 2))

              # Complexity (custom field, safe fallback)
              complexity = str(fields.get("complexity") or "medium").lower()

              # Dependencies
              dependencies = len(fields.get("issuelinks") or [])

              backlog_items.append({
                  'key': issue["key"],
                  "story_points": int(story_points),
                  "type": issue_type,
                  "priority_score": priority_score,
                  "complexity": complexity,
                  "dependencies": dependencies,
              })

            # for issue in issues:
            #   fields = issue["fields"]
            #   # Extract fields, adjust field names as per your JIRA config
            #   story_points = fields.get('customfield_10016', 0)  # Change customfield_10016 to your Story Points field
            #   issue_type = fields.get('issuetype', {}).get('name', 'task').lower()
            #   priority_score = fields.get('priority', {}).get('id', 2)  # Use priority id or map as needed
            #   complexity = fields.get('complexity', 'medium')  # Adjust if you have a custom field
            #   dependencies = len(fields.get('issuelinks', []))
            #   backlog_items.append({
            #       'story_points': int(story_points) if story_points else 0,
            #       'type': issue_type,
            #       'priority_score': int(priority_score),
            #       'complexity': str(complexity).lower(),
            #       'dependencies': dependencies
            #   })
            return backlog_items
        except Exception as e:
          print(f"ERROR: Skipping issue {issue.get('key')} due to error: {e}")
          return []
    """
    AI-powered Sprint Planning Agent
    - Velocity prediction using machine learning
    - Optimal story point distribution
    - Risk-balanced sprint composition
    """
 
    def __init__(self):
        self.jira = JiraConnector.getConnection()
 
        # Historical sprint data (in real implementation, load from database)
        # Fetch historical sprint data from JIRA if connector is provided, else use empty/default
        if self.jira:
            try:
                self.historical_data = self.jira.get_historical_sprint_data()
            except Exception:
                self.historical_data = {'sprints': []}
        else:
            self.historical_data = {'sprints': []}
 
        # Team velocity patterns
        self.team_metrics = {
            'average_velocity': 43.6,
            'velocity_std': 4.2,
            'completion_rate': 0.89,
            'bug_impact_factor': 0.95,
            'feature_complexity_factor': 1.1
        }
 
    def predict_velocity(self, team_size: int, planned_story_types: dict, external_factors: Optional[dict]) -> dict:
        """Predict sprint velocity using ML algorithms"""
        try:
            # Prepare historical data for ML
            df = pd.DataFrame(self.historical_data['sprints'])
 
            # Feature engineering
            df['bug_ratio'] = df['bugs'] / (df['bugs'] + df['features'] + df['tasks'])
            df['feature_ratio'] = df['features'] / (df['bugs'] + df['features'] + df['tasks'])
            df['task_ratio'] = df['tasks'] / (df['bugs'] + df['features'] + df['tasks'])
            df['points_per_person'] = df['planned_points'] / df['team_size']
            df['completion_ratio'] = df['completed_points'] / df['planned_points']
 
            # Features for prediction
            features = ['team_size', 'bug_ratio', 'feature_ratio', 'task_ratio', 'points_per_person']
            X = df[features]
            y = df['completion_ratio']
 
            # Train models
            linear_model = LinearRegression()
            rf_model = RandomForestRegressor(n_estimators=10, random_state=42)
 
            linear_model.fit(X, y)
            rf_model.fit(X, y)
 
            # Prepare prediction data
            total_items = sum(planned_story_types.values())
            if total_items == 0:
                total_items = 1
 
            pred_data = {
                'team_size': team_size,
                'bug_ratio': planned_story_types.get('bugs', 0) / total_items,
                'feature_ratio': planned_story_types.get('features', 0) / total_items,
                'task_ratio': planned_story_types.get('tasks', 0) / total_items,
                'points_per_person': (planned_story_types.get('total_points', 50)) / team_size
            }
 
            pred_df = pd.DataFrame([pred_data])
 
            # Make predictions
            linear_pred = linear_model.predict(pred_df)[0]
            rf_pred = rf_model.predict(pred_df)[0]
 
            # Ensemble prediction (weighted average)
            ensemble_pred = 0.6 * rf_pred + 0.4 * linear_pred
 
            # Apply external factors
            external_factor = 1.0
            if external_factors:
                if external_factors.get('holidays', 0) > 0:
                    external_factor *= (1 - external_factors['holidays'] * 0.1)  # 10% reduction per holiday
                if external_factors.get('new_team_members', 0) > 0:
                    external_factor *= 0.9  # 10% reduction for new members
                if external_factors.get('technical_debt', False):
                    external_factor *= 0.85  # 15% reduction for technical debt
 
            final_completion_ratio = ensemble_pred * external_factor
            predicted_velocity = planned_story_types.get('total_points', 50) * final_completion_ratio
 
            # Calculate confidence interval
            std_dev = np.std([linear_pred, rf_pred])
            confidence_lower = max(0, predicted_velocity - std_dev * predicted_velocity)
            confidence_upper = predicted_velocity + std_dev * predicted_velocity
 
            return {
                'predicted_velocity': round(predicted_velocity, 1),
                'completion_probability': round(final_completion_ratio * 100, 1),
                'confidence_range': {
                    'lower': round(confidence_lower, 1),
                    'upper': round(confidence_upper, 1)
                },
                'model_predictions': {
                    'linear_regression': round(linear_pred * planned_story_types.get('total_points', 50), 1),
                    'random_forest': round(rf_pred * planned_story_types.get('total_points', 50), 1),
                    'ensemble': round(ensemble_pred * planned_story_types.get('total_points', 50), 1)
                },
                'external_impact': round((1 - external_factor) * 100, 1),
                'recommendation': self._get_velocity_recommendation(final_completion_ratio)
            }
 
        except Exception as e:
            return {
                'predicted_velocity': self.team_metrics['average_velocity'],
                'error': str(e),
                'fallback': True
            }
 
    def _get_velocity_recommendation(self, completion_ratio: float) -> str:
        """Get recommendation based on predicted completion ratio"""
        if completion_ratio >= 0.95:
            return "Excellent capacity utilization. Consider adding stretch goals."
        elif completion_ratio >= 0.85:
            return "Good sprint capacity. Plan looks achievable."
        elif completion_ratio >= 0.70:
            return "Moderate risk. Consider reducing scope or complexity."
        else:
            return "High risk of incomplete sprint. Recommend significant scope reduction."
 
 
    def optimize_story_distribution(

        self,

        backlog_items: Optional[List[Dict[str, object]]] = None,

        team_capacity: int = 20,

        project_key: Optional[str] = None,

        sprint_duration: int = 2

    ) -> dict:

        """

        Optimize story distribution into a sprint plan based on backlog items and team capacity.
    
        Args:

            backlog_items: List of backlog item dicts (from Jira). If None, will fetch using project_key.

            team_capacity: Maximum story points the team can deliver in the sprint.

            project_key: Jira project key (used if backlog_items not provided).

            sprint_duration: Sprint length in weeks (optional, default=2).
    
        Returns:

            dict containing:

                - selected_items: list of Jira issue keys chosen for the sprint

                - total_story_points: int, total points of selected items

                - capacity_utilization: % of capacity used

                - risk_level: low/medium/high based on dependencies + utilization

                - success_probability: estimated %

                - recommendations: list of suggestions

                - mitigation_plan: string

        """

        try:

            # --- Fetch backlog from Jira if not provided ---

            if not backlog_items:

                if not project_key:

                    return {"error": "Either backlog_items or project_key must be provided"}

                backlog_items = self.fetch_jira_backlog_items(project_key)
    
            if not backlog_items:

                return {"error": "No backlog items available to optimize"}
    
            # --- Normalize items ---

            normalized = []



            for it in backlog_items:

                sp = it.get("story_points")
                if sp in (None, 0, ""):  # default to 4 if not provided or 0
                    sp = 4

                normalized.append({

                    "key": it.get("key"),

                    "story_points": int(sp),

                    "type": str(it.get("type", "other") or "other").lower(),

                    "priority_score": int(it.get("priority_score", 0) or 0),

                    "complexity": str(it.get("complexity", "medium") or "medium").lower(),

                    "dependencies": int(it.get("dependencies", 0) or 0),

                    "_orig": it

                })
    
            # --- Sort backlog: high priority, low deps first ---

            sorted_items = sorted(

                normalized,

                key=lambda x: (-x["priority_score"], x["dependencies"], -x["story_points"])

            )
    
            # --- Select items until capacity is reached ---

            selected_items, total_points = [], 0

            for item in sorted_items:

                if total_points + item["story_points"] <= team_capacity:

                    selected_items.append(item)

                    total_points += item["story_points"]
    
            utilization = int((total_points / team_capacity) * 100) if team_capacity > 0 else 0
    
            # --- Risk assessment ---

            risk_level = "low"

            if utilization > 90 or any(it["dependencies"] > 3 for it in selected_items):

                risk_level = "medium"

            if utilization > 100:

                risk_level = "high"
    
            # --- Probability estimate ---

            success_probability = max(30, 100 - abs(team_capacity - total_points) * 5)
    
            return {

                "selected_items": [s["key"] for s in selected_items],

                "total_story_points": total_points,

                "capacity_utilization": utilization,

                "risk_level": risk_level,

                "success_probability": success_probability,

                "recommendations": [

                    "Prioritize high-value, low-dependency stories.",

                    "Keep sprint load close to velocity range.",

                    "Watch out for items with many dependencies."

                ],

                "mitigation_plan": "Split large stories; reassign high-dependency items."

            }
    
        except Exception as e:

            return {"error": str(e)}

 
 
    def _calculate_item_risk(self, item: dict) -> float:
        """Calculate risk score for individual item"""
        base_risk = 0.1
 
        # Complexity factor
        complexity_factors = {'simple': 0.1, 'medium': 0.3, 'complex': 0.6}
        complexity_risk = complexity_factors.get(item.get('complexity', 'medium').lower(), 0.3)
 
        # Dependency factor
        dependency_risk = min(0.4, item.get('dependencies', 0) * 0.1)
 
        # Story points factor (higher points = higher risk)
        points_risk = min(0.3, item.get('story_points', 0) * 0.05)
 
        return base_risk + complexity_risk + dependency_risk + points_risk
 
    def _analyze_sprint_balance(self, story_types: dict, complexity_dist: dict, total_points: int, capacity: int) -> dict:
        """Analyze sprint balance and composition"""
        total_items = sum(story_types.values())
 
        # Ideal ratios
        ideal_ratios = {'bugs': 0.2, 'features': 0.6, 'tasks': 0.2}
 
        balance_score = 100
        issues = []
 
        if total_items > 0:
            for story_type, ideal_ratio in ideal_ratios.items():
                actual_ratio = story_types.get(story_type, 0) / total_items
                deviation = abs(actual_ratio - ideal_ratio)
 
                if deviation > 0.2:  # 20% deviation threshold
                    balance_score -= deviation * 50
                    issues.append(f"Imbalanced {story_type} ratio: {actual_ratio:.1%} vs ideal {ideal_ratio:.1%}")
 
        # Complexity balance check
        total_complexity = sum(complexity_dist.values())
        if total_complexity > 0:
            complex_ratio = complexity_dist.get('complex', 0) / total_complexity
            if complex_ratio > 0.4:  # More than 40% complex items
                balance_score -= 20
                issues.append(f"Too many complex items: {complex_ratio:.1%}")
 
        return {
            'score': max(0, round(balance_score)),
            'issues': issues,
            'recommendations': self._get_balance_recommendations(issues)
        }
 
    def _get_balance_recommendations(self, issues: list) -> list:
        """Generate recommendations based on balance issues"""
        recommendations = []
 
        for issue in issues:
            if 'bugs' in issue.lower():
                recommendations.append("Consider deferring non-critical bugs to next sprint")
            elif 'features' in issue.lower():
                recommendations.append("Balance feature development with technical debt")
            elif 'complex' in issue.lower():
                recommendations.append("Mix complex items with simpler ones for better flow")
 
        return recommendations
 
    def _get_risk_level(self, risk_score: float) -> str:
        """Convert risk score to risk level"""
        if risk_score <= 0.3:
            return "Low"
        elif risk_score <= 0.5:
            return "Medium"
        else:
            return "High"
 
    def _get_distribution_recommendations(self, story_types: dict, complexity_dist: dict, total_points: int, capacity: int) -> list:
        """Generate distribution recommendations"""
        recommendations = []
        utilization = (total_points / capacity) * 100
 
        if utilization < 70:
            recommendations.append("Consider adding more items to fully utilize team capacity")
        elif utilization > 95:
            recommendations.append("Sprint is at high capacity - consider buffer for unexpected work")
 
        total_items = sum(story_types.values())
        if total_items > 0:
            bug_ratio = story_types.get('bugs', 0) / total_items
            if bug_ratio > 0.4:
                recommendations.append("High bug ratio detected - consider technical debt sprint")
 
        return recommendations
 
 
 
    def create_sprint_plan(self,
                        project_key: str,
                        team_size: int = 5,
                        external_factors: Optional[dict] = None,
                        sprint_goal: str = "") -> dict:
        """
        Always fetches backlog from JIRA for the given project key (status and resolution are always the same).
        Returns a sprint_plan dict.
        """
        try:
            backlog_items = self.fetch_jira_backlog_items(project_key)
            if not isinstance(backlog_items, list):
                return {'error': 'Failed to fetch backlog from JIRA', 'status': 'sprint_planning_failed'}
 
            # validate each item is a dict and normalize missing fields
            normalized_items = []
            for idx, item in enumerate(backlog_items):
                if not isinstance(item, dict):
                    return {'error': f'backlog_items[{idx}] is not an object/dict', 'status': 'sprint_planning_failed'}
                normalized_items.append({
                    'story_points': int(item.get('story_points', 0)),
                    'type': str(item.get('type', 'other')).lower(),
                    'priority_score': int(item.get('priority_score', 0)),
                    'complexity': item.get('complexity', None),
                    'dependencies': int(item.get('dependencies', 0))
                })
 
            # --- Step 1: Calculate team capacity ---
            base_capacity_per_person = 8  # story points per person per sprint (tweakable)
            team_capacity = team_size * base_capacity_per_person
 
            # --- Step 2: Analyze story types and total points ---
            story_type_counts = {'bugs': 0, 'features': 0, 'tasks': 0, 'total_points': 0}
            for item in normalized_items:
                item_type = item['type']
                if item_type in story_type_counts:
                    story_type_counts[item_type] += 1
                # Map synonyms
                elif item_type in ('bug',):
                    story_type_counts['bugs'] += 1
                elif item_type in ('story', 'feature', 'enhancement', 'new feature'):
                    story_type_counts['features'] += 1
                elif item_type in ('task', 'sub-task', 'subtask'):
                    story_type_counts['tasks'] += 1
                story_type_counts['total_points'] += item['story_points']
 
            # --- Step 3: Predict velocity (call your model/helper) ---
            # Ensure predict_velocity exists on self; stub below if not
            velocity_prediction = self.predict_velocity(team_size, story_type_counts, external_factors)
 
            # --- Step 4: Optimize distribution ---
            distribution_result = self.optimize_story_distribution(normalized_items, team_capacity)
 
            # --- Step 5: Compose sprint plan ---
            sprint_plan = {
                'sprint_metadata': {
                    'created_at': datetime.now().isoformat(),
                    'team_size': team_size,
                    'sprint_goal': sprint_goal,
                    'duration_weeks': 2
                },
                'backlog_ticket_count': len(backlog_items),
                'capacity_analysis': {
                    'team_capacity': team_capacity,
                    'predicted_velocity': velocity_prediction.get('predicted_velocity'),
                    'utilization_target': 85  # Target 85% utilization
                },
                'velocity_prediction': velocity_prediction,
                'story_distribution': distribution_result,
                'success_probability': velocity_prediction.get('completion_probability', 75),
                'recommendations': self._generate_sprint_recommendations(velocity_prediction, distribution_result),
                'risk_mitigation': self._generate_risk_mitigation_plan(distribution_result.get('risk_assessment', {}))
            }
 
            return sprint_plan
 
        except Exception as e:
            return {'error': str(e), 'status': 'sprint_planning_failed'}
       
 
    def _generate_sprint_recommendations(self, velocity_pred: dict, distribution: dict) -> list:
        """Generate comprehensive sprint recommendations"""
        recommendations = []
 
        completion_prob = velocity_pred.get('completion_probability', 0)
        if completion_prob < 70:
            recommendations.append("⚠️ Low completion probability - consider reducing scope")
        elif completion_prob > 95:
            recommendations.append("✅ High completion probability - consider stretch goals")
 
        risk_level = distribution.get('risk_assessment', {}).get('risk_level', 'Medium')
        if risk_level == 'High':
            recommendations.append("🔴 High risk sprint - add buffer time and simplify complex items")
        elif risk_level == 'Low':
            recommendations.append("🟢 Well-balanced sprint composition")
 
        return recommendations
 
    def _generate_risk_mitigation_plan(self, risk_assessment: dict) -> list:
        """Generate risk mitigation strategies"""
        mitigation_plan = []
 
        risk_level = risk_assessment.get('risk_level', 'Medium')
        overall_risk = risk_assessment.get('overall_risk', 0)
 
        if risk_level == 'High' or overall_risk > 0.5:
            mitigation_plan.extend([
                "Schedule daily check-ins for complex items",
                "Identify dependencies early and create blockers",
                "Have backup stories ready in case of scope reduction",
                "Pair programming for high-risk items"
            ])
 
        return mitigation_plan
 
    def get_agent(self):
        """Return the ADK agent configuration"""
        return LlmAgent(  
            name="sprint_planning_agent",
            model="gemini-2.0-flash",
            description="AI agent for intelligent sprint planning with velocity prediction and optimal story distribution",
            instruction="""
            You are an expert Sprint Planning Agent with strong expertise in Agile methodologies, project management, and advanced machine learning for predictive analysis. 
            Your primary responsibility is to assist teams in planning effective and realistic sprints by leveraging historical data, workload balancing strategies, and risk awareness.
            Conisder Story points for each backlog issue to be 4.

            CORE CAPABILITIES:
            1. VELOCITY PREDICTION:
            - Analyze historical sprint data to forecast the team’s velocity.
            - Adjust predictions based on current sprint variables (e.g., team size, availability, complexity).
            - Provide confidence intervals (e.g., 80% likely to complete 25–30 story points).
            - Highlight factors that may impact the prediction (holidays, team changes, technical debt, blockers).

            2. OPTIMAL STORY DISTRIBUTION:
            - Recommend how to best allocate stories across different work categories (features, bugs, technical debt, spikes).
            - Ensure workload balance across team members while considering specialization and skillsets.
            - Optimize based on business priority, risk assessment, risk balance and available capacity.
            - Identify risks in sprint planning (overcommitment, underutilization, critical dependencies, or unclear requirements).
            - Provide early warnings about likely bottlenecks.
            - Recommend mitigation strategies to reduce risk and improve delivery confidence.
            - Returns list of backlogs issues to include according to the sprint plan based on SELECTION STRATEGY
            - SELECTION STRATEGY to include issues in the Sprint:
                1. **Category Distribution (Work Type Mix)**:
                - Maintain approximate work allocation of:
                    - 60% Bugs
                    - 15% Features
                    - 25% Tasks
                - These ratios guide workload balancing but may flex when spillovers or critical items exist.
                - For Example: If we have calculated capacity to be 60, which can include let's say 20 issues, so 60% of 12 i.e 12 bugs, likewise 5 tasks and 3 features.

                2. **Ticket Age Consideration**:
                - Any ticket older than 1 month takes precedence over newer tickets,
                    regardless of its assigned priority level.
                - Ensures stale or long-pending work is addressed promptly.

                3. **Priority-Based Ordering**:
                - Within each category, select tickets in order of priority:
                    - High → Medium → Low
                - Priorities are only overridden if the ticket age rule applies.

                4. **Velocity Alignment**:
                - The number of tickets selected must align with the team’s historical velocity (average story points completed per sprint).
                - Prevents overcommitment and ensures sprint commitments are realistic.

                5. **Spillover Handling**:
                - Tickets that spill over from previous sprints must be counted first towards the velocity limit before adding new tickets.
                - Spillovers retain their original priority and category but are considered mandatory inclusions.

            3. CREATE SPRINT PLAIN:
            - Identify risks in sprint planning (overcommitment, underutilization, critical dependencies, or unclear requirements).
            - Provide early warnings about likely bottlenecks.
            - Recommend mitigation strategies to reduce risk and improve delivery confidence.

            KEY TOOLS (Agent Functions):
            - **predict_velocity**:
            Input: Historical sprint data, team availability, and current sprint context.
            Output: Predicted velocity with confidence range, plus reasoning for the prediction.
            Usage: Helps set realistic sprint goals and avoids over/under commitment.

            - **optimize_story_distribution**:
            Input: List of candidate stories with estimates, priorities, and categories.
            Output: Optimized distribution of stories across work types and/or team members.
            Usage: Ensures fair and efficient workload balancing while aligning with business objectives.

            - **create_sprint_plan**:
            Input: Predicted velocity, optimized story distribution, and business priorities.
            Output: A complete sprint plan including selected stories, workload distribution, identified risks, and confidence levels.
            Usage: Provides a holistic, data-driven sprint plan ready for review and execution.

            INTERACTION STYLE:
            - Always explain reasoning and assumptions behind recommendations.
            - Present results in clear, structured, and actionable formats (tables, bullet points, or summaries).
            - When uncertain, state limitations and suggest ways to gather missing data.
            - Balance quantitative predictions (ML-based) with qualitative insights (Agile best practices).""",
            tools=[
                FunctionTool(self.predict_velocity),
                FunctionTool(self.optimize_story_distribution),
                FunctionTool(self.create_sprint_plan)
            ]
        )
    
sprint_planning_agent = SprintPlanningAgent().get_agent()