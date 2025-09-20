import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import IsolationForest
import json
from typing import Optional
import warnings

from ...utils.jira_connector import JiraConnector

warnings.filterwarnings('ignore')

class ProjectHealthAgent:
    def get_sprint_history(self) -> list:
        """
        Return the full sprint history for the current project, including all sprints (active, closed, future) with metadata.
        Returns a list of dicts: [{id, name, state, startDate, endDate, completeDate}]
        """
        if not self.project_key:
            return []
        try:
            boards = self.jira.boards(projectKeyOrID=self.project_key)
            if not boards:
                return []
            board_id = boards[0].id
            # Get all sprints (active, closed, future)
            sprints = []
            for state in ['active', 'closed', 'future']:
                sprints += self.jira.sprints(board_id, state=state)
            # Remove duplicates by id
            seen = set()
            sprint_history = []
            for s in sprints:
                if s.id in seen:
                    continue
                seen.add(s.id)
                sprint_history.append({
                    'id': s.id,
                    'name': getattr(s, 'name', str(s.id)),
                    'state': getattr(s, 'state', 'unknown'),
                    'startDate': getattr(s, 'startDate', None),
                    'endDate': getattr(s, 'endDate', None),
                    'completeDate': getattr(s, 'completeDate', None)
                })
            # Sort by startDate if available, else by id
            def sort_key(s):
                return s['startDate'] or str(s['id'])
            sprint_history = sorted(sprint_history, key=sort_key)
            return sprint_history
        except Exception as e:
            print(f"Error fetching sprint history: {e}")
            return []

    """

    AI-powered Project Health Monitoring Agent

    - Predictive risk analysis and early warnings

    - Velocity trend detection and forecasting

    - Team burnout prevention

    """

 

    def __init__(self, project_key=None):
        self.jira = JiraConnector.getConnection()
      
        self.risk_thresholds = {
            'velocity_decline': 0.15,  # 15% decline triggers warning
            'bug_increase': 2.0,       # 2x bug increase triggers warning
            'overtime_threshold': 3,    # 5+ hours overtime per week
            'story_load_threshold': 5   # 8+ stories per person
        }

        self.project_key = project_key
        self.project_metrics = None
        if self.project_key:
            self.project_metrics = self._fetch_project_metrics_from_jira(self.project_key)

    def set_project_key(self, project_key: str) -> str:
        """Set the Jira project key and reload metrics."""
        self.project_key = project_key
        self.project_metrics = self._fetch_project_metrics_from_jira(self.project_key)
        return f"Project key set to {project_key}. Metrics loaded."
    
    def get_agent(self):
        """Return the ADK agent configuration with function tools for chat."""
        from google.adk.agents import LlmAgent
        from google.adk.tools import FunctionTool
        return LlmAgent(
            name="project_health_agent",
            model="gemini-2.0-flash",
            description="AI agent for comprehensive project health monitoring with predictive analytics and burnout prevention",
            instruction="""
            You are an expert Project Health Monitoring Agent that leverages advanced predictive analytics, agile metrics, and behavioral insights to keep projects on track.
            Your primary responsibility is to continuously monitor project health, forecast potential risks, and recommend preventive actions for sustainable delivery.

            CORE CAPABILITIES:
            1. PREDICTIVE RISK ANALYSIS:
            - Identify early warning signals in delivery timelines, dependencies, or resource allocation.
            - Use historical and real-time data to forecast risks before they materialize.
            - Provide confidence scores and suggested mitigation strategies.

            2. VELOCITY FORECASTING:
            - Analyze historical velocity and sprint performance trends.
            - Predict future delivery capacity with confidence intervals.
            - Flag anomalies such as sudden drops, inconsistent throughput, or unrealistic commitments.

            3. BURNOUT PREVENTION:
            - Detect signs of team fatigue or overcommitment based on workload patterns and velocity trends.
            - Recommend balancing strategies such as reallocation of tasks, improved prioritization, or workload adjustments.
            - Provide early alerts to prevent productivity loss and team disengagement.

            4. EARLY WARNING SYSTEM:
            - Continuously scan project data for deviations from expected performance.
            - Highlight issues before they escalate into critical problems.
            - Present insights as actionable alerts for stakeholders.

            KEY TOOLS (Agent Functions):
            - **set_project_key**  
            Input: Project identifier (e.g., JIRA project key).  
            Purpose: Establish context for subsequent analysis by linking all queries to a specific project.  

            - **analyze_velocity_trends**  
            Input: Historical sprint velocity and performance data.  
            Output: Trend analysis, forecasted velocity range, and anomaly detection.  
            Purpose: Helps teams plan capacity realistically and identify potential delivery risks.  

            - **detect_burnout_risk**  
            Input: Workload metrics, velocity history, team size, and allocation patterns.  
            Output: Burnout likelihood score, indicators, and mitigation strategies.  
            Purpose: Prevents overcommitment and ensures sustainable pace.  

            - **predict_project_risks**  
            Input: Current sprint/project data, dependencies, and external risk factors.  
            Output: Risk categories, probability, impact level, and recommended mitigations.  
            Purpose: Provides early warnings about delivery, quality, or resourcing risks.  

            - **generate_health_report**  
            Input: Aggregated metrics and risk insights.  
            Output: Comprehensive health report including velocity forecasts, burnout indicators, risks, and actionable recommendations.  
            Purpose: Gives stakeholders a structured, data-driven snapshot of overall project health.  

            INTERACTION STYLE:
            - Always provide confidence scores, probability ranges, and clear reasoning.
            - Support insights with trend charts, tables, or bullet-point summaries for clarity.
            - When uncertainty exists, state assumptions explicitly and suggest additional data to improve accuracy.
            - Recommendations must always be **action-oriented**, enabling stakeholders to take preventive steps immediately.""",
            tools=[
                FunctionTool(self.set_project_key),
                FunctionTool(self.analyze_velocity_trends),
                FunctionTool(self.detect_burnout_risk),
                FunctionTool(self.predict_project_risks),
                FunctionTool(self.generate_health_report)
            ]
        )

    def _fetch_project_metrics_from_jira(self, project_key):
        """
        Fetch velocity history and team workload from Jira for the given project key.
        Returns a dict with 'velocity_history', 'team_workload', 'sprint_names', 'completed_sprints', and 'current_sprint'.
        """
        velocity_history = []
        team_workload = {}
        sprint_names = []
        completed_sprints = []
        current_sprint = None
        try:
            boards = self.jira.boards(projectKeyOrID=project_key)
            if not boards:
                raise Exception(f"No boards found for project {project_key}")
            board_id = boards[0].id
            # Get all sprints (closed and active)
            all_sprints = self.jira.sprints(board_id, state='active') + self.jira.sprints(board_id, state='closed')
            # Remove duplicates (by id)
            seen_sprint_ids = set()
            unique_sprints = []
            for s in all_sprints:
                if s.id not in seen_sprint_ids:
                    unique_sprints.append(s)
                    seen_sprint_ids.add(s.id)
            # Sort sprints by startDate or id
            def sprint_sort_key(s):
                if hasattr(s, 'startDate') and s.startDate:
                    return s.startDate
                return str(s.id)
            unique_sprints = sorted(unique_sprints, key=sprint_sort_key)
            # Track sprint names and completion
            for sprint in unique_sprints:
                sprint_names.append({'id': sprint.id, 'name': getattr(sprint, 'name', str(sprint.id)), 'state': getattr(sprint, 'state', 'unknown')})
                if getattr(sprint, 'state', '').lower() == 'closed':
                    completed_sprints.append({'id': sprint.id, 'name': getattr(sprint, 'name', str(sprint.id))})
                elif getattr(sprint, 'state', '').lower() == 'active':
                    current_sprint = {'id': sprint.id, 'name': getattr(sprint, 'name', str(sprint.id))}
            # For velocity, use last 6 closed sprints
            sprints = [s for s in unique_sprints if getattr(s, 'state', '').lower() == 'closed'][-6:]
            for sprint in sprints:
                print(f"Processing sprint: {getattr(sprint, 'name', sprint.id)} (ID: {sprint.id})")
                jql = f'project={project_key} AND sprint={sprint.id} AND issuetype in (Story, Bug, Task)'
                issues = self.jira.search_issues(jql, maxResults=1000)
                for i in issues:
                    print(f"Issue: {i.key}, Type: {i.fields.issuetype.name}, Status: {i.fields.status.name}")
                completed_stories = sum(
                    1 for i in issues
                    if i.fields.issuetype.name in ['Story', 'Task']
                    and 'done' in i.fields.status.name.lower()
                )
                bugs_created = sum(1 for i in issues if i.fields.issuetype.name == 'Bug')
                team_members = set(
                    getattr(i.fields.assignee, 'displayName', None) or getattr(i.fields.assignee, 'accountId', 'unassigned')
                    for i in issues if i.fields.assignee
                )
                velocity = completed_stories
                # Use endDate if available, else fallback to sprint name or ID
                if hasattr(sprint, 'endDate') and sprint.endDate:
                    sprint_date = str(sprint.endDate)[:10]
                elif hasattr(sprint, 'name') and sprint.name:
                    sprint_date = str(sprint.name)
                else:
                    sprint_date = str(sprint.id)
                velocity_history.append({
                    'date': sprint_date,
                    'velocity': velocity,
                    'team_size': len(team_members),
                    'completed_stories': completed_stories,
                    'bugs_created': bugs_created
                })
                for i in issues:
                    if i.fields.assignee:
                        assignee = getattr(i.fields.assignee, 'displayName', None) or getattr(i.fields.assignee, 'accountId', 'unassigned')
                    else:
                        assignee = 'unassigned'
                    if assignee not in team_workload:
                        team_workload[assignee] = {'stories_assigned': 0, 'hours_logged': 0, 'overtime_hours': 0, 'stress_indicators': 0}
                    if i.fields.issuetype.name in ['Story', 'Task']:
                        team_workload[assignee]['stories_assigned'] += 1
            # Optionally, fetch worklogs for hours_logged, etc.
        except Exception as e:
            print(f"Error fetching Jira data: {e}")
        # Add checks for empty data
        if not velocity_history:
            print(f"No velocity history found for project {project_key}. Ensure the project has closed sprints and issues.")
        if not team_workload:
            print(f"No team workload data found for project {project_key}. Ensure issues are assigned to users.")
        return {
            'velocity_history': velocity_history,
            'team_workload': team_workload,
            'sprint_names': sprint_names,
            'completed_sprints': completed_sprints,
            'current_sprint': current_sprint
        }

 

        # ...existing code...

    def analyze_velocity_trends(self, lookback_weeks: int = 8) -> dict:

        """Analyze velocity trends and predict future performance"""
        if not self.project_key:
            return {"error": "No Jira project key set. Please set the project key using the set_project_key tool."}
        if not self.project_metrics:
            return {"error": "No project metrics loaded. Please set the project key using the set_project_key tool."}
        
        try:

            # Convert to DataFrame for analysis

            df = pd.DataFrame(self.project_metrics['velocity_history'])

            df['date'] = pd.to_datetime(df['date'])

            df = df.sort_values('date')

 

            # Calculate trends

            recent_velocities = df['velocity'].tail(lookback_weeks).tolist()

            if len(recent_velocities) < 2:

                return {'error': 'Insufficient data for trend analysis'}

 

            # Linear regression for trend

            X = np.arange(len(recent_velocities)).reshape(-1, 1)

            y = np.array(recent_velocities)

 

            model = LinearRegression()

            model.fit(X, y)

 

            trend_slope = model.coef_[0]

            trend_direction = 'increasing' if trend_slope > 0.5 else 'decreasing' if trend_slope < -0.5 else 'stable'

 

            # Calculate velocity statistics

            avg_velocity = np.mean(recent_velocities)

            velocity_std = np.std(recent_velocities)

            current_velocity = recent_velocities[-1]

            previous_velocity = recent_velocities[-2] if len(recent_velocities) > 1 else current_velocity

 

            # Velocity change percentage

            velocity_change = ((current_velocity - previous_velocity) / previous_velocity) * 100 if previous_velocity > 0 else 0

 

            # Predict next sprint velocity

            next_velocity_pred = model.predict([[len(recent_velocities)]])[0]

 

            # Confidence interval

            confidence_lower = max(0, next_velocity_pred - velocity_std)

            confidence_upper = next_velocity_pred + velocity_std

 

            # Risk assessment

            risk_factors = []

            if velocity_change < -self.risk_thresholds['velocity_decline'] * 100:

                risk_factors.append(f"Velocity declined by {abs(velocity_change):.1f}%")

 

            if trend_slope < -1:

                risk_factors.append("Strong negative velocity trend detected")

 

            # Overall health score (0-100)

            health_score = 100

            if velocity_change < 0:

                health_score -= abs(velocity_change)

            if trend_slope < 0:

                health_score -= abs(trend_slope) * 10

 

            health_score = max(0, min(100, health_score))

 

            return {

                'velocity_metrics': {

                    'current_velocity': current_velocity,

                    'average_velocity': round(avg_velocity, 1),

                    'velocity_change_percent': round(velocity_change, 1),

                    'trend_direction': trend_direction,

                    'trend_slope': round(trend_slope, 2)

                },

                'predictions': {

                    'next_sprint_velocity': round(next_velocity_pred, 1),

                    'confidence_range': {

                        'lower': round(confidence_lower, 1),

                        'upper': round(confidence_upper, 1)

                    }

                },

                'health_assessment': {

                    'health_score': round(health_score),

                    'risk_level': self._get_risk_level(health_score),

                    'risk_factors': risk_factors

                },

                'recommendations': self._generate_velocity_recommendations(trend_direction, velocity_change, risk_factors)

            }

 

        except Exception as e:

            return {

                'error': str(e),

                'fallback_health_score': 75

            }

 

    def detect_burnout_risk(self) -> dict:

        """Analyze team workload and detect burnout risks"""

        if not self.project_key:
            return {"error": "No Jira project key set. Please set the project key using the set_project_key tool."}
        if not self.project_metrics:
            return {"error": "No project metrics loaded. Please set the project key using the set_project_key tool."}
        

        try:

            burnout_analysis = {}

            team_risk_score = 0

            high_risk_members = []

 

            for member_id, workload in self.project_metrics['team_workload'].items():

                # Calculate individual burnout risk factors

                risk_factors = []

                risk_score = 0

 

                # Overtime factor

                overtime = workload.get('overtime_hours', 0)

                if overtime > self.risk_thresholds['overtime_threshold']:

                    risk_factors.append(f"High overtime: {overtime} hours")

                    risk_score += (overtime / 5) * 20  # 20 points per 5 hours overtime

 

                # Story load factor

                stories_assigned = workload.get('stories_assigned', 0)

                if stories_assigned > self.risk_thresholds['story_load_threshold']:

                    risk_factors.append(f"Overloaded: {stories_assigned} stories")

                    risk_score += (stories_assigned - 8) * 5

 

                # Stress indicators

                stress_indicators = workload.get('stress_indicators', 0)

                if stress_indicators > 2:

                    risk_factors.append(f"High stress indicators: {stress_indicators}")

                    risk_score += stress_indicators * 15

 

                # Work hours factor

                hours_logged = workload.get('hours_logged', 40)

                if hours_logged > 45:

                    risk_factors.append(f"Long hours: {hours_logged}h/week")

                    risk_score += (hours_logged - 40) * 2

 

                # Classify burnout risk

                if risk_score >= 60:

                    burnout_risk = 'high'

                    high_risk_members.append(member_id)

                elif risk_score >= 30:

                    burnout_risk = 'medium'

                else:

                    burnout_risk = 'low'

 

                burnout_analysis[member_id] = {

                    'risk_score': min(100, risk_score),

                    'risk_level': burnout_risk,

                    'risk_factors': risk_factors,

                    'workload_metrics': workload,

                    'recommendations': self._generate_burnout_recommendations(burnout_risk, risk_factors)

                }

 

                team_risk_score += risk_score

 

            # Team-level analysis

            avg_team_risk = team_risk_score / len(self.project_metrics['team_workload'])

 

            team_health = {

                'overall_risk_score': round(avg_team_risk),

                'overall_risk_level': self._get_risk_level(100 - avg_team_risk),

                'high_risk_members': high_risk_members,

                'members_analyzed': len(self.project_metrics['team_workload']),

                'immediate_actions': self._generate_team_burnout_actions(high_risk_members, avg_team_risk)

            }

 

            return {

                'individual_analysis': burnout_analysis,

                'team_health': team_health,

                'analysis_timestamp': datetime.now().isoformat()

            }

 

        except Exception as e:

            return {

                'error': str(e),

                'team_health': {'overall_risk_level': 'unknown'}

            }



    def predict_project_risks(self, project_data: Optional[dict] = None) -> dict:
        """
        Predict potential project risks using ML and historical data with comprehensive error handling.

        Returns:
            dict: Complete risk analysis with predictions, warnings, and recommendations
        """
        if not self.project_key:
            return {"error": "No Jira project key set. Please set the project key using the set_project_key tool."}

        if not self.project_metrics:
            return {"error": "No project metrics loaded. Please set the project key using the set_project_key tool."}

        try:
            # Get velocity history with validation
            velocity_history = self.project_metrics.get('velocity_history', [])

            if not velocity_history or not isinstance(velocity_history, list):
                return {
                    "error": "No velocity history available",
                    "overall_assessment": {"overall_risk_score": 50, "risk_level": "medium", "confidence": 0},
                    "early_warnings": ["No historical data available for risk prediction"],
                    "recommended_actions": ["Ensure project has completed sprints with tracked metrics"],
                    "next_review_date": (datetime.now() + timedelta(days=7)).isoformat()
                }

            # Check minimum data requirements
            if len(velocity_history) < 3:
                return {
                    "risk_predictions": self._generate_basic_risk_predictions(),
                    "overall_assessment": {"overall_risk_score": 50, "risk_level": "medium", "confidence": 30},
                    "early_warnings": [f"Limited historical data ({len(velocity_history)} sprints). Need 3+ sprints for reliable predictions"],
                    "recommended_actions": ["Continue collecting sprint data for better risk analysis"],
                    "next_review_date": (datetime.now() + timedelta(days=7)).isoformat()
                }

            # Create DataFrame with safe defaults
            df = self._create_safe_dataframe(velocity_history)

            if df.empty or len(df) < 2:
                return {
                    "error": "Unable to create valid dataset from velocity history",
                    "overall_assessment": {"overall_risk_score": 50, "risk_level": "medium", "confidence": 0}
                }

            # Feature engineering with error handling
            df = self._engineer_features_safely(df)

            # Anomaly detection (only if sufficient data)
            df = self._detect_anomalies_safely(df)

            # Generate risk predictions
            risk_predictions = self._generate_comprehensive_risk_predictions(df)

            # Calculate overall assessment
            overall_assessment = self._calculate_overall_risk_assessment(risk_predictions, len(df))

            # Generate warnings and recommendations
            early_warnings = self._generate_early_warnings(df, risk_predictions)
            recommended_actions = self._generate_risk_mitigation_actions(risk_predictions)

            return {
                'risk_predictions': risk_predictions,
                'overall_assessment': overall_assessment,
                'early_warnings': early_warnings,
                'recommended_actions': recommended_actions,
                'next_review_date': (datetime.now() + timedelta(days=7)).isoformat(),
                'analysis_metadata': {
                    'data_points_analyzed': len(df),
                    'anomalies_detected': int(df['is_anomaly'].sum()) if 'is_anomaly' in df.columns else 0,
                    'confidence_factors': self._get_confidence_factors(df)
                }
            }

        except Exception as e:
            return {
                'error': f"Risk prediction failed: {str(e)}",
                'overall_assessment': {'overall_risk_score': 50, 'risk_level': 'medium', 'confidence': 0},
                'debug_info': {
                    'velocity_history_length': len(self.project_metrics.get('velocity_history', [])),
                    'team_workload_keys': list(self.project_metrics.get('team_workload', {}).keys()),
                    'exception_type': type(e).__name__
                },
                'fallback_recommendations': [
                    "Review project data quality",
                    "Ensure sprint metrics are properly tracked",
                    "Check team workload data completeness"
                ]
            }

    def _create_safe_dataframe(self, velocity_history: list) -> pd.DataFrame:
        """Create DataFrame with safe defaults and validation."""
        try:
            df = pd.DataFrame(velocity_history)

            # Ensure required columns exist with safe defaults
            required_columns = {
                'date': datetime.now().date().isoformat(),
                'velocity': 0,
                'completed_stories': 0,
                'bugs_created': 0,
                'team_size': 1
            }

            for col, default in required_columns.items():
                if col not in df.columns:
                    df[col] = default
                else:
                    # Handle missing values
                    df[col] = df[col].fillna(default)

            # Convert and validate data types
            df['date'] = pd.to_datetime(df['date'], errors='coerce')
            df['velocity'] = pd.to_numeric(df['velocity'], errors='coerce').fillna(0).clip(lower=0)
            df['completed_stories'] = pd.to_numeric(df['completed_stories'], errors='coerce').fillna(0).clip(lower=0)
            df['bugs_created'] = pd.to_numeric(df['bugs_created'], errors='coerce').fillna(0).clip(lower=0)
            df['team_size'] = pd.to_numeric(df['team_size'], errors='coerce').fillna(1).clip(lower=1)

            # Sort by date
            df = df.sort_values('date').reset_index(drop=True)

            # Remove rows with invalid dates
            df = df.dropna(subset=['date'])

            return df

        except Exception as e:
            print(f"Error creating safe dataframe: {e}")
            return pd.DataFrame()

    def _engineer_features_safely(self, df: pd.DataFrame) -> pd.DataFrame:
        """Engineer features with comprehensive error handling."""
        try:
            # Velocity change (percentage)
            df['velocity_change'] = df['velocity'].pct_change() * 100
            df['velocity_change'] = df['velocity_change'].fillna(0).replace([np.inf, -np.inf], 0)

            # Bug ratio (bugs per completed story)
            df['bug_ratio'] = np.where(
                df['completed_stories'] > 0,
                df['bugs_created'] / df['completed_stories'],
                0
            )
            df['bug_ratio'] = df['bug_ratio'].replace([np.inf, -np.inf], 0).fillna(0)

            # Team efficiency (velocity per team member)
            df['team_efficiency'] = np.where(
                df['team_size'] > 0,
                df['velocity'] / df['team_size'],
                0
            )
            df['team_efficiency'] = df['team_efficiency'].replace([np.inf, -np.inf], 0).fillna(0)

            # Quality score (inverse of bug ratio, normalized to 0-1)
            df['quality_score'] = 1.0 / (1.0 + df['bugs_created'])
            df['quality_score'] = df['quality_score'].fillna(0.5).clip(0, 1)

            # Productivity trend (moving average of velocity)
            window_size = min(3, len(df))
            df['velocity_ma'] = df['velocity'].rolling(window=window_size, min_periods=1).mean()

            return df

        except Exception as e:
            print(f"Error in feature engineering: {e}")
            # Return original df if feature engineering fails
            return df

    def _detect_anomalies_safely(self, df: pd.DataFrame) -> pd.DataFrame:
        """Detect anomalies using Isolation Forest with error handling."""
        try:
            # Only run anomaly detection if we have enough data points
            min_points_for_anomaly = 5

            if len(df) < min_points_for_anomaly:
                df['is_anomaly'] = False
                df['anomaly_score'] = 0.0
                return df

            # Select features for anomaly detection
            feature_columns = ['velocity', 'team_efficiency', 'bug_ratio', 'quality_score']
            available_features = [col for col in feature_columns if col in df.columns]

            if not available_features:
                df['is_anomaly'] = False
                df['anomaly_score'] = 0.0
                return df

            X = df[available_features].copy()

            # Handle any remaining infinite or NaN values
            X = X.replace([np.inf, -np.inf], np.nan)
            X = X.fillna(X.mean())  # Fill with column means

            # Adjust contamination based on data size
            contamination = min(0.3, max(0.1, 2.0 / len(X)))

            # Fit Isolation Forest
            iso_forest = IsolationForest(
                contamination=contamination,
                random_state=42,
                n_estimators=50,  # Reduced for smaller datasets
                max_samples='auto'
            )

            anomaly_labels = iso_forest.fit_predict(X)
            anomaly_scores = iso_forest.score_samples(X)

            df['is_anomaly'] = anomaly_labels == -1
            df['anomaly_score'] = anomaly_scores

            return df

        except Exception as e:
            print(f"Error in anomaly detection: {e}")
            df['is_anomaly'] = False
            df['anomaly_score'] = 0.0
            return df

    def _generate_comprehensive_risk_predictions(self, df: pd.DataFrame) -> dict:
        """Generate comprehensive risk predictions from processed data."""
        risk_predictions = {}

        try:
            current_metrics = df.iloc[-1]

            # 1. Scope Creep Risk Analysis
            scope_risk = self._analyze_scope_creep_risk(df)
            risk_predictions['scope_creep'] = scope_risk

            # 2. Schedule Delay Risk Analysis
            schedule_risk = self._analyze_schedule_delay_risk(df)
            risk_predictions['schedule_delay'] = schedule_risk

            # 3. Quality Issues Risk Analysis
            quality_risk = self._analyze_quality_risk(df)
            risk_predictions['quality_issues'] = quality_risk

            # 4. Team Capacity Risk Analysis
            capacity_risk = self._analyze_team_capacity_risk()
            risk_predictions['team_capacity'] = capacity_risk

            # 5. Velocity Decline Risk Analysis
            velocity_risk = self._analyze_velocity_decline_risk(df)
            risk_predictions['velocity_decline'] = velocity_risk

            return risk_predictions

        except Exception as e:
            print(f"Error generating risk predictions: {e}")
            return self._generate_basic_risk_predictions()

    def _analyze_scope_creep_risk(self, df: pd.DataFrame) -> dict:
        """Analyze scope creep risk based on bug creation patterns."""
        try:
            recent_window = min(3, len(df))
            historical_window = max(1, len(df) - recent_window)

            recent_bugs = df['bugs_created'].tail(recent_window).mean()
            historical_bugs = df['bugs_created'].head(historical_window).mean() if historical_window > 0 else recent_bugs

            if historical_bugs <= 0:
                risk_probability = min(30, recent_bugs * 10)  # Base risk on absolute bug count
                severity = 'medium' if risk_probability > 20 else 'low'
                indicators = [f"Recent bug rate: {recent_bugs:.1f} (no baseline)"]
            else:
                change_ratio = recent_bugs / historical_bugs
                risk_probability = max(0, min(100, (change_ratio - 1.0) * 50))
                severity = 'high' if risk_probability > 40 else 'medium' if risk_probability > 20 else 'low'
                indicators = [f"Bug creation trend: {recent_bugs:.1f} vs {historical_bugs:.1f} (ratio: {change_ratio:.2f})"]

            return {
                'probability': round(risk_probability, 1),
                'severity': severity,
                'indicators': indicators,
                'trend': 'increasing' if recent_bugs > historical_bugs else 'stable'
            }

        except Exception as e:
            return {
                'probability': 25.0,
                'severity': 'medium',
                'indicators': [f"Scope analysis failed: {str(e)}"],
                'trend': 'unknown'
            }

    def _analyze_schedule_delay_risk(self, df: pd.DataFrame) -> dict:
        """Analyze schedule delay risk based on velocity trends."""
        try:
            if len(df) < 3:
                return {
                    'probability': 30.0,
                    'severity': 'medium',
                    'indicators': ['Insufficient data for schedule analysis'],
                    'trend': 'unknown'
                }

            recent_velocity = df['velocity'].tail(3).mean()
            historical_velocity = df['velocity'].head(max(1, len(df) - 3)).mean()

            if historical_velocity <= 0:
                risk_probability = 25.0
                severity = 'medium'
                trend = 'unknown'
                indicators = [f"Current velocity: {recent_velocity:.1f}"]
            else:
                velocity_ratio = recent_velocity / historical_velocity
                risk_probability = max(0, min(100, (1.0 - velocity_ratio) * 75))
                severity = 'high' if risk_probability > 35 else 'medium' if risk_probability > 15 else 'low'
                trend = 'declining' if velocity_ratio < 0.9 else 'improving' if velocity_ratio > 1.1 else 'stable'
                indicators = [f"Velocity trend: {recent_velocity:.1f} vs {historical_velocity:.1f} (ratio: {velocity_ratio:.2f})"]

            return {
                'probability': round(risk_probability, 1),
                'severity': severity,
                'indicators': indicators,
                'trend': trend
            }

        except Exception as e:
            return {
                'probability': 30.0,
                'severity': 'medium',
                'indicators': [f"Schedule analysis failed: {str(e)}"],
                'trend': 'unknown'
            }

    def _analyze_quality_risk(self, df: pd.DataFrame) -> dict:
        """Analyze quality risk based on bug ratios and quality scores."""
        try:
            current_quality = df['quality_score'].tail(3).mean()
            bug_ratio_trend = df['bug_ratio'].tail(3).mean()

            # Risk increases as quality decreases
            quality_risk = max(0, min(100, (1.0 - current_quality) * 80))

            # Additional risk from high bug ratios
            if bug_ratio_trend > 0.5:  # More than 0.5 bugs per story
                quality_risk += 20
            elif bug_ratio_trend > 0.3:
                quality_risk += 10

            quality_risk = min(100, quality_risk)
            severity = 'high' if quality_risk > 50 else 'medium' if quality_risk > 25 else 'low'

            indicators = [
                f"Quality score: {current_quality:.2f}",
                f"Bug ratio: {bug_ratio_trend:.2f} bugs per story"
            ]

            return {
                'probability': round(quality_risk, 1),
                'severity': severity,
                'indicators': indicators,
                'trend': 'declining' if current_quality < 0.7 else 'stable'
            }

        except Exception as e:
            return {
                'probability': 25.0,
                'severity': 'medium',
                'indicators': [f"Quality analysis failed: {str(e)}"],
                'trend': 'unknown'
            }

    def _analyze_velocity_decline_risk(self, df: pd.DataFrame) -> dict:
        """Analyze velocity decline risk based on recent trends."""
        try:
            if len(df) < 2:
                return {
                    'probability': 20.0,
                    'severity': 'low',
                    'indicators': ['Insufficient data for velocity analysis'],
                    'trend': 'unknown'
                }

            # Calculate velocity trend over recent sprints
            recent_velocities = df['velocity'].tail(min(4, len(df))).tolist()
            velocity_changes = df['velocity_change'].tail(3).tolist()

            # Check for consecutive declines
            consecutive_declines = 0
            for change in velocity_changes:
                if change < -5:  # 5% decline threshold
                    consecutive_declines += 1
                else:
                    break

            # Risk calculation
            avg_decline = sum(v for v in velocity_changes if v < 0) / len(velocity_changes) if velocity_changes else 0
            decline_risk = max(0, min(100, abs(avg_decline) + (consecutive_declines * 15)))

            severity = 'high' if decline_risk > 40 else 'medium' if decline_risk > 20 else 'low'
            trend = 'declining' if consecutive_declines >= 2 else 'stable'

            indicators = [
                f"Recent velocity trend: {avg_decline:.1f}%",
                f"Consecutive declines: {consecutive_declines}",
                f"Current velocity: {recent_velocities[-1] if recent_velocities else 0}"
            ]

            return {
                'probability': round(decline_risk, 1),
                'severity': severity,
                'indicators': indicators,
                'trend': trend
            }

        except Exception as e:
            return {
                'probability': 25.0,
                'severity': 'medium',
                'indicators': [f"Velocity analysis failed: {str(e)}"],
                'trend': 'unknown'
            }

    def _analyze_team_capacity_risk(self) -> dict:
        """Analyze team capacity risk based on workload metrics."""
        try:
            team_workload = self.project_metrics.get('team_workload', {})

            if not team_workload:
                return {
                    'probability': 30.0,
                    'severity': 'medium',
                    'indicators': ['No team workload data available'],
                    'trend': 'unknown'
                }

            # Calculate team capacity metrics
            total_members = len(team_workload)
            high_workload_members = 0
            total_hours = 0
            total_overtime = 0
            total_stories = 0

            for member_id, workload in team_workload.items():
                hours = workload.get('hours_logged', 40)
                overtime = workload.get('overtime_hours', 0)
                stories = workload.get('stories_assigned', 0)

                total_hours += hours
                total_overtime += overtime
                total_stories += stories

                if hours > 45 or overtime > 5 or stories > 8:
                    high_workload_members += 1

            # Risk calculation
            avg_hours = total_hours / total_members if total_members > 0 else 40
            avg_overtime = total_overtime / total_members if total_members > 0 else 0
            overload_ratio = high_workload_members / total_members if total_members > 0 else 0

            capacity_risk = 0
            if avg_hours > 45:
                capacity_risk += (avg_hours - 40) * 2
            if avg_overtime > 3:
                capacity_risk += avg_overtime * 5
            if overload_ratio > 0.3:
                capacity_risk += overload_ratio * 30

            capacity_risk = min(100, capacity_risk)
            severity = 'high' if capacity_risk > 50 else 'medium' if capacity_risk > 25 else 'low'

            indicators = [
                f"Average hours per week: {avg_hours:.1f}",
                f"Average overtime: {avg_overtime:.1f} hours",
                f"High workload members: {high_workload_members}/{total_members}",
                f"Overload ratio: {overload_ratio:.1%}"
            ]

            return {
                'probability': round(capacity_risk, 1),
                'severity': severity,
                'indicators': indicators,
                'trend': 'increasing' if overload_ratio > 0.4 else 'stable'
            }

        except Exception as e:
            return {
                'probability': 30.0,
                'severity': 'medium',
                'indicators': [f"Capacity analysis failed: {str(e)}"],
                'trend': 'unknown'
            }

    def _calculate_overall_risk_assessment(self, risk_predictions: dict, data_points: int) -> dict:
        """Calculate overall risk assessment from individual predictions."""
        try:
            if not risk_predictions:
                return {
                    'overall_risk_score': 50,
                    'risk_level': 'medium',
                    'confidence': 0
                }

            # Extract probabilities with weights
            risk_weights = {
                'scope_creep': 0.15,
                'schedule_delay': 0.25,
                'quality_issues': 0.20,
                'team_capacity': 0.25,
                'velocity_decline': 0.15
            }

            weighted_risks = []
            total_weight = 0

            for risk_type, risk_data in risk_predictions.items():
                if isinstance(risk_data, dict) and 'probability' in risk_data:
                    weight = risk_weights.get(risk_type, 0.1)
                    probability = float(risk_data['probability'])
                    weighted_risks.append(probability * weight)
                    total_weight += weight

            if total_weight > 0:
                overall_risk = sum(weighted_risks) / total_weight
            else:
                overall_risk = 50.0

            # Calculate confidence based on data availability
            confidence = min(90, max(20, data_points * 10))

            # Adjust confidence based on data quality
            if data_points < 3:
                confidence = max(confidence, 30)
            elif data_points >= 6:
                confidence = max(confidence, 70)

            risk_level = self._get_risk_level(100 - overall_risk)

            return {
                'overall_risk_score': round(overall_risk, 1),
                'risk_level': risk_level,
                'confidence': confidence,
                'risk_distribution': {k: v.get('probability', 0) for k, v in risk_predictions.items() if isinstance(v, dict)}
            }

        except Exception as e:
            return {
                'overall_risk_score': 50.0,
                'risk_level': 'medium',
                'confidence': 20,
                'error': f"Assessment calculation failed: {str(e)}"
            }

    def _generate_early_warnings(self, df: pd.DataFrame, risk_predictions: dict) -> list:
        """Generate early warning alerts based on analysis."""
        warnings = []

        try:
            # High severity risks
            high_severity_risks = [
                risk_type for risk_type, risk_data in risk_predictions.items()
                if isinstance(risk_data, dict) and risk_data.get('severity') == 'high'
            ]

            if high_severity_risks:
                warnings.append(f"High-severity risks detected: {', '.join(high_severity_risks)}")

            # Anomaly warnings
            if 'is_anomaly' in df.columns and df['is_anomaly'].iloc[-1]:
                warnings.append("Current sprint metrics show anomalous patterns")

            # Trend warnings
            declining_trends = [
                risk_type for risk_type, risk_data in risk_predictions.items()
                if isinstance(risk_data, dict) and risk_data.get('trend') == 'declining'
            ]

            if len(declining_trends) >= 2:
                warnings.append(f"Multiple declining trends detected: {', '.join(declining_trends)}")

            # Velocity warnings
            if 'velocity_change' in df.columns:
                recent_velocity_change = df['velocity_change'].tail(2).mean()
                if recent_velocity_change < -15:
                    warnings.append(f"Significant velocity decline: {recent_velocity_change:.1f}%")

            # Team capacity warnings
            team_capacity_risk = risk_predictions.get('team_capacity', {})
            if team_capacity_risk.get('severity') == 'high':
                warnings.append("Team capacity at critical levels - burnout risk detected")

            return warnings[:5]  # Limit to top 5 warnings

        except Exception as e:
            return [f"Warning generation failed: {str(e)}"]

    def _generate_basic_risk_predictions(self) -> dict:
        """Generate basic risk predictions when detailed analysis fails."""
        return {
            'scope_creep': {
                'probability': 25.0,
                'severity': 'medium',
                'indicators': ['Basic risk assessment - limited data'],
                'trend': 'unknown'
            },
            'schedule_delay': {
                'probability': 30.0,
                'severity': 'medium',
                'indicators': ['Basic risk assessment - limited data'],
                'trend': 'unknown'
            },
            'quality_issues': {
                'probability': 20.0,
                'severity': 'low',
                'indicators': ['Basic risk assessment - limited data'],
                'trend': 'unknown'
            },
            'team_capacity': {
                'probability': 25.0,
                'severity': 'medium',
                'indicators': ['Basic risk assessment - limited data'],
                'trend': 'unknown'
            }
        }

    def _get_confidence_factors(self, df: pd.DataFrame) -> dict:
        """Get factors affecting analysis confidence."""
        return {
            'data_points': len(df),
            'data_completeness': (df.notna().sum() / len(df)).mean() if len(df) > 0 else 0,
            'historical_coverage': 'good' if len(df) >= 6 else 'limited' if len(df) >= 3 else 'insufficient',
            'anomaly_detection_enabled': 'is_anomaly' in df.columns and len(df) >= 5
        }

 

    def _calculate_team_capacity_risk(self) -> float:

        """Calculate team capacity risk based on workload distribution"""

        workload_data = list(self.project_metrics['team_workload'].values())

 

        # Calculate metrics

        avg_hours = np.mean([w['hours_logged'] for w in workload_data])

        max_hours = max([w['hours_logged'] for w in workload_data])

        avg_overtime = np.mean([w['overtime_hours'] for w in workload_data])

 

        # Risk factors

        capacity_risk = 0

 

        if avg_hours > 40:

            capacity_risk += (avg_hours - 40) * 2

 

        if max_hours > 50:

            capacity_risk += (max_hours - 50) * 3

 

        if avg_overtime > 3:

            capacity_risk += avg_overtime * 5

 

        return min(100, capacity_risk)

 

    def generate_health_report(self, include_predictions: bool = True) -> dict:

        """Generate comprehensive project health report"""

        if not self.project_key:
            return {"error": "No Jira project key set. Please set the project key using the set_project_key tool."}
        if not self.project_metrics:
            return {"error": "No project metrics loaded. Please set the project key using the set_project_key tool."}
        if not self.project_metrics.get('velocity_history') or not self.project_metrics.get('team_workload'):
            return {"error": "No Jira sprint or issue data found for this project. Please check that the project has closed sprints and assigned issues."}

        try:

            # Get all analyses

            velocity_analysis = self.analyze_velocity_trends()

            burnout_analysis = self.detect_burnout_risk()

 

            risk_analysis = {}

            if include_predictions:

                risk_analysis = self.predict_project_risks()

 

            # Calculate overall project health score

            health_components = []

 

            if 'health_assessment' in velocity_analysis:

                health_components.append(velocity_analysis['health_assessment']['health_score'])

 

            if 'team_health' in burnout_analysis:

                burnout_health = 100 - burnout_analysis['team_health']['overall_risk_score']

                health_components.append(burnout_health)

 

            if 'overall_assessment' in risk_analysis:

                risk_health = 100 - risk_analysis['overall_assessment']['overall_risk_score']

                health_components.append(risk_health)

 

            overall_health = np.mean(health_components) if health_components else 75

 

            # Generate executive summary

            executive_summary = self._generate_executive_summary(

                overall_health, velocity_analysis, burnout_analysis, risk_analysis

            )

 

            return {

                'report_metadata': {

                    'generated_at': datetime.now().isoformat(),

                    'report_type': 'comprehensive_health_analysis',

                    'coverage_period': '8_weeks',

                    'next_report_due': (datetime.now() + timedelta(days=7)).isoformat()

                },

                'executive_summary': executive_summary,

                'overall_health': {

                    'health_score': round(overall_health),

                    'health_grade': self._get_health_grade(overall_health),

                    'trend': self._determine_health_trend(velocity_analysis)

                },

                'detailed_analysis': {

                    'velocity_trends': velocity_analysis,

                    'team_burnout': burnout_analysis,

                    'risk_predictions': risk_analysis if include_predictions else {}

                },

                'action_items': self._consolidate_action_items(velocity_analysis, burnout_analysis, risk_analysis),

                'kpi_dashboard': self._generate_kpi_dashboard()

            }

 

        except Exception as e:

            return {

                'error': str(e),

                'report_metadata': {'generated_at': datetime.now().isoformat()},

                'overall_health': {'health_score': 50, 'health_grade': 'C'}

            }

 

    def _get_risk_level(self, score: float) -> str:

        """Convert score to risk level"""

        if score >= 80:

            return 'low'

        elif score >= 60:

            return 'medium'

        else:

            return 'high'

 

    def _get_health_grade(self, score: float) -> str:

        """Convert health score to letter grade"""

        if score >= 90:

            return 'A'

        elif score >= 80:

            return 'B'

        elif score >= 70:

            return 'C'

        elif score >= 60:

            return 'D'

        else:

            return 'F'

 

    def _generate_velocity_recommendations(self, trend_direction: str, velocity_change: float, risk_factors: list) -> list:

        """Generate velocity-specific recommendations"""

        recommendations = []

 

        if trend_direction == 'decreasing':

            recommendations.append("Investigate causes of velocity decline - check team capacity and blockers")

            recommendations.append("Consider backlog refinement to ensure story estimates are accurate")

 

        if velocity_change < -10:

            recommendations.append("Immediate action needed - significant velocity drop detected")

 

        if len(risk_factors) > 2:

            recommendations.append("Multiple risk factors identified - schedule team retrospective")

 

        return recommendations

 

    def _generate_burnout_recommendations(self, risk_level: str, risk_factors: list) -> list:

        """Generate burnout prevention recommendations"""

        recommendations = []

 

        if risk_level == 'high':

            recommendations.extend([

                "Immediate workload redistribution required",

                "Schedule one-on-one check-in with team member",

                "Consider temporary resource reallocation"

            ])

        elif risk_level == 'medium':

            recommendations.extend([

                "Monitor workload closely",

                "Ensure adequate breaks and time off",

                "Review story assignment distribution"

            ])

 

        return recommendations

 

    def _generate_team_burnout_actions(self, high_risk_members: list, avg_team_risk: float) -> list:

        """Generate team-level burnout prevention actions"""

        actions = []

 

        if high_risk_members:

            actions.append(f"Immediate attention required for {len(high_risk_members)} team members")

 

        if avg_team_risk > 50:

            actions.extend([

                "Team-wide workload review needed",

                "Consider sprint scope reduction",

                "Schedule team wellness check"

            ])

 

        return actions

 

    def _generate_risk_mitigation_actions(self, risk_predictions: dict) -> list:

        """Generate risk mitigation actions"""

        actions = []

 

        for risk_type, risk_data in risk_predictions.items():

            if risk_data['severity'] == 'high':

                if risk_type == 'scope_creep':

                    actions.append("Implement stricter change control process")

                elif risk_type == 'schedule_delay':

                    actions.append("Review sprint commitments and adjust scope")

                elif risk_type == 'quality_issues':

                    actions.append("Increase code review rigor and testing coverage")

                elif risk_type == 'team_capacity':

                    actions.append("Assess team capacity and consider additional resources")

 

        return actions

 

    def _determine_health_trend(self, velocity_analysis: dict) -> str:

        """Determine overall health trend"""

        if 'velocity_metrics' in velocity_analysis:

            trend_direction = velocity_analysis['velocity_metrics'].get('trend_direction', 'stable')

            velocity_change = velocity_analysis['velocity_metrics'].get('velocity_change_percent', 0)

 

            if trend_direction == 'increasing' or velocity_change > 5:

                return 'improving'

            elif trend_direction == 'decreasing' or velocity_change < -5:

                return 'declining'

            else:

                return 'stable'

 

        return 'stable'

 

    def _generate_executive_summary(self, overall_health: float, velocity_analysis: dict, burnout_analysis: dict, risk_analysis: dict) -> dict:

        """Generate executive summary"""

        key_findings = []

 

        # Velocity findings

        if 'velocity_metrics' in velocity_analysis:

            velocity_change = velocity_analysis['velocity_metrics'].get('velocity_change_percent', 0)

            if velocity_change < -10:

                key_findings.append(f"Velocity declined by {abs(velocity_change):.1f}% - requires attention")

            elif velocity_change > 10:

                key_findings.append(f"Velocity improved by {velocity_change:.1f}% - positive trend")

 

        # Burnout findings

        if 'team_health' in burnout_analysis:

            high_risk_members = burnout_analysis['team_health'].get('high_risk_members', [])

            if high_risk_members:

                key_findings.append(f"{len(high_risk_members)} team members at high burnout risk")

 

        # Risk findings

        if 'risk_predictions' in risk_analysis:

            high_risks = [risk for risk, data in risk_analysis['risk_predictions'].items() if data.get('severity') == 'high']

            if high_risks:

                key_findings.append(f"High-severity risks identified: {', '.join(high_risks)}")

 

        return {

            'overall_health_score': round(overall_health),

            'health_status': self._get_health_grade(overall_health),

            'key_findings': key_findings[:3],  # Top 3 findings

            'immediate_actions_required': len(key_findings) > 0

        }

 

    def _consolidate_action_items(self, velocity_analysis: dict, burnout_analysis: dict, risk_analysis: dict) -> list:

        """Consolidate action items from all analyses"""

        all_actions = []

 

        # Velocity recommendations

        if 'recommendations' in velocity_analysis:

            all_actions.extend(velocity_analysis['recommendations'])

 

        # Burnout actions

        if 'team_health' in burnout_analysis and 'immediate_actions' in burnout_analysis['team_health']:

            all_actions.extend(burnout_analysis['team_health']['immediate_actions'])

 

        # Risk mitigation actions

        if 'recommended_actions' in risk_analysis:

            all_actions.extend(risk_analysis['recommended_actions'])

 

        # Remove duplicates and prioritize

        unique_actions = list(set(all_actions))

        return unique_actions[:10]  # Top 10 actions

 

    def _generate_kpi_dashboard(self) -> dict:

        """Generate KPI dashboard metrics"""

        # Get latest metrics

        latest_velocity_data = self.project_metrics['velocity_history'][-1]

 

        return {

            'current_velocity': latest_velocity_data['velocity'],

            'team_size': latest_velocity_data['team_size'],

            'stories_completed': latest_velocity_data['completed_stories'],

            'bugs_created': latest_velocity_data['bugs_created'],

            'velocity_per_person': round(latest_velocity_data['velocity'] / latest_velocity_data['team_size'], 1),

            'bug_ratio': round(latest_velocity_data['bugs_created'] / latest_velocity_data['completed_stories'], 2),

            'last_updated': datetime.now().isoformat()

        }

project_health_agent = ProjectHealthAgent().get_agent()