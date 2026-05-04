"""PlantDbTeam CrewAI crew definition."""

from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task

from plant_db_team.tools.custom_tool import PlantDatabaseWriter


@CrewBase
class PlantDbTeam():
    """PlantDbTeam crew — researches, formats, validates, and persists plant data to DuckDB."""

    agents_config = 'config/agents.yaml'
    tasks_config = 'config/tasks.yaml'

    @agent
    def plant_researcher(self) -> Agent:
        return Agent(
            config=self.agents_config['plant_researcher'],
            verbose=True,
        )

    @agent
    def data_formatter(self) -> Agent:
        return Agent(
            config=self.agents_config['data_formatter'],
            verbose=True,
        )

    @agent
    def data_validator(self) -> Agent:
        return Agent(
            config=self.agents_config['data_validator'],
            verbose=True,
        )

    @agent
    def database_builder(self) -> Agent:
        return Agent(
            config=self.agents_config['database_builder'],
            verbose=True,
            tools=[PlantDatabaseWriter()],
        )

    @task
    def research_task(self) -> Task:
        return Task(
            config=self.tasks_config['research_task'],
        )

    @task
    def format_task(self) -> Task:
        return Task(
            config=self.tasks_config['format_task'],
        )

    @task
    def validate_task(self) -> Task:
        return Task(
            config=self.tasks_config['validate_task'],
        )

    @task
    def build_database_task(self) -> Task:
        return Task(
            config=self.tasks_config['build_database_task'],
        )

    @crew
    def crew(self) -> Crew:
        """Creates the PlantDbTeam crew."""
        return Crew(
            agents=self.agents,
            tasks=self.tasks,
            process=Process.sequential,
            verbose=True,
        )
