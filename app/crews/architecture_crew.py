from crewai import Agent, Task, Crew


research_agent = Agent(
    role="AI Researcher",
    goal="Research the topic deeply",
    backstory="Expert AI researcher",
    verbose=True
)

architect_agent = Agent(
   role="AI Architect",
   goal="Design scalable architecture",
   backstory="Senior AI architect",
   verbose=True
)

reviewer_agent = Agent(
   role="Technical Reviewer",
   goal="Review and improve final response",
   backstory="Expert technical reviewer",
   verbose=True
)


def run_architecture_crew(user_input: str):

    research_task = Task(
        description=f"""
        Research this topic:
        {user_input}
        """,
        agent=research_agent,
        expected_output="Technical research"
    )

    architecture_task = Task(
      description=f"""
       Create architecture explanation for:
      {user_input}
    """,
       agent=architect_agent,
       expected_output="Architecture explanation"
    )

    review_task = Task(
       description="""
        Review and improve the response
        """,
        agent=reviewer_agent,
        expected_output="Final polished response"
    )

    crew = Crew(
        agents=[
            research_agent,
            architect_agent,
            reviewer_agent
        ],
        tasks=[
            research_task,
           architecture_task,
           review_task
        ],
        verbose=True
    )

    result = crew.kickoff()

    return str(result)