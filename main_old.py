from dotenv import load_dotenv
import os
from crewai import Agent, Task, Crew

# Load environment variables
load_dotenv()

# Check if API key is loaded
openai_key = os.getenv("OPENAI_API_KEY")
print(f"OpenAI API Key loaded: {openai_key[:10]}..." if openai_key else "OpenAI Key NOT found!")

# Define the Researcher Agent
researcher = Agent(
    role='Researcher',
    goal='Gather accurate and detailed information on a given topic',
    backstory='You are an expert researcher with access to the latest data sources.',
    llm="gpt-4o-mini",  # Fast and cheap OpenAI model
    verbose=True,
    allow_delegation=False
)

# Define the Summarizer Agent
summarizer = Agent(
    role='Summarizer',
    goal='Create a concise and clear summary of the research provided',
    backstory='You are a skilled writer who excels at distilling complex information.',
    llm="gpt-4o-mini",
    verbose=True,
    allow_delegation=False
)

# Define the Research Task
research_task = Task(
    description='Research the topic: "The benefits of AI in healthcare"',
    agent=researcher,
    expected_output='A detailed report with key findings'
)

# Define the Summary Task
summary_task = Task(
    description='Summarize the research report provided by the Researcher',
    agent=summarizer,
    expected_output='A 200-300 word summary'
)

# Create the Crew
my_crew = Crew(
    agents=[researcher, summarizer],
    tasks=[research_task, summary_task],
    verbose=True
)

if __name__ == "__main__":
    print("\n🚀 Starting CrewAI execution...\n")
    result = my_crew.kickoff()
    print("\n\n=== FINAL OUTPUT ===")
    print(result)