import os
import requests
from dotenv import load_dotenv
from langchain.agents import initialize_agent, Tool
from langchain_community.llms import Ollama

# -------------------------------
# Load API keys
# -------------------------------
load_dotenv()
WEATHER_KEY = os.getenv("WEATHER_API_KEY")

# -------------------------------
# Weather forecast function
# -------------------------------
def get_weather(city, start_date, end_date):
    try:
        url = f"http://api.openweathermap.org/data/2.5/forecast?q={city}&appid={WEATHER_KEY}&units=metric"
        response = requests.get(url)
        data = response.json()

        if data.get("cod") != "200":
            return "Weather data not found. Please check city name or API key."

        forecast = []
        for entry in data["list"]:
            dt_txt = entry["dt_txt"]
            if start_date in dt_txt or end_date in dt_txt:
                forecast.append(f"{dt_txt}: {entry['main']['temp']}°C, {entry['weather'][0]['description']}")

        return "\n".join(forecast) if forecast else "No specific weather data found for the given dates."
    except Exception as e:
        return f"Error fetching weather: {str(e)}"

# -------------------------------
# Travel routes (static)
# -------------------------------
def get_routes(city):
    return f"Recommended transport in {city}: Metro, Taxi, Bus services, and ride-hailing apps (Uber/Bolt)."

# -------------------------------
# AI-generated attractions (no API)
# -------------------------------
def ai_suggest_attractions(city):
    return f"Suggesting top 5 attractions in {city} based on AI knowledge ."

# -------------------------------
# Tools for the Agent
# -------------------------------
tools = [
    Tool(
        name="Weather Info",
        func=lambda q: get_weather(q.split('|')[0], q.split('|')[1], q.split('|')[2]),
        description="Get weather forecast. Format: city|start_date|end_date"
    ),
    Tool(
        name="Tourist Attractions",
        func=ai_suggest_attractions,
        description="Suggest top attractions for a city based on AI knowledge (no API)."
    ),
    Tool(
        name="Travel Routes",
        func=get_routes,
        description="Get recommended transportation methods for a city."
    )
]

# -------------------------------
# Initialize LLaMA 3 LLM
# -------------------------------
llm = Ollama(model="llama3")

# Initialize Agent
agent = initialize_agent(tools, llm, agent="zero-shot-react-description", verbose=True)

# -------------------------------
# Main app
# -------------------------------
def main():
    print(" AI Travel Planner Agent (Powered by LLaMA 3)")
    city = input("Enter destination city: ").strip()
    start = input("Enter start date (YYYY-MM-DD): ").strip()
    end = input("Enter end date (YYYY-MM-DD): ").strip()

    print("\n Generating your travel plan... Please wait...\n")
    query = f"Plan a 3-day trip to {city}. Use weather ({start} to {end}) and suggest attractions and routes. Give a detailed itinerary."
    result = agent.run(query)

    print("\n Travel Plan:\n")
    print(result)

    # Save output to file
    with open("travel_plan.txt", "w", encoding="utf-8") as f:
        f.write(f"Trip Plan for {city} ({start} to {end}):\n\n{result}")
    print("\n Plan saved to travel_plan.txt")

if __name__ == "__main__":
    main()
