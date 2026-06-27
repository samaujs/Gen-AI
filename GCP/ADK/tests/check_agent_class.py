import google.adk as adk
print(dir(adk))
try:
    from google.adk import Agent
    print("Agent is present in google.adk")
    for name, field in Agent.model_fields.items():
        print(f"Agent field {name}: {field.annotation}")
except Exception as e:
    print("Error importing Agent:", e)
