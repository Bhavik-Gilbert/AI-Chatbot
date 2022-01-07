from BotBackend import chatbot
from BotBackend import read_dictionary
from BotBackend import bots

while True:
    print("Hi, what would you like me to do?")
    user_input = input("")
    #exit code
    if user_input.lower() == "goodbye" or user_input.lower() == "quit":
        break
    #outputs list of commands each bot accepts
    if user_input.lower() == "commands":
      for agents in bots:
        print(agents[0] + "'s trigger words =>")
        commands = read_dictionary(agents[1])
        for command in commands:
          print(command)
        print("")
      continue

    response = chatbot.get_response(user_input)
    print(response)
    print("")