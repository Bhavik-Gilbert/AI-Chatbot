import tkinter as tk

from BotBackend import chatbot
from BotBackend import read_dictionary
from BotBackend import bots
from BotBackend import image_bots

root = tk.Tk()

canvas = tk.Canvas(root, width = 400, height = 300)
canvas.pack()

question = tk.Label(root, text='Hi, what would you like me to do?')
question.config(font=('helvetica', 12))
canvas.create_window(200, 25, window=question)

entry = tk.Entry(root)
canvas.create_window(20)

label = tk.Label(root, text="enter filename or sentence to be analysed\nleave blank for text response")
label.config(font=('helvetica', 12))
canvas.create_window(200, 100, window=label)


data = tk.Entry(root)
canvas.create_window(20)

button = tk.Button(text='Submit', command=getResponse, font=('helvetica', 9, 'bold'))
canvas.create_window(200, 180, window=button)

root.mainloop()

def getResponse():
  user_input = entry.get()
  if user_input.lower() == "goodbye" or user_input.lower() == "quit":
    quit()
    return
  if user_input.lower() == "commands":
    getCommands()
    return
  response = chatbot.get_response(user_input)
  label = tk.Label(root, text=response)
  label.config(font=('helvetica', 10))
  canvas.create_window(200, 100, window=label)

def quit():
  root.destroy()

def getCommands():
    for agents in bots:
      label = tk.Label(root, text=agents[0] + "'s trigger words =>")
      label.config(font=('helvetica', 12))
      canvas.create_window(200, 25, window=label)
      
      commands = read_dictionary(agents[1])
      for command in commands:
        label = tk.Label(root, text=command)
        label.config(font=('helvetica', 10))
        canvas.create_window(200, 100, window=label)