import os
import openai

openai.api_key = "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"
model_input = ""
while True:
  my_input = input("我:")
  model_input += ("我:"+my_input+"\n\n")
  
  response = openai.Completion.create(
    model="text-davinci-003",
    prompt=model_input,
    temperature=0.7,
    max_tokens=256,
    top_p=1,
    frequency_penalty=0,
    presence_penalty=0
  )
  # 顯示回覆
  AI_response = response.choices[0].text
  print(AI_response+"\n")
  model_input += (AI_response+"\n\n")
  # print(str(model_input))