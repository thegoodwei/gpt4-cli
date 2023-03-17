import openai
import sqlite3
openai.api_key = os.environ["OPENAI_API_KEY"]
openai.api_key = APIKEY
temp=.5
maxspend=1000
conversation = []
name=""
def chat():
#    instructions="Briefly revise, simplify, output concise keyphrases:" # this works alright
    usermsg=input("type the name of a conversation to load that database, or enter to continue new")
    if usermsg=="":
        instructions=input("Sys instructions.")
        print(instructions)
        conversation.append({'role': 'system', 'content':instructions}) #+ transcript_segments[j]
    else:
        print("todo LOAD conversation from db")
    while usermsg!="kill":
        usermsg=getuserinput()
        conversation.append({'role': 'user', 'content': usermsg})
        response  = openai.ChatCompletion.create(
            model="gpt-4",
            messages= conversation,   #consise 
            #"Completely summarize the following text:\n"+transcript_segment +" \n",
            temperature=temp,
            max_tokens=maxspend,
            top_p=1,
            frequency_penalty=2,
            presence_penalty=-2,
        )
        print("................................")
        print(str(response.choices[-1].message.content))
        print("\n\n")
        
        
def getuserinput():
    usrmsg=input("User Msg: ")
    if usrmsg=="help":
        print("type a message to GPT4. Or, Type 'params' to modify temp or maxspend. Or, type 'save' and continue")
        usrmsg=input("usr msg:")
    if usrmsg=="save":
        conn = sqlite3.connect("gpt4chatconversations.db")
        c = conn.cursor()
        if name=="":
            name=input("name the conversation db to save")
        print(name)
        conn.execute('''CREATE TABLE {name}
             (id INTEGER PRIMARY KEY AUTOINCREMENT,
             message TEXT NOT NULL,
             response TEXT NOT NULL);''')
        for message in conversation:
            conn.execute("INSERT INTO {name} (message, response) VALUES (?, ?)", (message['text'], message['response']))
        conn.commit()
        conn.close()
        print("{name} saved!")
        usrmsg=input("now, \n provide more user input, 'params' to mod, or 'kill' to end")

    if usrmsg=="params":
            params=input("provide new parameters, either temp:0-1 or spend:1-8000     ")
            if "temp" in params:
                temp=params.split(":")[1]
                print("temp is"+temp)
            if "spend" in params:
                maxspend=params.split(":")[1]
                print("maxspend is" + maxspend)
            params=input("provide new parameters, either temp:0-1 or spend:1-8000     ")
            if "temp" in params:
                temp=params.split(":")[1]
                print("temp is"+temp)
            if "maxspend" in params:
                maxspend=params.split(":")[1]
                print("maxspend is" + maxspend)
            usrmsg=input("usr msg:")
    return usrmsg

if __name__ == "__main__": 
    chat() 
