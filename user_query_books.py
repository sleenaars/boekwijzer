#Running a user embedding
"""Steps to do in cmd before running this script:
1.  set PINECONE_API_KEY=
    set OPENAI_API_KEY=

2.  cd C:/Users/SanderLeenaars/Documents/SanLLMander/Python Scripts
3.  python m- user_query"""


from pinecone import Pinecone
from openai import OpenAI
from utils_books import answer_query

# Initialize API clients
pc = Pinecone()
index = pc.Index("boekwijzer-index")
client = OpenAI()

# Example query                                             #to get answer in cmd: """query = "Waar zit het notariskantoor? answer = "answer = answer_query(query, index, client), print(answer)"""
query = input("Please enter your question: ")
answer = answer_query(query, index, client)
print("\nAnswer:")
#print(answer)
#query = "Waar zit het notariskantoor?"
#answer = answer_query(query, index, client)
#print(answer)







