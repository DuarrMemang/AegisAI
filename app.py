import os
from dotenv import load_dotenv

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from langchain_google_genai import ChatGoogleGenerativeAI

def main(news_url):
    load_dotenv()
    api_key = os.getenv("GOOGLE_API_KEY")

    if not api_key:
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            print("Error: GOOGLE_API_KEY or GEMINI_API_KEY not found in environment variables or .env file.")
            return

    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=api_key)
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", 
            "You are a fact-checking AI. "
            "Your task is to judge the accuracy of user-provided text.\n"
            "1. First decide if the text is an **opinion** or a **factual statement**.\n"
            "2. If it is an opinion, respond only with: 'Opinion'.\n"
            "3. If it is a factual statement, respond with either:\n"
            "   - 'Likely True' (if it matches generally known facts)\n"
            "   - 'Likely False' (if it contradicts generally known facts)\n"
            "4. Keep responses concise, without extra explanations unless asked.\n"),
            ("human", f"Check the news article ar {news_url}")
        ]
    )

    fact_check_chain = prompt | llm | StrOutputParser()

    result = fact_check_chain.invoke({"text_to_check": news_url})
    return result

if __name__ == "__main__":
    main()
