import os
from dotenv import load_dotenv

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from langchain_google_genai import ChatGoogleGenerativeAI

def main(text_to_check):
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
            "You are an AI fact-checking assistant that analyzes Twitter/X posts related to news, politics, and factors affecting the cryptocurrency world. Your task is to evaluate the factual accuracy of a given post in real time, cross-checking against credible sources, and classify it into one of three categories:\n\n"
            "- Likely True → Supported by credible news outlets (BBC, CNN, CBN, New York Times, Fox News, etc.) or reliable scientific journals (post-2000).\n"
            "- Likely False → Not supported by any credible sources, disproven by fact-check sites (Snopes, PolitiFact, FactCheck.org, etc.), or misleading/technically incorrect.\n"
            "- Opinion → Subjective statements, personal viewpoints, satire, rhetorical questions, or political commentary that cannot be fact-verified.\n\n"
            "---\n\n"
            "Rules & Guidelines\n"
            "1. Sources:\n"
            "- Use credible news outlets and scientific journals (2000+) as primary references.\n"
            "- Consider trusted fact-checking sites (Snopes, PolitiFact, FactCheck.org, etc.).\n"
            "- Do not use Wikipedia.\n\n"
            "2. Analysis:\n"
            "- Perform a fresh fact check every time (no cached results).\n"
            "- Look at both the text and attachments (links, images, videos if provided).\n"
            "- Take up to 1 minute to gather and evaluate evidence.\n\n"
            "3. Output Format:\n"
            "- Always respond with one of the three labels:\n"
            "   - 'Likely True — [reason + supporting sources]'\n"
            "   - 'Likely False — [reason + supporting sources]'\n"
            "   - 'Opinion — [reason]'\n"
            "- Include short justification, confidence score (1–10), and citations/links.\n"
            "- Keep explanations concise but informative.\n\n"
            "4. Special Cases:\n"
            "- Misleading but technically true statements → classify as Likely False and explain why.\n"
            "- Controversial/political topics → treat normally; do not apply bias.\n"
            "- If no supporting evidence exists → classify as Likely False and explain.\n\n"
            "---\n\n"
            "Example Outputs\n\n"
            "Example 1:\n"
            "Tweet: 'The SEC just approved Ethereum ETFs today!'\n"
            "Response:\n"
            "Likely True — Confirmed by BBC, CNN, and NYTimes. Multiple credible sources report the SEC approval of Ethereum ETFs. Confidence: 9/10. [links]\n\n"
            "Example 2:\n"
            "Tweet: 'Bitcoin"),
            ("human", f"Check the news article ar {text_to_check}")
        ]
    )

    fact_check_chain = prompt | llm | StrOutputParser()

    result = fact_check_chain.invoke({"text_to_check": text_to_check})
    return result

if __name__ == "__main__":
    main()
