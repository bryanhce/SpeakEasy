from langchain_core.prompts import PromptTemplate


def translate_cn():
    prompt = """
        Translate the following Chinese text to English:
        {chinese}
        """

    return PromptTemplate.from_template(prompt)


def feedback_to_user():
    prompt = """
        You are a helpful language learning assistant. 
        
        User Input: "{user_input}"
        Context: "{context}"
        
        Evaluate the latest user's input based on overall relevancy, coherence and complexity.
        Output constructive feedback on the user's input, 
        highlighting overall strengths or enhancements in English.
        Users should be penalised for using English.
        
        Feedback should be concise in 2 sentences.
        """

    return PromptTemplate.from_template(prompt)


def response_to_user():
    prompt = """
            You are a helpful language learning assistant. 
            
            User Input: "{user_input}"
            Context: "{context}"
            
            Based on the user input and context, provide a meaningful and contextually appropriate response in Chinese.
            If the user's input is irrelevant to the context or in English, you should reproach them and
            remind them to continue the conversation in Chinese, focusing on maintaining relevance to the topic.
            
            If the context objective is not met,
            encourage user by concluding your response with a guiding question.
            If context objective is met, conclude appropriately without further questions.
            Response should not be longer than 2 sentences.
            """

    return PromptTemplate.from_template(prompt)


def get_user_score():
    prompt = """
            You are a helpful language learning assistant. 
            User Input: "{user_input}"
            Context: "{context}"
            Evaluate the latest user's input based on overall relevancy, coherence and complexity.
            Users should be penalised for using English.
            Output only a numerical score between 0 and 100.
            """
    return PromptTemplate.from_template(prompt)


def conversation_suggestion():
    prompt = """
            You are a helpful language learning assistant. 
            Provide a structured list response containing suggestions in 
            responding to the previous message and strictly nothing else.
            No ```code declaration needed, just the list object.
             
            Previous Input: "{prev_message}"
            Context:  "{context}"

            Suggestions should be at most 1 sentence without special formatting.
            Requirements:
            - first: A simple and relevant response to the Previous Input in Chinese.
            - second: A meaningful and contextually appropriate response to the Previous Input in Chinese.
            - third: A meaningful and contextually relevant response to the Previous Input in Chinese.

            return a list of suggestions, example: ["first", "second", "third"]
                
            Ensure the response is in valid list format.
            """

    return PromptTemplate.from_template(prompt)


def process_image():
    prompt = """
            Provide a detailed description of the image based on its relevance to a given context.

            User's initial context: '{context}'
            
            If the user's initial context is unclear or inappropriate, output an empty string.
            
            If the image is relevant to the context, provide a detailed description.
            If the image is not relevant, output an empty string.
    """
    return PromptTemplate.from_template(prompt)


def refine_context():
    prompt = """
            You are a helpful language learning assistant tasked with refining user inputs into clear, actionable tasks.
        
            Parameters:
            - Image Description: '{description}'
            - User's Initial Context: '{context}'
        
            Instructions:
            1. If the user's initial context is unclear or inappropriate, 
            use the default context: "Discuss with a friend about how your day has been."
            2. You may incorporate relevant details from the image description only if it helps to enrich the context.
            3. Reframe the context into a coherent and specific objective task. 
            Avoid any special formatting in your response.
        
            Output:
            Provide a refined version of the user's initial context or the default context as a 
            clear, actionable task suitable for a language learning scenario.
            """
    return PromptTemplate.from_template(prompt)


def generate_init_message():
    prompt = """
            You should take on the receiving role of the context,
            without responding to this prompt. 
            The user's objective is '{context}'
            Output an initial message to initiate a conversation 
            as the receiving role of the context in Chinese.
            
            For example, if the user has to order food, you should assume the role of a waiter.
            For example, if the user has to make a phone call, you should be the receiver.
            
            It should be short and less than 2 sentences.
            """

    return PromptTemplate.from_template(prompt)


def get_user_inputs():
    prompt = """
           You are a helpful language learning assistant. 
           Extract ALL text output by the role 'User'. 
           Do not include any text output by 'assistant'. 
           Do not alter the phrasing of the user input in any way.
           User Input: "{user_input}"
           """
    return PromptTemplate.from_template(prompt)


def generate_final_feedback():
    prompt = """
            You are a helpful language learning assistant.
            
            User Input: "{user_input}"
            Context: "{context}"
            
            Output constructive feedback SOLELY about the user's inputs,
            highlighting overall strengths or enhancements:
            - Users should only be using Chinese. Reproach for use of other languages.  
            - Relevancy
            - Coherence
            - Complexity
            
            Feedback should be concise in English and at most 3 sentences without any special formatting.
            """

    return PromptTemplate.from_template(prompt)
