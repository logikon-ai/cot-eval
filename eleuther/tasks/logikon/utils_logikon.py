# Copied from Master
def doc_to_text(doc) -> str:
    """
    Answer the following question about the given passage.
    
    Passage: <passage>
    
    Question: <question>
    A. <choice1>
    B. <choice2>
    C. <choice3>
    D. <choice4>
    [E. <choice5>]
        
    Answer:
    """
    k = len(doc["options"])
    choices = ["a", "b", "c", "d", "e"][:k]
    prompt = "Answer the following question about the given passage.\n\n"
    prompt = "Passage: " + doc["passage"] + "\n\n"
    prompt += "Question: " + doc["question"] + "\n"
    for choice, option in zip(choices, doc["options"]):
        prompt += f"{choice.upper()}. {option}\n"
    prompt += "\n"
    prompt += "Answer:"
    return prompt

def doc_to_text_cot(doc) -> str:
    """
    Answer the following question about the given passage. [Base your answer on the reasoning below.]
    
    Passage: <passage>
    
    Question: <question>
    A. <choice1>
    B. <choice2>
    C. <choice3>
    D. <choice4>
    [E. <choice5>]
    
    [Reasoning: <reasoning>]
    
    Answer:
    """
    k = len(doc["options"])
    choices = ["a", "b", "c", "d", "e"][:k]
    prompt = "Answer the following question about the given passage. Base your answer on the reasoning below.\n\n"
    prompt = "Passage: " + doc["passage"] + "\n\n"
    prompt += "Question: " + doc["question"] + "\n"
    for choice, option in zip(choices, doc["options"]):
        prompt += f"{choice.upper()}. {option}\n"
    prompt += "\n"
    prompt += "Reasoning: " + doc["reasoning"] + "\n\n"    
    prompt += "Answer:"
    return prompt
