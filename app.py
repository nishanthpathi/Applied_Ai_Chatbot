
import streamlit as st
import torch
from transformers import BertForQuestionAnswering
from transformers import BertTokenizer
import textwrap

st.title("Applied-AI-Chatbot")
st.text('Ask me any "Basic" question related to Applied-AI-Course at Applied-Courses you have in the below box')
tokenizer = BertTokenizer.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')


@st.cache(allow_output_mutation=True)
def load_model():
    model = BertForQuestionAnswering.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')
    return model




    

def answer_question(question, answer_text):
    '''
    Takes a `question` string and an `answer_text` string (which contains the
    answer), and identifies the words within the `answer_text` that are the
    answer. Prints them out.
    '''
    # ======== Tokenize ========
    # Apply the tokenizer to the input text, treating them as a text-pair.
    input_ids = tokenizer.encode(question, answer_text)

    # Report how long the input sequence is.
    print('Query has {:,} tokens.\n'.format(len(input_ids)))

    # ======== Set Segment IDs ========
    # Search the input_ids for the first instance of the `[SEP]` token.
    sep_index = input_ids.index(tokenizer.sep_token_id)

    # The number of segment A tokens includes the [SEP] token istelf.
    num_seg_a = sep_index + 1

    # The remainder are segment B.
    num_seg_b = len(input_ids) - num_seg_a

    # Construct the list of 0s and 1s.
    segment_ids = [0]*num_seg_a + [1]*num_seg_b

    # There should be a segment_id for every input token.
    assert len(segment_ids) == len(input_ids)

    # ======== Evaluate ========
    # Run our example question through the model.
    start_scores, end_scores = model(torch.tensor([input_ids]), # The tokens representing our input text.
                                    token_type_ids=torch.tensor([segment_ids])) # The segment IDs to differentiate question from answer_text

    # ======== Reconstruct Answer ========
    # Find the tokens with the highest `start` and `end` scores.
    answer_start = torch.argmax(start_scores)
    answer_end = torch.argmax(end_scores)

    # Get the string versions of the input tokens.
    tokens = tokenizer.convert_ids_to_tokens(input_ids)

    # Start with the first token.
    answer = tokens[answer_start]

    # Select the remaining answer tokens and join them with whitespace.
    for i in range(answer_start + 1, answer_end + 1):
        
        # If it's a subword token, then recombine it with the previous token.
        if tokens[i][0:2] == '##':
            answer += tokens[i][2:]
        
        # Otherwise, add a space then the token.
        else:
            answer += ' ' + tokens[i]
    st.write(answer)
    #print('Answer: "' + answer +'"')
    #return answer
    
    

# Wrap text to 80 characters.
wrapper = textwrap.TextWrapper(width=80)
    
    
#sprint(wrapper.fill(bert_abstract))

bert_abstract = " Applied AI is a online course to learn practical machine learning concepts with job guarantee program,if you complete all the assignments we assist you for job gaurantee program,we conduct mock interviews aswell for job gaurantee program, Validity of the Course is one year, Fee for the Course is twenty five thousand rupees only, You will get a Certificate once you complete the Course, i3 Processor with 8Gb RAM atleast is sufficient to complete the Course, We respond to your queries regarding any topic with in twenty four hours, feel free to drop us a mail at team@appliedaicourse.com, We conduct Live-Session based on student's demand every weekend, We do have EMI facility for Fee"

question = st.text_input('Ask your question here:')

model = load_model()

if question:
    answer_question(question, bert_abstract)

#question = st.text_input('Input your question here:') 
#if st.button("Go"):
#answer_question(question, bert_abstract)
    #st.write(answer)

#question = "what are system requirements"

#answer_question(question, bert_abstract)