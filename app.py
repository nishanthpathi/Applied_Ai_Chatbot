
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
    input_ids = tokenizer.encode(question, answer_text, max_length=512)

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

bert_abstract = " Applied AI Course is a complete online course, validity of the Applied AI Course is 365 days. A mentor is allocated to you once you complete 15 mandatory assignments. You will get a certificate once you complete all course assignments please check https://www.appliedaicourse.com/certificate/8fa2c6377d to find the sample certificate. The Applied AI course provides a good overview of most of the popular ML and Deep Learning algorithms. Applied AI course is also constantly evolving as machine learning that is Syllabus  is not fixed. The Applied AI Course keeps adding the new contents with time. The Applied AI Course conducts live sessions every week. laptop/system requirements with around 8Gb of RAM minimum and i7 processor with 4G graphics card from NVIDIA. laptop/system requirements to run deep learning programs are 1050Ti and above is preferable. You can also look into Google Colab for running the code. Average package depends on different factors like the candidate's educational background, skillset the candidate possesses, type of the company the candidate is trying for, type of projects/portfolio the candidate has worked on call us on 91-8106-920-029. Mail us to team@appliedaicourse.com . No prerequisites are required, all the concepts discussed have been intuited from a fundamental level to an advanced level. text mining/NLP(Natural Language Processing) covers simple Bag-Of-Words, TF-IDF to more complex Word2Vec and LSTM networks which are state-of-the-art in text mining. We are providing a job guarantee program in India and the USA. The only reason behind not providing this job guarantee program in other countries is that we don't have relationships with the companies in those countries. Notes are not provided to students. Students are expected to prepare their own notes so that they can retain more and prepare notes as per their way of learning. For implementing the projects we use sklearn for ML models and tf.Keras for DL models. For understanding tf-IDF vectorizer, BOW vectorizer, Bootstrapping, Gradient Descent algorithm we need to write code from scratch. Difficulty level for assignment are if you understand the theory well it is very easy to follow the step by step instructions given under each assignment. Course content of Applied AI course has 150+hours of industry focused and extremely simplified content it also includes applied aspects of artificial intelligence. we respond to your queries regarding any topic with in twenty four hours."

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