import numpy as np
import pandas as pd
import sounddevice as sd
import numpy as np
import scipy.io.wavfile as wav
import wave
from faster_whisper import WhisperModel
import nltk
import sys
import os

def loadStory(filename):
    """
        Loads tory Raw as well as getting read of labels
    """
    f = open('C:\\Users\\barak\\Documents\\GitHub\\strategicMW\\TextGeneration\\GeneratedStories\\' + filename,'r',encoding ='utf-8')
    content = f.readlines()
    newLineCount = 0;
    newContent = [];
    for i in range(len(content)):
        if content[i] != '\n' and content[i] != '#Wstęp\n' and content[i] != '#Janek\n' and content[i] != '#Kasia\n'  and content[i] != '#Karolina\n' and content[i] != '#Koniec\n' :
            newContent.append(content[i])

    return(newContent,content)


def cleanFragment(text):
    """
    tokenize Words in text for Polish usage
    """
    from nltk import word_tokenize
    import re
    # Tokenize Words
    words = word_tokenize(text,language='polish')
    # Get Every Interpucntion Sing
    cleaned_tokens = [re.sub(r'[^\w\s]', '', token) for token in words if re.sub(r'[^\w\s]', '', token)]

    return cleaned_tokens


def splitStoryEntity(cont):
    """
    REMEBMEBR Need to adjust depending if you have an introduction or not
    """
    #introduction = cont[1]
    #ending  = cont[len(cont)-1]
    storyEntitiesOnly = cont[0:len(cont)]

    storyWithoutHeader = [storyEntitiesOnly[i] for i in range(1,len(storyEntitiesOnly),2)]
    storyFirstEntity   = [storyWithoutHeader[i] for i in range(0,len(storyWithoutHeader),2)]
    storySecondEntity  = [storyWithoutHeader[i] for i in range(1,len(storyWithoutHeader),2)]
    #"introd" : introduction, "ending": ending,
    storyBundle = { "firstEntity": storyFirstEntity, "secondEntity" : storySecondEntity}


    return storyBundle


def tagStory(Story1):
    import morfeusz2

    from nltk.tokenize import word_tokenize
    # Initialize the analyzer
    morf = morfeusz2.Morfeusz()

    pos_abbrev = {
        'subst': 'N',          # Noun (rzeczownik)
        'depr': 'N',           # Deprecated noun forms (archaic)
        'adj': 'ADJ',          # Adjective (przymiotnik)
        'adjp' : 'ADJ',       # Adjective verblike (Przzymiotnik odczasownikowy)
        'adv': 'ADV',          # Adverb (przysłówek)
        'num': 'NUM',          # Numeral (liczebnik)

        # Verbs (czasowniki)
        'praet': 'V',          # Past tense verb (czas przeszły)
        'fin': 'V',            # Finite verb (czasownik odmienny przez osoby)
        'bedzie': 'V',         # Future auxiliary form (będzie)
        'impt': 'V',           # Imperative verb (tryb rozkazujący)
        'inf': 'V',            # Infinitive verb (bezokolicznik)
        'pact': 'V',           # Active participle (imiesłów czynny)
        'pant': 'V' ,             # Anticipative participle  Imiesłów artcypacyjny
        'ppas': 'V',           # Passive participle (imiesłów bierny)
        'pcon': 'V',           # Converb/Adverbial participle (imiesłów przysłówkowy współczesny)
        'ger': 'V',            # Gerund (odimiesłowowy rzeczownik)
        'aglt': 'V',           # Agglutinative verb forms (e.g., "byśmy")
        'imps': 'V',            # Impersonal verb
        # Pronouns (zaimki)
        'ppron3': 'PRON',      # Personal pronoun, 3rd person (np. on, jego)
        'ppron12': 'PRON',     # Personal pronoun, 1st/2nd person (ja, ty)
        'siebie': 'PRON',      # Reflexive pronoun (siebie)
        'qub': 'PRON',         # Quasi-pronouns (e.g., "to", "tamto")

        # Prepositions, conjunctions, particles
        'prep': 'PREP',        # Preposition (przyimek)
        'conj': 'CONJ',        # Conjunction (spójnik)
        'comp': 'CONJ',        # Complementizer (np. "żeby")
        'part': 'PART',        # Particle (partykuła, np. "niech", "by")

        # Other categories
        'brev': 'ABBR',        # Abbreviation (skrót)
        'pred': 'ADV',         # Predicative adverbs (np. "trzeba")
        'interj': 'INTERJ',    # Interjection (wykrzyknik, np. "hej!")
        'xxs': 'UNK',          # Unknown or unrecognized word forms
        'xxx': 'SYM',          # Symbols/punctuation
        'interp': 'PUNC',      # Interpunctuation (kropki, przecinki, etc.)
        'ign': 'IGN',           # Ignored segments (e.g., foreign or corrupted text)
        'frag' : 'FRAG',       # Fragment Zdania
        'winien': 'MISC'
    }



    tokens = word_tokenize(" ".join(Story1),language='polish')
    tokens
    SpeechTagged = {};
    for word in tokens:
        if word != '' and word != '\n':
            analysis = morf.analyse(word)
            dd = analysis[0][2][2].split(":")[0]
            print(f"{analysis[0][2][0]} - {pos_abbrev[dd]}")
            if pos_abbrev[dd] not in list(SpeechTagged.keys()):
                SpeechTagged[pos_abbrev[dd]] = [analysis[0][2][0]]
            else:
                SpeechTagged[pos_abbrev[dd]].append(analysis[0][2][0])
    return SpeechTagged



##### COSINE SIMILARITY

def cosineSim(text1,text2):
    """
        Calculates Cosine Similarity between two texts. Can be used as a proxy to assess which kind of text to choose.
    
    """

    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
  
    #Preprocess Words - Tokenize, and remove Stopwords
    tokens1 = preprocess(text1.lower())
    tokens2 = preprocess(text2.lower())

    # Create the TF-IDF vectors
    vectorizer = TfidfVectorizer()


    vector1 = vectorizer.fit_transform(tokens1)
    vector2 = vectorizer.transform(tokens2)

    # Mean Similarity ( between each word)
    similarity = np.sum(sum(cosine_similarity(vector1, vector2)))/np.shape(vector2)[0]
    return similarity


def preprocess(text):
    from nltk.tokenize import word_tokenize
    import os
    """
        Preprocessing, and tokenizing words with NLTK, as well as removing polish stopwords
        Polish stopwords are from external file.
        Then filtering Non-word characters.
    """

    # Tokenizing
    tokens = word_tokenize(text,language='polish')

    #Removing Stopwords
    f = open("c:\\Users\\barak\\Documents\\GitHub\\startegicMW_lite\\TextGeneration\polish.stopwords.txt", "r", encoding='utf-8')
    plstopwords = f.read().split("\n")
    filtered_tokens = [word for word in tokens if word.lower() not in plstopwords]

    # Filtering Non-words
    filtered_tokens = [word for word in filtered_tokens if word.lower() not in [',','.',':',';','?','!']]
    return filtered_tokens



def bertCosineSim(text1,text2,tokenizer,model):
    """
    Computes the average pairwise cosine dissimilarity between token embeddings
    from one text, and the other: 

    Specifically:
    - The input text is tokenized and passed through the BERT model.
    - Token embeddings from layers 6 and 7 are averaged (as these layers are rich in semantic information).
    - Cosine similarity is computed between all non-padding token embeddings.
    - The result is transformed into dissimilarity (1 - cosine similarity).
    - The function returns the average dissimilarity across all token pairs.

    This provides a rough estimate of how semantically coherent the input text is:
    - Low average dissimilarity → tokens are semantically similar → cohesive text
    - High average dissimilarity → tokens diverge semantically → less cohesive or topic-shifting text

    Parameters:
    -----------
    text : str
        A single string of Polish text to analyze.

    Returns:
    --------
    avgSim : List[float]
        A list of average dissimilarity values for each input (typically just one value).
    """
    import torch

    # Load tokenizer and model (official version)

    ### Tokenize without return_offsets_mapping
    tokens1 = tokenizer(text1, return_tensors="pt", padding=True, truncation=True,  max_length=300) # Padding and Truncation allows for joinign multiple Fragments without worry about messups
    tokens2 = tokenizer(text2, return_tensors="pt", padding=True, truncation=True,  max_length=300) # Padding and Truncation allows for joinign multiple Fragments without worry about messups

    ### Get Names of Each token to see to what they refer to.
    #tokenNames = tokenizer.tokenize(text)
    #print("Tokens:", tokenNames)

    ### Decoding Tokens
    #print("Token IDs:", tokens["input_ids"])
    #print("Tokens:", [tokenizer.decode([id]) for id in tokens["input_ids"][0]])
    attention_mask1 = tokens1["attention_mask"].cpu().numpy()  # Shape: [batch_size, sequence_length]
    attention_mask2 = tokens2["attention_mask"].cpu().numpy()  # Shape: [batch_size, sequence_length]

    ### Get All Embeddings and Hidden States
    with torch.no_grad():
        outputs1 = model(**tokens1)
        outputs2 = model(**tokens2)

        # Hidden states for all 12 layers - for All Layers we got a Tensor
    hidden_states1 = outputs1.hidden_states  # Shape: (12 layers, batch_size, sequence_length, hidden_size) 
    hidden_states2 = outputs2.hidden_states  # Shape: (12 layers, batch_size, sequence_length, hidden_size) 

        # Batch Size - IF doing one sentence at a Time We got 1....
        # Sequence_length - Number of Langauge Tokens ([CLS] a begin token, Subwords (via Sub separation) - or Punctuations, [SEP] - end token)
        # hidden_size - what is the length of Embedding vector for each word?

    ### WARNING - BERT Considers a "Context" a single input that you provide it with... it may be a Sentence, but a whole thing BUT NO LONGER THAN 512 Tokens


    ### Layer 6 and 7th Focuses on Semantic Information Mostly: et those
    layer11 = hidden_states1[6]
    layer12 = hidden_states1[7]

    layer21 = hidden_states2[6]
    layer22 = hidden_states2[7]

    ### Option 1st: Avearaging Stories across Cosine Sim: 
    from sklearn.metrics.pairwise import cosine_similarity
    cosSim = [];
    avgSim = [];



    for i in range(layer11.shape[0]):

        # Average layers 6 and 7 for this fragment
        combined_embeddings1 = (layer11[i] + layer12[i]) / 2  # Shape: [sequence_length, hidden_size]
        combined_embeddings2 = (layer21[i] + layer22[i]) / 2  # Shape: [sequence_length, hidden_size]

        # Mask to select only non-padded embeddings
        non_padded_embeddings1 = combined_embeddings1[attention_mask1[i] == 1]  # Shape: [actual_tokens, hidden_size]
        non_padded_embeddings2 = combined_embeddings2[attention_mask2[i] == 1]  # Shape: [actual_tokens, hidden_size]

        if non_padded_embeddings1.shape[0] > 1 and  non_padded_embeddings2.shape[0] > 1 :  # Ensure there are enough tokens to compute pairwise distances
            cosSim.append(cosine_similarity(non_padded_embeddings1,non_padded_embeddings2))  # Shape: [actual_tokens, actual_tokens]

            np.fill_diagonal(cosSim[i], 0)

            avgSim.append(np.sum(cosSim[i])/(cosSim[i].shape[0] * (cosSim[i].shape[0] - 1)))

    return avgSim



###### # STORY EXTRACTION TOOLS AND ALL: #############


#### Get the Original Story Data: 
def cluster_consecutive_sum(df, entity_col, score_col):
    """
        Concatenates Crucial Information and their score Fragment-Wise 
        Input:
        + df (memory score dataframe)
        + entity_col - which etnity is currently concatenated
        + score_col - what score will be concatentated fragment wise 

        Output: DataFrame of 40 rows correspondign to scores in fragments: 

    """
    results = []
    if df.empty:
        return pd.DataFrame(columns=['Part', 'Hero', 'Joined_Memory_Score'])
    current_entity = df.iloc[0][entity_col]
    current_sum = df.iloc[0][score_col]
    part = 1
    for i in range(1, len(df)):
        row = df.iloc[i]
        if row[entity_col] == current_entity:
            current_sum += row[score_col]
        else:
            results.append({'Part': part, 'Hero': current_entity, 'Joined_Memory_Score': current_sum})
            part += 1
            current_entity = row[entity_col]
            current_sum = row[score_col]
    # Add the last group
    results.append({'Part': part, 'Hero': current_entity, 'Joined_Memory_Score': current_sum})
    return pd.DataFrame(results)



def createTranscribedSimilarity(path,savePath,subjects,Story1,janekStory,karolinaStory,makeTranscriptions=0,device = "cuda"):
    janekScore    = [];
    karolinaScore = [];
    allScore      = [];
    similarity    = []
    allTexts = []
    if makeTranscriptions:
        nltk.download('punkt_tab')
        lineCount = 0;

        for i, subj in enumerate(subjects):
           
            if os.path.isfile(os.path.join(savePath,subj,'recall.txt')):
                continue
            currentSubj = os.path.join(path,subj, 'Recalls', 'story1.wav')
            print(f"Processing subject {i+1}/{len(subjects)}: {subj}")
            # Open wave file
            wf = wave.open(currentSubj, "rb")

            # Check audio format
            if wf.getnchannels() != 1 or wf.getsampwidth() != 2 or wf.getcomptype() != "NONE":
                print("Audio file must be WAV format mono PCM.")
                sys.exit(1)

            # Load Whisper model (can use "base", "small", "medium", etc.)
            model = WhisperModel("medium", device=device)  # or "cuda" if you have a GPU

            segments, _ = model.transcribe(currentSubj, language="pl", beam_size=20) # włąściwy 
            originalText = [s.text for s in segments]
            allTexts.append(originalText)
            print("You said: ", originalText)

            # Save Recall Text: C
            with open(os.path.join(savePath,subj,'recall.txt'), "w",encoding="utf-8") as text_file:
                text_file.write(" ".join(originalText))

    else:

        lineCount = 0;
        for i, subj in enumerate(subjects):
            text_file =  open(os.path.join(savePath,subj,'recall.txt'), "r+",encoding="utf-8")
            for line in text_file: # One Text - One Line - thats why it worrks although it looks INCREDIBELY SKETCHY
                allTexts.append(line)

            ### Get Similarity Indexes
            janekScore.append(cosineSim(janekStory,line)) # Similarity between Janek's Recollection and the Line
            karolinaScore.append(cosineSim(karolinaStory,line)) 
            allScore.append(cosineSim("".join(Story1),line))

            for j,fragment in enumerate(Story1): # For Each Fragment of the story
                lineCount +=1;
                similarity.append(cosineSim(line,fragment))

        return similarity,allScore,janekScore,karolinaScore,allTexts