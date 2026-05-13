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
import language_tool_python,morfeusz2
import audioop          # <-- add at top of file (built-in, no extra install)

from bertopic import BERTopic
#from bertopic._bertopic import SeededBERTopic

from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import CountVectorizer
import hdbscan, umap
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import random
import numpy as np
from sentence_transformers import SentenceTransformer,util
import torch
from nltk.tokenize import word_tokenize
from transformers import AutoTokenizer, AutoModelForMaskedLM
import math
from transformers import AutoTokenizer, AutoModelForCausalLM

import networkx as nx

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




def reLabel(dataDf,renameDict={"Focus":"Focus","Task-Related Thoughts":"Focus","Mind Wandering":"Mind Wandering","Mind Blanking":"Mind Wandering"}):
    """
        This function Renames some Labels to some other
    """
    
    results = dataDf.copy()
    for k,v in renameDict.items():
        results.loc[results['Attention'] == v,'Attention'] = k

    
    return results

#####################################
##### TEXT ASSESSMENT FUNCTIONS #####
#####################################


def licz_sylaby(slowo):
    import re
    """Count syllables in Polish by correctly handling vowel clusters and glides."""
    
    # Polish vowels and semivowels/glides
    samogloski = "aąeęioóuy"
    glidy = "łw"  # Glides to watch out for: 
                 # Glides are something that "glides" thorugh language so consonatns taht are a tad bit different
                 # they Glide between vowelas NOT starting a new SYllable
    # Step 1: Identify potential syllable splits using consonant-vowel structure
    # So we check wehther there is a Consonant BEFORE the vowel '[^aeiouyąęó]*, then we match the vowerl 1 or more
    # and lastly we get any consonant WITHOUT glides 
    wzorzec_sylab = re.findall(r'[^aeiouyąęó]*[aeiouyąęó]+[^aeiouyąęółw]*', slowo.lower())
    
    # Step 2: Merge where necessary (e.g., handle diphthongs and glides)
    liczba_sylab = 0
    poprzednia_sylaba = ""

    for sylaba in wzorzec_sylab:
        # If the previous syllable ends with a glide, merge it
        if poprzednia_sylaba and poprzednia_sylaba[-1] in glidy:
            liczba_sylab -= 1  # Merge this with the previous syllable
        liczba_sylab += 1
        poprzednia_sylaba = sylaba

    return liczba_sylab

def FOGScore(text,morf):
    """ Computing Gunning-Fog Index. It is an estimation of Readability:
    
    0.4 * number of words/ number of sentences + 100* number of Words with more than 3 syllables / number of words 
    """
    from re import split,sub
    from nltk import word_tokenize
    globalSylableCount = 0
    text = text.replace('\n',' ') # Remove new Lines

    # Get sentences (by splitting by . !? )
    sentences = split(r'[.!?]', text)
    sentences = [sentence.strip() for sentence in sentences if sentence] # strip empty sentences

    # get Words by splitting through " ", with removing of empty sequences
    #words = word_tokenize(text,language='polish')
    #words = [sub(r'[^\w\s]', '', token) for token in words if sub(r'[^\w\s]', '', token)]
    words = dictStemmer(text,morf)
    # Ger Syllable count
    for i in words:
        if licz_sylaby(i) >3:
            globalSylableCount += 1
    return 0.4*(len(words)/len(sentences)) + 100*globalSylableCount/len(words)


def lexical_diversity(tokens):
    """
        Calculates Lexical Diveristy as a number of Unique Tokens divided by general number of Tokens
    """
    return len(set(tokens)) / len(tokens) if len(tokens) > 0 else 0


def lexical_diversity_storywise(tokens,storyTokens):
    """
        Calculates Lexical Diveristy as a number of Unique Tokens divided by general number of Tokens
    """
    return len(set(tokens)) / len(set(storyTokens)) if len(tokens) > 0 else 0


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

    # Create the TF-IDF vectors
    vectorizer = TfidfVectorizer()


    vector1 = vectorizer.fit_transform([text1])
    vector2 = vectorizer.transform([text2])

    # Mean Similarity ( between each word)
    similarity = np.sum(sum(cosine_similarity(vector1, vector2)))/np.shape(vector2)[0]
    return similarity


preNorm = {
    "Janka": "Janek",
    "Jankowi" : "Janek",
    "Janku": "Janek",
    "Jana":  "Janek",
    "Jan":   "Janek",
    "Karoliny":  "Karolina",
    "Karolinie": "Karolina",
}

lems= {
    "trocha": "trochę",
    "troszka": "trochę",
    "myślić": "myśli",
    "oczyć":  "oczy",
    "swoić":  "swój",
    "tyli" : "tyle",
    "pewne": "pewny",
    "dzienić" : "dzień",
    "miecić" : "mieć",
    "Janko" : "Janek",
    "słowić": "słowo",
    "spokoić": "spokój",
    "wzorzać": "wzór",
    "stola": "stół",
    "ciszyć": "cisza",
    "wieczor": "wieczór",
    "potrzeb": "potrzeba",
    "got" :"gotowy",
    "ciężeć" : "ciężko",
    "odbiegły": "odbiegać",
    "rozprószać" : "rozproszyć",
    "rozprószyć" : "rozproszyć"

}


def dictStemmer(text,morf,norms= None,lems=None):

    """
        This function stemms the current text using Morpheus:

    """

    tokens = preprocess(text)
    if norms !=  None:
        tokens = [norms.get(t, t) for t in tokens]


    stemmed = []
    for tok in tokens:
        analysis = morf.analyse(tok)
        if analysis:
            # if len([i[2][1] for i in morf.analyse(tok) if 'ć' in i[2][1]]):  # Bierz pierwszy lemma z listy analiz JEŚLI nie, to pierwszy Bezokolicznik Jak Leci 
            #     lemma = [i[2][1] for i in morf.analyse(tok) if 'ć' in i[2][1]][0]
            # else:
            lemma = analysis[0][2][1]            

            if ':' in lemma: #Brak dodatkowych form: odcięcie na podstawie ';'
                lemma = lemma[:lemma.index(':')]


            #### Manual Adjustements
            if lems !=  None:
                lemma = lems.get(lemma, lemma)
            
            if analysis[0][2][2] != 'ign': # Jeśli coś nei jest rozpoznane jako słowo, jest IGNOROWANE
                stemmed.append(lemma)

    return stemmed


def preprocess(text):
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
    filtered_tokens = [word for word in filtered_tokens if word.lower() not in [',','.',':',';','?','!',')','(','...','„','”','—','-']]
    return filtered_tokens


def bertCosineSim(text1, text2, tokenizer, model):
    """
    Compute semantic similarity between two texts using BERT hidden states,
    with minimal changes to your original structure.

    KEY FIXES vs original:
    - Remove special tokens explicitly (attention_mask==1 still includes specials).
    - Use L2-normalized token embeddings so cosine has a proper scale.
    - Use a *greedy alignment* between token sets instead of all-vs-all averaging:
        For each token in A, take max cosine to any token in B (and vice versa),
        then average the two directions. This reduces length bias a lot.
    - Keep your layer choice (6 & 7), but you can switch to "last4" by editing below.

    Returns:
        avgSim: list[float]  # one scalar similarity per batch item (you usually have 1)
                              # in [-1, 1], higher = more similar
    """
    import torch
    import numpy as np
    from sklearn.metrics.pairwise import cosine_similarity  # kept to minimize changes, but we won't use the diagonal trick

    # --- 1) Tokenize (same as you had) --------------------------------------
    tokens1 = tokenizer(text1, return_tensors="pt", padding=True, truncation=True, max_length=300)
    tokens2 = tokenizer(text2, return_tensors="pt", padding=True, truncation=True, max_length=300)

    # Attention masks as tensors (we'll still keep your numpy version if you like)
    attention_mask1 = tokens1["attention_mask"]  # [1, L1]
    attention_mask2 = tokens2["attention_mask"]  # [1, L2]

    # --- 2) Forward pass with hidden states ---------------------------------
    # IMPORTANT: ensure hidden states are returned even if model wasn't created with output_hidden_states=True
    with torch.no_grad():
        outputs1 = model(**tokens1, output_hidden_states=True)
        outputs2 = model(**tokens2, output_hidden_states=True)

    hidden_states1 = outputs1.hidden_states  # tuple: [embeddings, layer1, ..., layerN]
    hidden_states2 = outputs2.hidden_states

    # --- 3) Pick layers and combine (you used 6 & 7; that's fine) ----------
    # Reminder: hidden_states[0] is the embedding layer, so [6] and [7] are transformer blocks 6 and 7.
    layer11 = hidden_states1[6]  # [batch, L1, H]
    layer12 = hidden_states1[7]  # [batch, L1, H]
    layer21 = hidden_states2[6]  # [batch, L2, H]
    layer22 = hidden_states2[7]  # [batch, L2, H]

    # Option: uncomment to use "last4" which often works even better:
    # layer11 = layer12 = (hidden_states1[-1] + hidden_states1[-2] + hidden_states1[-3] + hidden_states1[-4]) / 4.0
    # layer21 = layer22 = (hidden_states2[-1] + hidden_states2[-2] + hidden_states2[-3] + hidden_states2[-4]) / 4.0

    # --- 4) Build outputs in the same outer structure you had ---------------
    cosSim = []   # we'll store the full similarity matrices here if you want to inspect
    avgSim = []   # final scalar similarity per item (we keep the list to match your return type)

    # Special-token ids to drop (CLS/SEP/PAD/MASK, etc.)
    SPECIAL = set(tokenizer.all_special_ids)

    # Loop over batch items (you typically have batch size 1)
    for i in range(layer11.shape[0]):

        # Average the two chosen layers (your approach)
        combined_embeddings1 = (layer11[i] + layer12[i]) / 2.0   # [L1, H]
        combined_embeddings2 = (layer21[i] + layer22[i]) / 2.0   # [L2, H]

        # --- 5) Build masks: keep real tokens, drop specials ----------------
        ids1 = tokens1["input_ids"][i]           # [L1]
        ids2 = tokens2["input_ids"][i]           # [L2]

        # padding mask (1 = real token)
        pad_mask1 = attention_mask1[i].bool()    # [L1]
        pad_mask2 = attention_mask2[i].bool()    # [L2]

        # special-token mask (False for specials)
        spec_mask1 = torch.tensor([tid.item() not in SPECIAL for tid in ids1], dtype=torch.bool)
        spec_mask2 = torch.tensor([tid.item() not in SPECIAL for tid in ids2], dtype=torch.bool)

        # final keep mask = real token AND not special
        keep1 = (pad_mask1 & spec_mask1)
        keep2 = (pad_mask2 & spec_mask2)

        E1 = combined_embeddings1[keep1]         # [n1, H]
        E2 = combined_embeddings2[keep2]         # [n2, H]

        # If you want to also drop stopwords, do it HERE by inspecting tokenizer.convert_ids_to_tokens(ids1[keep1]) etc.

        # Guard: if nothing left, append NaN and continue
        if E1.shape[0] == 0 or E2.shape[0] == 0:
            avgSim.append(float("nan"))
            # also push an empty matrix placeholder to keep structure similar
            cosSim.append(np.zeros((E1.shape[0], E2.shape[0])))
            continue

        # --- 6) L2-normalize token vectors (crucial for cosine stability) ---
        E1 = torch.nn.functional.normalize(E1, p=2, dim=-1)  # [n1, H]
        E2 = torch.nn.functional.normalize(E2, p=2, dim=-1)  # [n2, H]

        # --- 7) Cosine similarity matrix between tokens ---------------------
        # (Because rows are normalized, dot = cosine.)
        S = (E1 @ E2.T).cpu().numpy()            # [n1, n2]
        cosSim.append(S)                         # keep for debugging/inspection

        # --- 8) Greedy alignment score (A->B and B->A), NOT all-pairs mean --
        # This fixes the length bias from naive averaging.
        # For each token in text1, find its best match in text2; average these maxima.
        a_to_b = S.max(axis=1).mean()
        # For each token in text2, find its best match in text1; average these maxima.
        b_to_a = S.max(axis=0).mean()
        sim = float((a_to_b + b_to_a) / 2.0)     # final symmetric similarity in [-1, 1]

        # If you truly want "dissimilarity", use: sim = 1.0 - sim   (but then it's not bounded well)
        avgSim.append(sim)

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


def quickLangCorrection(originalText,mode=0):
    """
        Used for quick Language Correction by either using Langauge Tool or Morfeusz removal
        default (0) - Morfeusz
    """

    if mode:
        correctedText = originalText
        morf = morfeusz2.Morfeusz()
        suspicious_tags = {"ign", "xxx"}
        analysis = morf.analyse(originalText)
        wrong_utterances = []
        for i,(start, end, interp) in enumerate(analysis):
            form, lemma, tags,_,_ = interp
            if tags in suspicious_tags:
                print(f"     Warning! removing wrongly transcribed word: {form}")
                correctedText = correctedText.replace(form,'')
    else:
        tool = language_tool_python.LanguageTool('pl')
        matches = tool.check(originalText)
        correctedText = language_tool_python.utils.correct(originalText, matches)

    return correctedText

def transcibeText(path,savePath,subjects,device="cpu",langCorr = 0):
    """
        This function transcribes a story Recalls in a path to a Text 

        
        Arguments:
        - path - has a path to recalls
        - savePath - path where we want to save our recalls
        - subjects - folder names in which my recalls are located
        - device - whether to transcirbe using CUDA or CPU 
        - langCorr - Whether to correct language after transcrption
            - opt 1: Using Morfeusz - polish syntax parser- to establish closest sounding word
            - opt 2: Using Language_tool_python language corrector for polish language 
    
    """
    nltk.download('punkt_tab')
     # Load Whisper model (can use "base", "small", "medium", etc.)
    model = WhisperModel("medium", device=device)  # or "cuda" if you have a GPU


    ### Select Language Correction at hand 
    if langCorr == 2:
        tool = language_tool_python.LanguageTool('pl')
    elif  langCorr == 1:
        morf = morfeusz2.Morfeusz()
        suspicious_tags = {"ign", "xxx"}


    for i, subj in enumerate(subjects):

        if os.path.isdir(os.path.join(savePath,subj)) == 0:
            os.makedirs(os.path.join(savePath,subj)) # Create a directory for a subject in transcripotion Path

        currentSubj = os.path.join(path,subj, 'Recalls') # Folder with Curren Subject Recalls
        print(f"== Processing subject {i+1}/{len(subjects)}: {subj}")

        # Get all wave files in the current subject's recall folder
        wave_files = [f for f in os.listdir(currentSubj) if f.endswith('.wav')]

        for f in wave_files:
            print(f"      processing file: {f}")

            ### -- Load Audio and Check it (If it is even Playing )
            wf = wave.open(os.path.join(currentSubj,f), "rb")             # Open wave file
            if wf.getnchannels() != 1 or wf.getsampwidth() != 2 or wf.getcomptype() != "NONE":             # Check audio format
                print("Audio file must be WAV format mono PCM.")
                sys.exit(1)

            wf.rewind()                          # make sure we start at the beginning
            rms = audioop.rms(wf.readframes(wf.getnframes()) , wf.getsampwidth())            # audioop.rms returns root-mean-square amplitude for the whole buffer
            SILENCE_THRESHOLD = 20   # adjust to taste (0..32767 for 16-bit audio)
            if rms < SILENCE_THRESHOLD:
                print("          File seems silent – skipping transcription.")
                continue              # jump to the next .wav

            if os.path.isfile(os.path.join(savePath,subj,f'{os.path.splitext(f)[0]}.txt')):
                print("          File already transcribed")

                continue # Continue if the  text has been alread#y transcribed 


            ### -- Transcribe Audio 
            segments, _ = model.transcribe(os.path.join(currentSubj,f), language="pl", beam_size=20) # Our Translation
            originalText =  " ".join([s.text for s in segments])


            ### -- Language Correction (Optional) -- 
            if langCorr   == 1: #Remove Words that are deemed Unclear by Morfeusz
                correctedText = originalText;
                analysis = morf.analyse(originalText)
                wrong_utterances = []
                for i,(start, end, interp) in enumerate(analysis):
                    form, lemma, tags,_,_ = interp
                    if tags in suspicious_tags:
                        print(f"     Warning! removing wrongly transcribed word: {form}")
                        correctedText = correctedText.replace(form,'')

            elif langCorr == 2: # Language correction with language-tools-python
                matches = tool.check(originalText)
                correctedText = language_tool_python.utils.correct(originalText, matches)

            else:               # No Language Correction at all! 
                correctedText = originalText;


            ### -- Save Recall Text: C
            with open(os.path.join(savePath,subj,f'{os.path.splitext(f)[0]}.txt'), "w",encoding="utf-8") as text_file:
                text_file.write("".join(correctedText))
            print("         Transcription: ", correctedText)

def createTranscribedSimilarity(path,savePath,subjects,Story1,janekStory,karolinaStory,device = "cuda"):
    janekScore    = [];
    karolinaScore = [];
    allScore      = [];
    similarity    = []
    allTexts = []
   
    lineCount = 0;
    for i, subj in enumerate(subjects):
        text_file =  open(os.path.join(savePath,subj,'story1.txt'), "r+",encoding="utf-8")
        for line in text_file: # One Text - One Line: Always! - thats why it worrks although it looks INCREDIBELY SKETCHY
            allTexts.append(line)

        ### Get Similarity Indexes
        janekScore.append(cosineSim(janekStory,line))        # Similarity between Janek's Recollection and the Line
        karolinaScore.append(cosineSim(karolinaStory,line))  # SImilartiy between Karolina's Recollection and the line
        allScore.append(cosineSim("".join(Story1),line))     # Similarity between All Story Scores 

        for j,fragment in enumerate(Story1): # Similarities  For Each Fragment of the story
            lineCount +=1;
            similarity.append(cosineSim(line,fragment))

    return similarity,allScore,janekScore,karolinaScore,allTexts


###########################################
##### MORPHOLOGICAL ANALYSIS FUNCTIONS ####
###########################################

def spacyMorph(text,NLP,morf,polish_stopwords,verbosity=0):

    """
        This function uses either Spacy or Morfeusz based stemming and lemmatisation to get
        Syntactic and Semantic Features of current text 
    
    """


    def is_participle_adj(tok):
        # Detect adjectival participles like "wchodzący/wchodzącą".
        # In spaCy (UD), they are ADJ tokens with morph feature VerbForm=Part.
        return tok.pos_ == "ADJ" and tok.morph.get("VerbForm") == ["Part"]

    doc = NLP(text)  # spaCy builds tokens, POS, lemmas, morphology, and dependency parse.

    """
    What spaCy provides per token:
    1) token.pos_   → Coarse POS tag (NOUN, VERB, ADJ, ADV, ADP, PROPN, ...)
    2) token.morph  → Morphological features (e.g., VerbForm=Part, Case=Acc, Number=Pl)
    3) token.lemma_ → Lemma (base form)
    4) token.dep_   → Dependency relation to its head (obj, obl, nsubj, amod, case, ...)
    token.lefts  → Children of the token that appear BEFORE it in the text (left of the head)
    token.rights → Children of the token that appear AFTER it in the text (right of the head)
    """

    noun_phrases = []
    verb_phrases = []

    ### TEMP RECONFIGURE!!!

    lemmas       = dictStemmer(text,morf,norms=preNorm,lems=lems) # === Raw Lemmas Extraction === - based SOLELY on MORFEUSZ

    sentences   = [s.text for s in doc.sents]  # Get sentences from the document
    #lemmas = [token.lemma_ for token in doc if not token.is_punct and not token.is_space and token.text not in polish_stopwords]
    for token in doc:
        # Getting Verb/Noun with spacy 

        


        # Debug line: shows surface form, lemma, POS, fine-grained tag, and dependency label.
        if verbosity:
            print(f"{token.text:15} | {token.lemma_:15} | {token.pos_} | {token.tag_}  | {token.dep_}")
        # === Noun Phrase Extraction ===
        """
        Goal: Build a compact noun phrase around the NOUN/PROPN head.

        How this block works:
        • We treat 'token' as the head noun (NOUN/PROPN).
        • LEFT modifiers: we take children in token.lefts (i.e., dependents that occur before the head)
        and keep only ADJ/NUM/DET (and also adjectival participles if you use is_participle_adj).
        → Example: 'tę', 'starą', 'wchodzącą' placed before the noun.
        • RIGHT tail: we scan token.rights and, in this simple version, if we see an ADP (preposition),
        we attach that preposition and its right dependents (NOUN/PROPN/ADJ) as a mini PP.
        → Example: 'pracownię ... przy porcie'

        Note (Polish UD nuance, FYI only): in UD-Polish the preposition (ADP) is typically a 'case'
        child of the NOUN that anchors the PP, not the head. This simple "look for ADP on the right"
        can miss some cases. It's fine for a lightweight extractor; just be aware it's a heuristic.
        """
        if token.pos_ in ("NOUN", "PROPN"):
            # LEFT modifiers = children that appear before the noun in the text.
            # (These are NOT arbitrary words on the left, but actual dependents in the parse tree.)
            modifiers_left = [
                t.text for t in token.lefts  # children of 'token' that occur before it
                if t.pos_ in ("ADJ", "NUM", "DET") or is_participle_adj(t)
            ]

            # RIGHT modifiers (very simple PP heuristic):
            # Look to the right for a preposition token (ADP) that depends on this noun
            # and then grab its right dependents (NOUN/PROPN/ADJ) to form a short PP.
            modifiers_right = []
            for t in token.rights:              # children of 'token' that occur after it
                if t.pos_ == "ADP":             # preposition to start a PP
                    pp = [t.text]               # include the preposition itself
                    pp += [child.text for child in t.rights if child.pos_ in ("NOUN", "PROPN", "ADJ")]
                    modifiers_right += pp

            # Compose the noun phrase: [left modifiers] + head noun + [right PP tail]
            phrase = " ".join(modifiers_left + [token.text] + modifiers_right)
            if phrase.strip():
                noun_phrases.append(phrase)

        # === Verb Phrase Extraction ===
        """
        Goal: Build a short predicate phrase around a VERB.

        Steps:
        • Adverbs (advmod) first → stylistically placed before the verb: 'szybko wyobraziłem'
        • Verb itself (surface form; you could switch to lemma if you prefer canonical form)
        • Reflexive marker (expl) → adds 'się' when present: 'wyobraziłem sobie' / 'skupiłem się'
        • Complements:
            - obj / iobj  → direct/indirect objects (e.g., 'słyszał historię')
            - obl         → oblique dependents (often prepositional phrases; e.g., 'była przy porcie')
        In this minimal version we take the dependent node text and then append its children that are
        NOUN/PROPN/ADJ to make the complement a bit fuller.

        Note: This is intentionally simple. It won't always reconstruct the full PP ('przy porcie')
        if the preposition is attached differently in the parse, but it's a good lightweight heuristic.
        """
        if token.pos_ == "VERB":
            parts = []

            # 1) Adverbs (advmod) placed before the verb for readability
            advs = [t.text for t in token.children if t.dep_ == "advmod"]
            parts += advs

            # 2) Verb itself (surface form; change to token.lemma_ if you want canonical action names)
            parts.append(token.text)

            # 3) Reflexive marker 'się' (expl)
            parts += [t.text for t in token.children if t.dep_ == "expl"]

            # 4) Complements: objects and obliques (very lightweight expansion)
            for t in token.children:
                if t.dep_ in ("obj", "obl"):
                    parts.append(t.text)  # head of the complement
                    # attach its immediate nominal/adj dependents for a compact phrase
                    parts += [child.text for child in t.children if child.pos_ in ("NOUN", "PROPN", "ADJ")]
                    

            # Join the pieces into a verb phrase
            phrase = " ".join(parts)
            if phrase.strip():
                verb_phrases.append(phrase)

        # === Participles as mini-verbs (wdowa wchodząca) ===
        """
        Idea: An adjectival participle (ADJ with VerbForm=Part) often encodes an action attached to a noun.
        We glue the head noun + the participle, and then collect the participle's own complements.

        • head_noun + participle  → 'wdowę wchodzącą'
        • children of the participle:
            - obj / obl → add short complements like 'z dziećmi', 'do środka'
        """
        if is_participle_adj(token):
            head_noun = token.head.text if token.head.pos_ in ("NOUN", "PROPN") else ""
            parts = [head_noun, token.text]
            for t in token.children:
                if t.dep_ in ("obj", "obl"):
                    parts.append(t.text)  # head of the complement
                    # attach immediate nominal/adj dependents for a compact complement
                    parts += [child.text for child in t.children if child.pos_ in ("NOUN", "PROPN", "ADJ")]
            phrase = " ".join(p for p in parts if p)
            if phrase.strip():
                verb_phrases.append(phrase)

    # Results collected in:
    #   noun_phrases → compact entity concepts (e.g., "pracownię przy porcie", "wdowę wchodzącą z dziećmi")
    #   verb_phrases → compact action concepts (e.g., "Wyobraziłem sobie pracownię", "słyszał historię")
    return verb_phrases,noun_phrases,lemmas,sentences


def lexical_diversity(tokens):
    """
        Calculates Lexical Diveristy as a number of Unique Tokens divided by general number of Tokens
    """
    return len(set(tokens)) / len(tokens) if len(tokens) > 0 else 0


def lexical_diversity_storywise(tokens,storyTokens):
    """
        Calculates Lexical Diveristy as a number of Unique Tokens divided by general number of Tokens
    """
    return len(set(tokens)) / len(set(storyTokens)) if len(tokens) > 0 else 0





##################################
#### BERT Similarity Analysis ####
##################################

def bertSentenceCosineSim(text1, text2, model,aggregation="mean_pairs",mode=1):
    """
    Quick BertCosine  Similarity Catered towards Sentence Embeddings. But Works also for Lemmas 
    (providing they have been properly filtered)

    It is Not Fancy at all, i
    """
    ### Sanity: Remove empty tokens:
    text1 = [t for t in text1 if t != " " and t != ' ']
    text2 = [t for t in text2 if t != " " and t != ' ']
    # print(text1)
    # --- encode all sentences in one batch ---
    texts = text1 + text2 # No Worrries since it is a Batch Mode so there is no corss-sentence interactions!

    if mode ==0: ### SEntence BERT Type Model
        # normalize_embeddings=True -> L2-normalized vectors; dot == cosine
        E = model.encode(texts, convert_to_tensor=True, normalize_embeddings=True)  # [F+U, D]
    elif mode == 1:    ### Fast-text Type Embedding: (Warning! Usefull ONLY for Single Word Lemmas)
        E =  [model.wv[lemma] for lemma in texts]

    elif mode == 2: ### GloVe embeddings
        vecs = []
        for token in texts:
            if token in model:
                vecs.append(model[token])
            else:
                continue
        if not vecs:
            return float("nan")
        E = torch.tensor(np.vstack(vecs), dtype=torch.float32)
        E = torch.nn.functional.normalize(E, p=2, dim=-1)

    else:
        None
        
    F = E[:len(text1)]      # [F, D]
    U = E[len(text1):]      # [U, D]
    # ---- Test Diffetent Aggregation Methods ----
    if aggregation == "centroid":
        # average sentence embeddings on each side (then re-normalize)
        f_centroid = torch.nn.functional.normalize(F.mean(dim=0, keepdim=True), p=2, dim=-1)  # [1, D]
        u_centroid = torch.nn.functional.normalize(U.mean(dim=0, keepdim=True), p=2, dim=-1)  # [1, D]
        return float((u_centroid @ f_centroid.T).item())

        # otherwise we need the full pairwise cosine matrix
    S = util.cos_sim(U, F)  # [U, F], tensor
    if S.numel() == 0:
        return float("nan")

    if aggregation == "mean_pairs":
        return float(S.mean().item())

    if aggregation == "sym_max":
        row_max = S.max(dim=1).values.mean()   # U -> best F, averaged over U
        col_max = S.max(dim=0).values.mean()   # F -> best U, averaged over F
        return float(0.5 * (row_max + col_max).item())

    if aggregation == "max":
        S[[i for i,u in enumerate(text2) for j,f in enumerate(text1) if u==f], 
         [j for i,u in enumerate(text2) for j,f in enumerate(text1) if u==f]] = -float("inf")
        row_max = S.max(dim=1).values.mean()   # default: utterance-centric
        return float(row_max.item())
    



def wmd_gensim(doc1_tokens, doc2_tokens, kv) -> float:
    """
    Word Mover's Distance between two documents (lists of lemmas/tokens).
    - Uses gensim's exact WMD (Earth Mover's Distance over word2vec space).
    - Duplicates matter (term frequency -> transport mass).
    - Unknown tokens are ignored.
    Returns:
      distance (float) — lower is more similar (0 = identical bags).
    """
    # Filter OOV upfront (fastText covers most via subwords; gensim FT handles it)
    d1 = [t for t in doc1_tokens if t in kv]
    d2 = [t for t in doc2_tokens if t in kv]
    if not d1 or not d2:
        return float("nan")
    return kv.wmdistance(d1, d2)







#################################
#### BERTopic Based Analysis ####
#################################




def get_topic_assignments_dataframe(dfData,subjList,fragmentList, model,mode ="unsup",id2name= {1:"Focus", 2:"Task-Related Thoughts", 3:"Mind Wandering", 4:"Mind Blanking"}):
    """
    Returns a DataFrame where each row corresponds to an utterance with:
    - its index
    - the text
    - the most likely topic
    - the probability for each topic
    """

    topic_assignments = []
    if mode == "unsup":
        all_topics, all_probs = model.transform(dfData['Utterance'].tolist())
        topic_labels = model.get_topic_info()["Topic"].tolist()

    elif mode == "sup":
        X = model[0].encode(dfData['Utterance'].tolist(), convert_to_numpy=True, normalize_embeddings=False)
        all_probs = model[1].predict_proba(X)
        all_topics = model[1].predict(X)
        topic_labels =list(range(0,len(list(id2name.keys()))));
    for i, (utt, assigned_topic, prob_vec) in enumerate(zip(dfData['Utterance'].tolist(), all_topics, all_probs)):
        #print(f"{subjList[i]} | {fragmentList[i]} | {1 - prob_vec[0] -prob_vec[2] -prob_vec[1]:1.4f} | {prob_vec[0]:1.4f} | {prob_vec[1]:1.4f} | {prob_vec[2]:1.4f}")
        #print(utt)
        row = {
            "Subject": subjList[i],
            "Part": fragmentList[i],
            "utterance": utt,
            "assigned_topic": assigned_topic,
            "true_topic": id2name[int(dfData['Attention'].iloc[i])]
        }
        # Add individual probabilities
        for tid in topic_labels:
            row[f"prob_topic_{tid}"] = prob_vec[tid] if tid < len(prob_vec) else 0.0
        topic_assignments.append(row)

    return pd.DataFrame(topic_assignments)



def own_Bertopic(dfData,subjList,fragmentList,mode="unsup",
                 seed_topics=None,
                 validation = False,Classess =  {1:"Focus", 2:"Task-Related Thoughts", 3:"Mind Wandering", 4:"Mind Blanking"}): #Seeded BERT parameters
    
    with open(r"C:\Users\barak\Documents\Python_Scripts\CoproraTools\polish.stopwords.txt", encoding="utf-8") as f:
        polish_stopwords = set(line.strip().lower() for line in f if line.strip())

    embedding_model = SentenceTransformer(
    "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
    )

    if mode =="sup": # Suppervissed BERTopic


        # --- 1) Data ---
        # rawCorpus: list[str] of all utterances
        # dfData['Attention']: labels in {1,2,3,4} aligned with rawCorpus
        id2name = Classess
        y = (dfData["Attention"].astype(int).values - 1)  # -> 0..3

        # --- 2) Embeddings ---
        X = embedding_model.encode(dfData['Utterance'], convert_to_numpy=True, normalize_embeddings=False)

        # --- 3) Train/val split ---
        Xtr, Xva, ytr, yva = train_test_split(X, y, test_size=0.1, stratify=y, random_state=42)

        # --- 4) Classifier (multinomial softmax LR) ---
        clf = LogisticRegression(
            max_iter=2000,
            multi_class="multinomial",
            solver="lbfgs",
            class_weight="balanced",
            n_jobs=None
        )
        clf.fit(Xtr, ytr)

        # --- 5) Quick eval ---
        pred = clf.predict(Xva)
        if validation:
            print("Val accuracy:", accuracy_score(yva, pred))
            
            print(classification_report(yva, pred, target_names=[id2name[+1] for i in range(len(id2name))]))
        # --- 6) Predict on ALL utterances ---
        dfTopic = get_topic_assignments_dataframe(dfData,subjList,fragmentList, [embedding_model,clf],mode ="sup",id2name=id2name)
        
        return [embedding_model,clf],dfTopic

    elif mode == "unsup" or mode == "seed": # Classic BERTopic or Seeded (The same Prep Required)

        # =========================
        # 4. Configure UMAP (dimensionality reduction)
        # =========================
        umap_model = umap.UMAP(
            n_neighbors  = 15,      # affects local vs global structure
            n_components = 5,      # reduced embedding dimensions
            metric="cosine",     # works well for semantic embeddings
            random_state = 42
        )

        # =========================
        # 5. Configure HDBSCAN (soft clustering with probabilities)
        # =========================
        hdbscan_model = hdbscan.HDBSCAN(
            min_cluster_size=3,  # minimum size of a topic cluster
            min_samples=1,       # lower = more clusters, higher = more conservative
            metric="euclidean",  # UMAP already reduced cosine → euclidean OK
            cluster_selection_method="eom",
            prediction_data=True # <-- enables .transform() with probabilities
        )

        # =========================
        # 6. Vectorizer for topic labels (Polish stopwords)
        # =========================
        vectorizer_model = CountVectorizer(
            stop_words=list(polish_stopwords),  # or pass a custom list if needed
            ngram_range=(1, 2),   # unigrams + bigrams
            min_df=2              # ignore very rare terms
        )

        # ----------------------------
        # 4) SEEDS: your two target classes
        #    (You can add more seed topics if you want.)

        if  mode == "seed" and seed_topics is None:
            # ----------------------------
            seed_topics = {
                # Class A: "Story Attention" (attentive to the story)
                "Focus": [
                    "historia", "fabuła", "bohater", "narracja", "opowieść", "nie byłem rozproszony", "byłem skopiony","skupiona"
                    "słuchałem uważnie", "skupienie na historii", "skupiłem się na fabule","skupiony","Karolina","Janek"
                ],
                "Task-Related Thoughts":[ "Co zdarzy się", "za chwilę", "co będzie dalej", "jak długo","Janek","Karolina"],
                # Class B: "Inattentiveness" (your minority class to bias toward)
                "Mind Wandering": [
                    "spotkanie", "przyjaciel", "kolega", "koleżanka", "rozmowa",
                    "umówić się", "wyjście na kawę", "randka", "znajomy", "znajoma", "byłem rozproszony","rozproszyło"
                    "przypomniało mi się spotkanie", "myślałem o koledze", "nie byłem skupiony", "czymś innym","prywatnie","po badaniu","trochę"
                ],
                "Mind Blanking":["nie myślałem o niczym", "niczym", "niczym konkretnym","nie potrafię", "przypomnieć"],
                
            }

        # ----------------------------
        # 5) Create topic model (Seeded if available)
        # ----------------------------
        def build_topic_model(mode, nr_topics=4):
            if  mode == "seed":
                print("Yay!!!")
                return BERTopic(
                    seed_topic_list=seed_topics,                # <- seeds steer discovery
                    embedding_model=embedding_model,
                    umap_model=umap_model,
                    hdbscan_model=hdbscan_model,
                    vectorizer_model=vectorizer_model,
                    calculate_probabilities=True,
                    verbose=True,
                    nr_topics=nr_topics                     # e.g. 4 or None; if None, auto
                )
            else:
                return BERTopic(
                    embedding_model=embedding_model,
                    umap_model=umap_model,
                    hdbscan_model=hdbscan_model,
                    vectorizer_model=vectorizer_model,
                    calculate_probabilities=True,
                    verbose=True,
                    nr_topics=nr_topics
                )
        topic_model = build_topic_model(mode,nr_topics=4)

        # =========================
        # 8. Train topic model on ALL utterances
        # =========================
        topics, probs = topic_model.fit_transform(dfData['Utterance'])
        dfTopic = get_topic_assignments_dataframe(dfData,subjList,fragmentList, topic_model) # Get the Corpus Labeled Data Frame

        # Optional: Reduce outliers (assign -1 to nearest topics)
        #topics = topic_model.reduce_outliers(all_utterances, topics, strategy="c-tf-idf")
        return topic_model,dfTopic
    

# =========================
# 11. Function to tag NEW utterances
# =========================
def tag_top_topics(utterance, model,topic_info=None,mode="unsup", topk=3):
    results = []

    if mode == "sup":

        id2name = {0:"Focus", 1:"Task-Related Thoughts", 2:"Mind Wandering", 3:"Mind Blanking"}

        # --- 6) Predict probabilities for new texts (batch or single) ---

        X_new = model[0].encode(utterance, convert_to_numpy=True)
        proba_new = model[1].predict_proba(X_new)  # shape: (n_samples, 4)

        # tidy table: rows = texts, cols = classes
        results = pd.DataFrame(proba_new, columns=[id2name[i] for i in model[1].classes_])
        results.insert(0, "text", utterance)
    else:
        if topic_info is not None:
            _, prob_vec = model.transform([utterance])  # returns (topic_id, probs)
            prob_vec = prob_vec[0]                      # shape: (n_topics,)
            top_ids = np.argsort(prob_vec)[::-1][:topk] # sort indices by prob desc

            for tid in top_ids:
                if tid == -1:
                    label = "OUTLIER"
                    words = []
                else:
                    label = topic_info.set_index("Topic").loc[tid, "Name"]
                    words = [w for (w, _) in model.get_topic(tid)[:3]]  # top-3 words
                results.append({
                    "topic_id": int(tid),
                    "label": label,
                    "prob": float(prob_vec[tid]),
                    "top_words": words
                })
    return results

def printExampleTopics(topicDf,topic_model,topicN = "random",topicId = 1,exNumb=3,prob=True):
    """
        Prints Examples of Topics from the Topic DataFrame
        If topicId is "random" - prints a random utterance from a given topic
    """
    print(f"Topic {topicId} Examples:")
    if topicN == "random":
        if exNumb > len(topicDf[topicDf['assigned_topic'] == topicId]):
            exNumb = len(topicDf[topicDf['assigned_topic'] == topicId])
        topicN = random.sample(list(topicDf[topicDf['assigned_topic'] == topicId].index),exNumb)
    else:
        if type(topicN) is int:
            topicN = [topicN]
    for i in topicN:
        if i in topicDf[topicDf['assigned_topic'] == topicId].index:
            print(f"{i}: {topicDf[topicDf['assigned_topic'] == topicId].loc[i]['utterance']}")
            if prob:
                dd = [f"{topicDf[topicDf['assigned_topic'] == topicId][j].loc[i]}" for j in topicDf.columns[4:]]
                print(f"    Probabilities: {dd}")

    return topicN


def renameTopics(topicDf,newDict = {"Focus":0,"Task-Related Thoughts":1,"Mind Wandering":2,"Mind Blanking":3}):
    """
        Renames the Topics in the Topic DataFrame
        newDict: {NewName:TopicId}
    """
    for k,v in newDict.items():
        topicDf.loc[topicDf['assigned_topic'] == v,'assigned_topic'] = k

    rewDict   = {"prob_topic_"+str(v): k for k, v in newDict.items()}
    topicDf.rename(columns=rewDict,inplace=True)
    for cl in topicDf.columns[4:]:
        topicDf[cl]
    return topicDf


def inspectMisclass(dfTopic,cond = None):

    if cond is None: # If no other rule is specificed, then use the rule to collapse to Focus and MW instances Only
        cond = (
            (dfTopic['assigned_topic'] != dfTopic['true_topic']) &
            ~(
                ((dfTopic['assigned_topic'] == 'Focus') & (dfTopic['true_topic'] == 'Task-Related Thoughts')) |
                ((dfTopic['assigned_topic'] == 'Task-Related Thoughts') & (dfTopic['true_topic'] == 'Focus'))
            )
        )

    for pr in range(len(dfTopic[cond])):
        print(f"({pr+1}) {dfTopic[cond]['Subject'].iloc[pr]} / {dfTopic[cond]['Part'].iloc[pr]}: Assigned vs True: {dfTopic[cond]['assigned_topic'].iloc[pr]} / {dfTopic[cond]['true_topic'].iloc[pr]}")
        print(f"    Text: {dfTopic[cond]['utterance'].iloc[pr]}")
        if len(dfTopic.columns[5:]) >2:
            print(f"    Prob: FOCUS ({dfTopic[cond]['Focus'].iloc[pr]:1.4f}), \n       	  TUT   ({dfTopic[cond]['Task-Related Thoughts'].iloc[pr]:1.4f}),\n       	  MW    ({dfTopic[cond]['Mind Wandering'].iloc[pr]:1.4f}),\n       	  MB    ({dfTopic[cond]['Mind Blanking'].iloc[pr]:1.4f})\n")
        else:
            print(f"    Prob: FOCUS ({dfTopic[cond]['Focus'].iloc[pr]:1.4f}), \n       	  MW    ({dfTopic[cond]['Mind Wandering'].iloc[pr]:1.4f}),\n")

    print(f"Total: {len(dfTopic[cond])}/{len(dfTopic)} misclassified utterances ({len(dfTopic[cond]) / len(dfTopic) * 100:1.2f}%)\n")


#### EXPERIMENTAL!!!

# from bertopic import BERTopic
# from bertopic.dimensionality import BaseDimensionalityReduction
# from bertopic.vectorizers import ClassTfidfTransformer
# from sklearn.linear_model import LogisticRegression
# from sentence_transformers import SentenceTransformer
# from sklearn.feature_extraction.text import CountVectorizer
# import numpy as np
# import pandas as pd


# # ----------------------------
# # 1) Embeddings (Polish-capable)
# # ----------------------------
# embedding_model = SentenceTransformer(
#     "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
# )
# vectorizer_model = CountVectorizer(
#     stop_words=list(polish_stopwords),
#     ngram_range=(1, 2),
#     min_df=2,
#     token_pattern=r"(?u)\b\w\w+\b"
# )

# # ----------------------------
# # 2) Supervised labels (0..3)
# # ----------------------------
# # dfData['Attention'] is 1..4 → convert to 0..3
# y = (dfData['Attention'].values.astype(int) - 1).tolist()

# # ----------------------------
# # 3) Supervised BERTopic (no UMAP/HDBSCAN)
# # ----------------------------
# empty_dim = BaseDimensionalityReduction()
# clf = LogisticRegression(
#     max_iter=2000,
#     multi_class="multinomial",   # <-- softmax
#     solver="lbfgs",
#     class_weight="balanced"
# )
# ctfidf = ClassTfidfTransformer(reduce_frequent_words=True)

# topic_model = BERTopic(
#     embedding_model=embedding_model,
#     umap_model=empty_dim,
#     hdbscan_model=clf,                 # classifier inside BERTopic
#     vectorizer_model=vectorizer_model,
#     ctfidf_model=ctfidf,
#     calculate_probabilities=True,      # ensure probs from transform(...)
#     verbose=True
# )

# topics, probs = topic_model.fit_transform(rawCorpus, y=y)
# print("probs shape:", None if probs is None else probs.shape)  # should be (n_docs, n_topics)

# # ----------------------------
# # 4) Human-readable class names
# # ----------------------------
# id2name = {
#     0: "Focus",
#     1: "Mind Wandering",
#     2: "Task-Related Thoughts",
#     3: "Mind Blanking"
# }

# topic_info = topic_model.get_topic_info().copy()

# # Map original 0..3 labels → topic ids, then invert to topic_id → label
# mappings = topic_model.topic_mapper_.get_mappings()      # {orig_label(0..3) -> topic_id}
# topic2label = {v: k for k, v in mappings.items()}        # {topic_id -> 0..3}
# topic_info["Class"] = topic_info["Topic"].map(
#     lambda tid: "OUTLIER" if tid == -1 else id2name.get(topic2label.get(tid), "OTHER")
# )

# # Optional: inspect
# display(topic_info)



#######################################
#### Similarity to Story Functions ####
#######################################



def computeStorySims(dfData,dfAnalysis,model2,kv,aggr="centroid"):
    dfAnalysisC = dfAnalysis.copy()
    fragmentSentSim = [];
    fragmentLemmaSim = []
    wmdFragments = []
    for i,(t,u) in enumerate(zip(dfData['StoryFrag'].values, dfData['Utterance'].values)):
        #print(f"Processing Fragment: {dfData['Subject'].iloc[i]} | {dfData['Fragment'].iloc[i]}")
        fragmentLemmaSim.append(bertSentenceCosineSim(dfData['StoryLemmas'].iloc[i],dfData['Lemmas'].iloc[i],model2),aggregation = aggr)
        fragmentSentSim.append(bertSentenceCosineSim(dfData['StorySentences'].iloc[i],dfData['Sentences'].iloc[i],model2),aggregation = aggr)
        wmdFragments.append(wmd_gensim(dfData['StoryLemmas'].iloc[i],dfData['Lemmas'].iloc[i],kv))


    dfAnalysisC['FragmentSentCosSim'] = fragmentSentSim
    dfAnalysisC['FragmentLemmaCosSim'] = fragmentLemmaSim
    dfAnalysisC['wmdFragments'] = wmdFragments
    return dfAnalysisC





#################################
#### LLM Surprisal Functions ####
#################################


class PPPL():
        
    def __init__(self, model_name="allegro/herbert-large-cased", device=None, fp16=True, compile_model=False):

        self.mlm = AutoModelForMaskedLM.from_pretrained(model_name)
        self.herbertT = AutoTokenizer.from_pretrained(model_name)
        self.device = device or "cuda" if torch.cuda.is_available() else "cpu"
        self.mlm.to(self.device)
        self.mlm.eval()
        self.fp16 = bool(fp16 and self.device == "cuda")

        # try:
        #     self.mlm = torch.compile(self.mlm)
        # except Exception:
        #     pass


    def _is_word_start(self,tok: str) ->bool:
        # BERT WordPiece: continuation starts with "##"
        # RoBERTa BPE: word starts have "Ġ" prefix; continuation lacks it
        return not tok.startswith("##")

    def _select_positions(self,ids_row, *, only_word_starts=True, skip_special=True, skip_punct=True):
        """
        Choose which token positions to score.
        - only_word_starts: score only first subword of each word (fewer, cleaner)
        - skip_special: drop [CLS]/[SEP]/[PAD]/[MASK]
        - skip_punct: drop tokens that are pure punctuation
        """
        specials = set(self.herbertT.all_special_ids)
        toks = self.herbertT.convert_ids_to_tokens(ids_row.tolist())

        pos = []
        for i, (tid, tok) in enumerate(zip(ids_row.tolist(), toks)):
            if skip_special and tid in specials:
                continue
            if skip_punct and tok.isascii() and all(ch in ".,;:!?-–—()[]{}'\"«»„”…/\\|" for ch in tok):
                continue
            if only_word_starts:
                if self._is_word_start(tok):
                    pos.append(i)
            else:
                pos.append(i)
        return pos

    def _pppl_for_span_fast(self,input_ids, attn_mask, mask_positions, *, batch_size=256, fp16=True):
        """
        Fast PPPL:
        - vectorized masked variants
        - vectorized logprob gather
        - optional grouped masking (approximate)
        """
        if len(mask_positions) == 0:
            return float("nan")

        # --- ensure base tensors are on the SAME device as the model ---
        base_device = next(self.mlm.parameters()).device
        input_ids = input_ids.to(base_device)        # <== moved
        attn_mask = attn_mask.to(base_device)        # <== moved

        L = input_ids.size(1)
        mask_id = self.herbertT.mask_token_id

        N = len(mask_positions)
        masked = input_ids.expand(N, L).clone()          # [N, L] on base_device
        rows = torch.arange(N, device=base_device)       # indices on same device
        cols = torch.tensor(mask_positions, device=base_device, dtype=torch.long)
        masked[rows, cols] = mask_id
        # targets & positions on same self.device as logits later
        target_ids = input_ids[0, cols]                  # [N] on base_device
        target_pos = cols
        row_slices = (rows, target_pos, target_ids)
    

        total_logprob = 0.0
        total_count = 0

        with torch.inference_mode():
            for start in range(0, masked.size(0), batch_size):
                end = start + batch_size
                mb = masked[start:end]                 # already on base_device
                attn = attn_mask.expand(mb.size(0), -1)

                with torch.autocast(device_type=base_device.type, dtype=torch.float16, enabled=(fp16 and base_device.type=="cuda")):
                    logits = self.mlm(mb, attention_mask=attn).logits  # [B, L, V] on base_device

                r = torch.arange(start, min(end, masked.size(0)), device=base_device)
                cols = torch.tensor(mask_positions[start:end], device=base_device, dtype=torch.long)
                tgt = input_ids[0, cols]  # on base_device
                pos_logits = logits[r - start, cols, :]        # [B, V]
                log_probs = pos_logits.log_softmax(dim=-1)
                picked = log_probs.gather(dim=1, index=tgt.view(-1,1)).squeeze(1)
                total_logprob += picked.sum().item()
                total_count += picked.numel()


        return math.exp(- total_logprob / max(1, total_count))


    def pppl(self,text, max_length=None, *, only_word_starts=True, batch_size=256, fp16=True):
        """
        Unconditional PPPL for a single string.
        - Scores only chosen tokens (word starts by default) to speed up & reduce noise.
        - group_size>1 gives an approximate speedup (mask k tokens at once).
        """
        enc = self.herbertT(text, return_tensors="pt",
                    truncation=bool(max_length),
                    max_length=max_length or self.herbertT.model_max_length)
        ids, mask = enc["input_ids"], enc["attention_mask"]
        positions = self._select_positions(ids[0], only_word_starts=only_word_starts)
        return self._pppl_for_span_fast(ids, mask, positions,
                                batch_size=batch_size, fp16=fp16)

    def _utterance_positions_from_pair_encoding(self,enc):
        """
        Robustly get utterance token positions within pair (fragment, utterance)
        using sequence_ids when available; fallback to SEP heuristic.
        """
        # Prefer sequence_ids (fast tokenizers)
        try:
            seq_ids = enc.sequence_ids(0)
            utt_positions = [i for i, sid in enumerate(seq_ids) if sid == 1]
            if utt_positions:
                return utt_positions
        except Exception:
            pass
        # Fallback (BERT-style): [CLS] A [SEP] B [SEP]
        ids = enc["input_ids"][0].tolist()
        sep_id = self.herbertT.sep_token_id  # FIX: use the right tokenizer
        sep_positions = [i for i, tid in enumerate(ids) if tid == sep_id]
        if len(sep_positions) >= 2:
            start = sep_positions[0] + 1
            end = sep_positions[1]
            return list(range(start, end))
        return list(range(len(ids)))

    def pppl_conditioned(self,fragment, utterance, max_length=None, *,
                        only_word_starts=True, batch_size=256, fp16=True):
        """
        PPPL of UTTERANCE tokens given FRAGMENT context.
        - Encodes as a sentence pair.
        - Scores only utterance-side tokens (word starts by default).
        - group_size>1 = approximate speedup.
        """
        enc = self.herbertT(fragment, utterance,
                    return_tensors="pt",
                    truncation=True,
                    max_length=max_length or self.herbertT.model_max_length)
        ids, mask = enc["input_ids"], enc["attention_mask"]
        # get utterance positions then filter with the same selector
        cand = self._utterance_positions_from_pair_encoding(enc)
        # Apply the same selection criteria (word starts / punctuation) on utterance span only
        specials = set(self.herbertT.all_special_ids)
        toks = self.herbertT.convert_ids_to_tokens(ids[0].tolist())
        positions = []
        for i in cand:
            tid = int(ids[0, i])
            if tid in specials: 
                continue
            tok = toks[i]
            if tok.isascii() and all(ch in ".,;:!?-–—()[]{}'\"«»„”…/\\|" for ch in tok):
                continue
            if only_word_starts and not self._is_word_start(tok):
                continue
            positions.append(i)

        return self._pppl_for_span_fast(ids, mask, positions,
                                batch_size=batch_size, fp16=fp16)

    # ---- quick smoke test ----
    # print("PPPL:", pppl("To jest bardzo dziwne zdanie.", group_size=1))
    # print("PPPL (approx k=4):", pppl("To jest bardzo dziwne zdanie.", group_size=4))
    # print("PPPL conditioned:", pppl_conditioned("Chłopiec biegł za psem po parku.",
    #                                             "A ja myślałem o kolacji i o weekendzie.",
    #                                             group_size=4))


from transformers import BitsAndBytesConfig

class PPL():

    def __init__(self,model_name="facebook/xglm-564M",device="cuda",max_length =None):
        self.device = device or "cuda" if torch.cuda.is_available() else "cpu"
        bnb_config = BitsAndBytesConfig(load_in_4bit=True)

        torch_dtype = torch.float16 if self.device == "cuda" else None
        self.lm2 = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch_dtype, quantization_config=bnb_config)
        self.lm2.to(self.device).eval()

        self.tok2 = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        self.max_length = max_length

    # ====== (1) Plain perplexity (fast path + long-text sliding window) ======
    def ppl(self,text, max_length=None, *, fp16=True, stride=None):
        """
        Perplexity of a single string.
        - If len(tokens) <= model_max_length (or max_length), we do ONE forward (fast).
        - Else we use an overlapping sliding window ('stride') to cover the whole text exactly.
        Returns: scalar perplexity = exp(mean negative log-likelihood per token).
        """
        # Encode once on CPU (fast tokenizer), then move tensors when needed
        enc = self.tok2(text, return_tensors="pt", add_special_tokens=True)
        input_ids = enc["input_ids"].to(self.device)
        attn_mask = enc["attention_mask"].to(self.device)

        model_max = max_length or getattr(self.lm2.config, "n_positions", self.tok2.model_max_length)

        ### When all my text is fitting in a context window there is no need to slide, so we 
        seq_len = int(input_ids.size(1))
        if seq_len <= model_max:
            with torch.inference_mode(): # Gettinh to Context Manager - automtaically casting operation of reduced floating point to reduce comp time!
                with torch.autocast(device_type=self.device, dtype=torch.float16, enabled=(fp16 and self.device == "cuda")):
                    out = self.lm2(input_ids=input_ids, attention_mask=attn_mask, labels=input_ids)
            return float(math.exp(out.loss.item()))

        # ----- Long text: exact sliding-window perplexity (HF-recommended pattern) -----
        # Choose stride if not given: advance e.g. 256 tokens per step
        if stride is None:
            stride = min(256, model_max // 2)  # conservative default

        nll_sum = 0.0
        tok_count = 0

        # We will evaluate in windows of up to 'model_max' and only count the NEW tokens per window
        for i in range(0, seq_len, stride):
            begin_loc = max(i + model_max - stride, 0)
            end_loc   = min(i + model_max, seq_len)
            trg_len   = end_loc - i  # how many new tokens we learn this step

         
            input_ids_window = input_ids[:, begin_loc:end_loc]
            attn_window      = attn_mask[:, begin_loc:end_loc]
            labels_window    = input_ids_window.clone()
            window_len = input_ids_window.size(1)
            trg_len = max(0, min(trg_len, window_len))  # 🩹 fix

            # Mask out the loss on the "context" tokens at the left of the window
            labels_window[:, : (labels_window.size(1) - trg_len)] = -100

            with torch.inference_mode():
                with torch.autocast(device_type=self.device, dtype=torch.float16, enabled=(fp16 and self.device == "cuda")):
                    out = self.lm2(input_ids=input_ids_window,
                            attention_mask=attn_window,
                            labels=labels_window)
            nll_sum  += out.loss.item() * trg_len
            tok_count += trg_len

            if end_loc == seq_len:
                break

        return float(math.exp(nll_sum / max(1, tok_count)))


    # ====== (2) Conditioned perplexity: PPL(utterance | fragment) ======
    def ppl_conditioned(self,fragment, utterance, sep="\n\n", max_length=None, *, fp16=True, stride=None):
        """
        Perplexity of UTTERANCE tokens given FRAGMENT context:
        text = fragment + sep + utterance
        We mask out (set -100) the fragment+sep tokens in labels so only utterance contributes to loss.

        Long-text safe: uses same sliding-window logic when needed.
        """
        # Build the full string once
        full_text = fragment + sep + utterance

        # Tokenize full for the model input
        enc_full = self.tok2(full_text, return_tensors="pt", add_special_tokens=True)
        full_ids = enc_full["input_ids"].to(self.device)
        full_att = enc_full["attention_mask"].to(self.device)

        # Compute boundary (utterance start) robustly by tokenizing the prefix with the same tokenizer
        # (This is cheap and avoids messing with special-token accounting.)
        pref_ids = self.tok2(fragment + sep, return_tensors="pt", add_special_tokens=True)["input_ids"][0]
        utt_start = int(pref_ids.size(0))  # position where utterance begins in full_ids[0]

        model_max = max_length or getattr(self.lm2.config, "n_positions", self.tok2.model_max_length)
        seq_len = int(full_ids.size(1))

        if seq_len <= model_max:
            labels = full_ids.clone()
            labels[:, :utt_start] = -100  # ignore fragment+sep
            with torch.inference_mode():
                with torch.autocast(device_type=self.device, dtype=torch.float16, enabled=(fp16 and self.device == "cuda")):
                    out = self.lm2(input_ids=full_ids, attention_mask=full_att, labels=labels)
            return float(math.exp(out.loss.item()))

        # ----- Long text: sliding-window, but only count utterance tokens -----
        if stride is None:
            stride = min(256, model_max // 2)

        nll_sum = 0.0
        tok_count = 0

        for i in range(0, seq_len, stride):
            begin_loc = max(i + model_max - stride, 0)
            end_loc   = min(i + model_max, seq_len)
            input_ids_window = full_ids[:, begin_loc:end_loc]
            attn_window      = full_att[:, begin_loc:end_loc]

            # Build labels: everything left of utterance start OR left context within the window is -100
            labels_window = input_ids_window.clone()

            # Compute the local index of utterance start within this window
            local_utt_start = max(0, utt_start - begin_loc)

            # Mask:
            # (a) tokens before local_utt_start are fragment/sep -> ignore
            # (b) additionally, ignore the left context part of the window (same as ppl() logic)
            # Determine how many "new" tokens this step contributes:
            trg_len = end_loc - i
            left_ctx = labels_window.size(1) - trg_len

            # Compose a mask: everything < local_utt_start or < left_ctx is ignored
            ignore_upto = max(local_utt_start, left_ctx)
            if ignore_upto > 0:
                labels_window[:, :ignore_upto] = -100

            with torch.inference_mode():
                with torch.autocast(device_type=self.device, dtype=torch.float16, enabled=(fp16 and self.device == "cuda")):
                    out = self.lm2(input_ids=input_ids_window,
                            attention_mask=attn_window,
                            labels=labels_window)

            # Count only tokens actually in the utterance region for this step:
            # count range = [max(i, utt_start), end_loc)
            count_this = max(0, end_loc - max(i, utt_start))
            if count_this > 0:
                nll_sum  += out.loss.item() * count_this
                tok_count += count_this

            if end_loc == seq_len:
                break

        if tok_count == 0:
            return float("nan")
        return float(math.exp(nll_sum / tok_count))


#####################################
#### Semantic Graphing Functions ####
#####################################


def addWeightIds(dfWn,dfRelations):
    """
        This function changes the names of a Relations for Parents WITH their Appropriat
    """

    dfWnC = dfWn.copy()

    # 2) Compute the correct ID for the (possibly replaced) name by merging
    dfWnC = dfWnC.merge(dfRelations[['name','description','id']].rename(columns={'id':'id_weights'}),
                on='name', how='left')

  
    return dfWnC



def getParentsOnly(dfWn,dfRelations):
    """
        This function changes the names of a Relations for Parents WITH their Appropriate
    """

    dfWnC = dfWn.copy()
    # 1) Replace 'name' with 'parent' when parent is present
    dfWnC['name'] = np.where(dfWnC['parent'].notna(), dfWnC['parent'], dfWnC['name'])

    # 2) Compute the correct ID for the (possibly replaced) name by merging
    dfWnC = dfWnC.merge(dfRelations[['name','description','id']].rename(columns={'id':'id_from_name'}),
                on='name', how='left')

    # 3) Overwrite id with id_from_name only where parent was present
    dfWnC['id'] = np.where(dfWnC['parent'].notna(), dfWnC['id_from_name'], dfWnC['id'])

    # 4) Tidy up
    dfWnC = dfWnC.drop(columns=['id_from_name'])
    return dfWnC

### Inspect Different Children of Common Relations:
def inspectRelationsChildren(dfRelations,keyword = None):
    return dfRelations[dfRelations['parent'].apply(lambda x: keyword in str(x))]




def showRelationExamples(dfWn, relList = None,N=3):
    """
        Prints N examples of a given Relation from a current Named Dataset or a List provided
    """

    if relList is None:
        relList = list(dfWn['name'].unique())

    for i,rel in enumerate(relList):
        print(f"\n\n ==== Current Relation: {rel} ==== N: {len(dfWn[dfWn['name']==rel][['NamesFirst','name','NamesSecond']])} \n")
        if len(dfWn[dfWn['name']==rel][['NamesFirst','name','NamesSecond']]) >= N:
            print(dfWn[dfWn['name']==rel][['NamesFirst','name','NamesSecond']].sample(n=N).to_string())
        else:
            print(dfWn[dfWn['name']==rel][['NamesFirst','name','NamesSecond']].sample(n=len(dfWn[dfWn['name']==rel][['NamesFirst','name','NamesSecond']])).to_string())


    

def get_RelationsCountDF(dfWn,dfRelations):
    """
        Getting Data Frame of Relation Count Supplemented by their Descriptions
    """
    EnglishRelations = ['Hypernym',
    'Substance_holonym',
    'equivalent',
    'sumo_instance',
    'Hyponym',
    'Member_holonym',
    'Part_holonym',
    'Part_meronym',
    'Domain_of_synset_-_TOPIC',
    'Substance_meronym',
    'Attribute',
    'Similar_to',
    'Also see',
    'Domain_of_synset_-_USAGE',
    'Domain_of_synset_-_REGION',
    'Member_of_this_domain_-_REGION',
    'Member_of_this_domain_-_TOPIC',
    'Instance_Hyponym',
    'Instance_Hypernym',
    'Member_meronym',
    'Member_of_this_domain_-_USAGE',
    'instance_of',
    'Verb_Group',
    'Entailment',
    'Cause']
    
    dfWnCount = dfWn.groupby("id").count().sort_values(by="firstSynset",ascending=False)

    ### Get Indicies of Relationwhich count is greater than 100:
    #good_indices = list(dfWnCount[dfWnCount['SecondSynset']>100].index)

    dfWnCount = pd.merge(dfWnCount[['firstSynset']],dfRelations[['id','name','parent','description']],on='id')
    dfWnCount['parent'] = dfWnCount['parent'] == 0
    dfWnCount.rename(columns = {'name':'RelationName','firstSynset':'RelationCount'},inplace=True)
    dfWnCount = dfWnCount[dfWnCount['RelationName'].apply(lambda x: x not in EnglishRelations)]
    return dfWnCount

def showRelationExamples(dfWn, relList = None,N=3):
    """
        Prints N examples of a given Relation from a current Named Dataset or a List provided
    """

    if relList is None:
        relList = list(dfWn['name'].unique())

    for i,rel in enumerate(relList):
        print(f"\n\n ==== Current Relation: {rel} ==== N: {len(dfWn[dfWn['name']==rel][['NamesFirst','name','NamesSecond']])} \n")
        print(dfWn[dfWn['name']==rel][['NamesFirst','name','NamesSecond']].head(N).to_string())

### Get All Relations

def weightCorrespondanceComp(dfWn,dfRelations):
  dfWnA = addWeightIds(dfWn,dfRelations)

  WeightIDCorrespondence = (
      dfWnA.groupby('id', sort=False)['id_weights']
        .agg(lambda s: sorted({x for x in s.dropna()}))
        .reset_index(name='id_weights_for_id')
  )

  WeightIDCorrespondence['id_weights_for_id'] = WeightIDCorrespondence['id_weights_for_id'].apply(lambda x: x[0])
  return WeightIDCorrespondence



################################
### Semantic Graph Overviews ###
################################




def ownGraphMaking(dfWn,dfRelationsW,dfSynsets):

    """
        This is Home-brewed function that aims at making a graph of Nodes Based on my own DFs.
        Synsets are Always Characterised by Word List - Nothing More - nothing Less! 

        We Iterate over positions in dfWn, giving us all relations that has been happening 
    """

    G = nx.Graph()
    omitted = 0
    DEFAULT_WEIGHT = 0.5
    ### 1. Creating A Nodes List
    for i in range(len(dfSynsets)):

        ### Adding Node
        G.add_node(("syn", dfSynsets['firstSynset'].iloc[i]), obj=dfSynsets['NamesFirst'].iloc[i])

    ### 2. Add Synsets Relations:

    for i in range(len(dfWn)):

        

        if not len(dfRelationsW[dfRelationsW['id'] == dfWn['id'].iloc[i]]['Weight']):
            omitted +=1
            #print(f" on {i}/{dfWn['name'].iloc[i]} there are no Weights: - Probably an English Word: AddingDefault ")
            #currWeight = DEFAULT_WEIGHT
            continue  # ← CHANGED: SKIP when weight is missing (no default)

        else:
            print( dfRelationsW[dfRelationsW['id'] == dfWn['id'].iloc[i]]['Weight'])
            currWeight = dfRelationsW[dfRelationsW['id'] == dfWn['id'].iloc[i]]['Weight'].iloc[0]

        u, v = ("syn", dfWn['firstSynset'].iloc[i]), ("syn", dfWn['SecondSynset'].iloc[i])
        if G.has_edge(u, v):
            G[u][v]["weight"] = min(G[u][v]["weight"], currWeight)
        else:
            G.add_edge(u, v, weight=currWeight)

    print(f"Omitted Values {omitted}")
    return G



def getNodeSympathsIDs(lemma,G):
    """
    Returns list of IDs and Names of Lemma adjacent Synsets
    """

    if len([synpos[1]['obj'] for synpos in G.nodes.items()  if lemma in synpos[1]['obj']]):
        return ([synpos[1]['obj'] for synpos in G.nodes.items()  if lemma in synpos[1]['obj']],
            [synpos[0] for synpos in G.nodes.items()  if lemma in synpos[1]['obj']])
    else: #### Look for Words inside Synsets:
        return ([synpos[1]['obj'] for synpos in G.nodes.items()  if len([True for word in synpos[1]['obj'] if lemma in word.split(" ")])],
            [synpos[0] for synpos in G.nodes.items()  if len([True for word in synpos[1]['obj'] if lemma in word.split(" ")])])


def computePaths(G,firstSyns,secondSyns):
    """
        get shortest path between two Synsets:
        Returns Ids of Nodes n Path and additionally the lsit of Names of Synsets on a Path
    """
    if nx.has_path(G,firstSyns, secondSyns):
        path = nx.shortest_path(G,firstSyns,secondSyns)
        names = [G.nodes[nd].get('obj', []) for nd in path]

    else:
        print("     Warning! No Path between ")
        path = [None]
        names = [firstSyns,secondSyns]
    
    return path,names


def computePathWeights(path,G,avg1):
    """
        Returns computed weights in a Path Both as list and a single Number: 
    """
    allWeights = []
    for i in range(len(path)-1):
        allWeights.append(G.edges[path[i],path[i+1]]['weight'])
    if avg1:
        avgNum = len(allWeights)
        hopP = 0.05*np.median*(allWeights)*(len(allWeights)-1)
    else:
        avgNum = 1;
        hopP = 0
    return sum(allWeights)/avgNum +hopP,allWeights



#### Compute Every Path Variation between conjoined Synsets 

def GetUtteranceWeights(G,lemmas,mode = "min",avg1=False):
    """
        gets Weight for Every Concept transition in an Utterance.
        Computes Shortest Path in our Semantic Graph G: Computing it for every possible Synset combination between two lemmas

        Returns - Dictionairy of all path specific
                - DataFrame   of Weights for every subsequent Concept Pairs with Paths and OnPath Synsets
    """

    if mode == "min":
        aggr = lambda x: min(x);
    elif mode == "max":
        aggr = lambda x: max(x);
    minimNames = []
    minimPath = []
    minimWeights = []
    pathsDict = {}
    outOfPlaceLemmas = []
    for i in range(len(lemmas)-1):
        synsFirst = getNodeSympathsIDs(lemmas[i],G)
        synsSecond = getNodeSympathsIDs(lemmas[i+1],G)
        SynsetPathsIds = []
        SynsetPathsNames = []
        SynsetPathWeights = []
      
        if len(synsFirst[1]) == 0 or  len(synsSecond[1]) == 0:
            SynsetPathsIds.append(None)
            SynsetPathsNames.append([lemmas[i],lemmas[i+1]])
        if len(synsFirst[1]) == 0:
            outOfPlaceLemmas.append([lemmas[i]])
            print(f"    No Synset found for 1st lemma: {lemmas[i]}. Setting the distance to 100")
            SynsetPathWeights.append(100)

        if len(synsSecond[1]) == 0:
            print(f"    No Synset found for 2nd lemma: {lemmas[i+1]}. Omitting!")
            SynsetPathWeights.append(None)
            

        for firstSyns in synsFirst[1]:

            for secondSyns in synsSecond[1]:

                synp,namep = computePaths(G,firstSyns,secondSyns)
                SynsetPathsIds.append(synp)
                SynsetPathsNames.append(namep)
                if synp is not None:
                    w,_  = computePathWeights(synp,G,avg1)
                else:
                    w = None # Figure Out what to do with it 
                SynsetPathWeights.append(w)

        pathsDict[f"{lemmas[i]}-{lemmas[i+1]}"]= pd.DataFrame([SynsetPathsIds,SynsetPathsNames,SynsetPathWeights],index=['PathIds','Synsets','Weights']).transpose()
        minimWeights.append(aggr(SynsetPathWeights))
        minimPath.append(SynsetPathsIds[SynsetPathWeights.index(aggr(SynsetPathWeights))])
        minimNames.append(SynsetPathsNames[SynsetPathWeights.index(aggr(SynsetPathWeights))])

    return pd.DataFrame([lemmas[:-1],lemmas[1:],minimWeights,minimPath,minimNames],index = ['firstEnt','secondEnt','Weights','Path','Synsets']).transpose(),pathsDict,outOfPlaceLemmas

def get_Subgraph():
    """
        Gets a subgraph based on provided synset list:
    """


def weightlessPath():
    

    """ A function that just Recomputes the Path Weights based SOLELY on number of nodes - assuming equidistant relations.
      
        Less Linguistcally and semantically viable, but less ambigious
          
        Returns --> new Data Frame containing "Weights" for each Lemma Pair  """
    

def computeMinimalViableTree():
    """
        Computes minimal Viable tree by taking eighter weights between all the nodes in a 
    
    """




#############################################
##### ---- FAULTY FUNCTIONS ------- ######
#############################################



# This section is about functions that are faulty or Old - can be reused in some capacity at some point! 






# def getNodeSympathsIDs(lemma,G):
#     """
#     Returns list of IDs and Names of Lemma adjacent Synsets
#     """

#     if len([synpos[1]['obj'] for synpos in G.nodes.items()  if lemma in synpos[1]['obj']]):
#         return ([synpos[1]['obj'] for synpos in G.nodes.items()  if lemma in synpos[1]['obj']],
#             [synpos[0] for synpos in G.nodes.items()  if lemma in synpos[1]['obj']])
#     else: #### Look for Words inside Synsets:
#         return ([synpos[1]['obj'] for synpos in G.nodes.items()  if len([True for word in synpos[1]['obj'] if lemma in word.split(" ")])],
#             [synpos[0] for synpos in G.nodes.items()  if len([True for word in synpos[1]['obj'] if lemma in word.split(" ")])])



# #### Compute Every Path Variation between conjoined Synsets 

# def GetUtteranceWeights(G,lemmas,mode = "min",avg1=False):
#     """
#         gets Weight for Every Concept transition in an Utterance.
#         Computes Shortest Path in our Semantic Graph G: Computing it for every possible Synset combination between two lemmas

#         Returns - Dictionairy of all path specific
#                 - DataFrame   of Weights for every subsequent Concept Pairs with Paths and OnPath Synsets
#     """

#     if mode == "min":
#         aggr = lambda x: min(x);
#     elif mode == "max":
#         aggr = lambda x: max(x);
#     minimNames = []
#     minimPath = []
#     minimWeights = []
#     pathsDict = {}
#     outOfPlaceLemmas = []
#     for i in range(len(lemmas)-1):
#         synsFirst = getNodeSympathsIDs(lemmas[i],G)
#         synsSecond = getNodeSympathsIDs(lemmas[i+1],G)
#         print(synsFirst)
#         print()
#         SynsetPathsIds = []
#         SynsetPathsNames = []
#         SynsetPathWeights = []

#          # Handle missing synsets for either lemma
#         if len(synsFirst[1]) == 0 or len(synsSecond[1]) == 0:
#             SynsetPathsIds.append(None)
#             SynsetPathsNames.append([lemmas[i], lemmas[i+1]])
#         if len(synsFirst[1]) == 0:
#             outOfPlaceLemmas.append([lemmas[i]])
#             print(f"    No Synset found for 1st lemma: {lemmas[i]}. Setting the distance to 100")
#             SynsetPathWeights.append(100)
#         elif len(synsSecond[1]) == 0:
#             print(f"    No Synset found for 2nd lemma: {lemmas[i+1]}. Omitting!")
#             SynsetPathWeights.append(None)

#         for firstSyns in synsFirst[1]:

#             for secondSyns in synsSecond[1]:

#                 synp,namep = computePaths(G,firstSyns,secondSyns)
#                 SynsetPathsIds.append(synp)
#                 SynsetPathsNames.append(namep)
#                 if synp is not None:
#                     w,_  = computePathWeights(synp,G,avg1)
#                 else:
#                     w = None # Figure Out what to do with it 
#                 SynsetPathWeights.append(w)

#         pathsDict[f"{lemmas[i]}-{lemmas[i+1]}"]= pd.DataFrame([SynsetPathsIds,SynsetPathsNames,SynsetPathWeights],index=['PathIds','Synsets','Weights']).transpose()

#         minimWeights.append(aggr(SynsetPathWeights))
#         minimPath.append(SynsetPathsIds[SynsetPathWeights.index(aggr(SynsetPathWeights))])
#         minimNames.append(SynsetPathsNames[SynsetPathWeights.index(aggr(SynsetPathWeights))])

#     return pd.DataFrame([lemmas[:-1],lemmas[1:],minimWeights,minimPath,minimNames],index = ['firstEnt','secondEnt','Weights','Path','Synsets']).transpose(),pathsDict,outOfPlaceLemmas



# import networkx as nx
# import numpy as np
# from collections import defaultdict

# def disambiguate_lemma(lemma, context_lemmas, G, window=2, verbose=False):
#     """
#     Word-Sense Disambiguation (graph-based, simplified PPR-like version).

#     Parameters
#     ----------
#     lemma : str
#         The target lemma whose correct synset we want.
#     context_lemmas : list[str]
#         Nearby lemmas in the same utterance (left/right window).
#     G : nx.Graph
#         Your weighted WordNet graph: nodes = synset IDs, 
#         node['obj'] = list of lemmas; edge['weight'] = semantic cost.
#     window : int
#         How many neighbours on each side to treat as context.
#     verbose : bool
#         Print diagnostic info.

#     Returns
#     -------
#     best_synset : hashable
#         The ID of the synset in G chosen for the lemma.
#     scores : dict
#         Score assigned to every candidate synset.
#     """

#     dataSyns,candidates = getNodeSympathsIDs(lemma,G)

#     # ---- 1. collect candidate synsets for the lemma ----
#    # candidates = [n for n, data in G.nodes(data=True)
#    #               if lemma in data.get('obj', [])]
    
#     if not candidates:
#         if verbose:
#             print(f"[WSD] no synsets for {lemma}")
#         return None, {}

#     # ---- 2. gather context synsets (for neighbouring lemmas) ----
#     context_nodes = set()
#     for neigh in context_lemmas:
#         for n, data in G.nodes(data=True):
#             if neigh in data.get('obj', []):
#                 context_nodes.add(n)

#     # ---- 3. compute connectivity score to context ----
#     scores = defaultdict(float)
#     for cand in candidates:
#         total = 0.0
#         count = 0
#         for ctx in context_nodes:
#             if cand == ctx:
#                 continue
#             if nx.has_path(G, cand, ctx):
#                 # shortest path cost (lower = closer)
#                 path_len = nx.shortest_path_length(G, cand, ctx, weight='weight')
#                 total += 1.0 / (1.0 + path_len)   # convert to similarity
#                 count += 1
#         if count > 0:
#             scores[cand] = total / count
#         else:
#             scores[cand] = 0.0

#     # ---- 4. choose the synset with max average similarity ----
    
#     best_synset = max(scores, key=scores.get)
#     if verbose:
#         synsetSayings = candidates.index(best_synset)
#         print(f"[WSD] {lemma:15s} → {best_synset} (score={scores[best_synset]:.3f})")
#     return best_synset, scores







# import networkx as nx
# from functools import lru_cache

# def build_lemma_index(G):
#     """
#     Build lemma -> list[synset_id] once. Expects node['obj'] to be list of lemmas.
#     """
#     idx = {}
#     iddata = {}
#     for nid, data in G.nodes(data=True):
#         for lemma in data.get('obj', []):
#             idx.setdefault(lemma, []).append(nid)
#             iddata.setdefault(nid,[]).append(data)
#     return idx,iddata

# def induce_khop_subgraph(G, seeds, k=2):
#     """
#     Induce a subgraph containing nodes within k hops (unweighted hop count) of seeds.
#     Greatly reduces Dijkstra's search space.
#     """
#     visited = set(seeds)
#     frontier = set(seeds)
#     for _ in range(k):
#         nbrs = set()
#         for u in frontier:
#             nbrs.update(G.predecessors(u) if G.is_directed() else G.neighbors(u))
#             if G.is_directed():
#                 nbrs.update(G.successors(u))
#         frontier = nbrs - visited
#         visited |= frontier
#     return G.subgraph(visited).copy()

# def average_similarity_from_context(dist_map, candidates):
#     """
#     Convert distances to mean similarity for each candidate.
#     dist_map: dict[node -> distance] returned by multi-source Dijkstra
#     """
#     scores = {}
#     for c in candidates:
#         d = dist_map.get(c, None)
#         # multi-source dijkstra gives the best (min) distance from any context node
#         # We can use 1/(1+d) as similarity; if you want an average over all context nodes,
#         # switch to single-source for each context, but that defeats the purpose.
#         if d is None:
#             scores[c] = 0.0
#         else:
#             scores[c] = 1.0 / (1.0 + d)
#     return scores

# def disambiguate_lemma_fast(lemma, ctx_lemmas, lemma2syn, G,
#                             window=2, khop=2, use_subgraph=True):
#     """
#     Fast, corpus-free WSD:
#     - collect candidate synsets for target lemma
#     - collect candidate synsets for context window
#     - (optional) induce small k-hop subgraph around seeds
#     - run ONE multi-source Dijkstra from all context synsets
#     - score each candidate by 1/(1+dist)
#     """
#     # candidates for the target lemma
#     candidates = lemma2syn.get(lemma, [])
#     if not candidates:
#         return None, {}



#     # context synsets (union)
#     context_syns = set()
#     for w in ctx_lemmas:
#         context_syns.update(lemma2syn.get(w, []))
#     if not context_syns:
#         # no context evidence; default to first or any prior you may keep
#         return candidates[0], {c: 0.0 for c in candidates}

#     # optionally work on a tiny induced subgraph to speed up Dijkstra
#     Guse = G
#     if use_subgraph:
#         seeds = set(candidates) | set(context_syns)
#         Guse = induce_khop_subgraph(G, seeds, k=khop)

#     # multi-source Dijkstra once per lemma
#     # NOTE: weights must be "costs" (lower=closer)
#     dist = nx.multi_source_dijkstra_path_length(Guse, context_syns, weight='weight')

#     scores = average_similarity_from_context(dist, candidates)
#     best = max(scores, key=scores.get)
#     return best, scores




# def getNodeSympathsIDs(lemma,G):
#     """
#     Returns list of IDs and Names of Lemma adjacent Synsets
#     """

#     if len([synpos[1]['obj'] for synpos in G.nodes.items()  if lemma in synpos[1]['obj']]):
#         return ([synpos[1]['obj'] for synpos in G.nodes.items()  if lemma in synpos[1]['obj']],
#             [synpos[0] for synpos in G.nodes.items()  if lemma in synpos[1]['obj']])
#     else: #### Look for Words inside Synsets:
#         return ([synpos[1]['obj'] for synpos in G.nodes.items()  if len([True for word in synpos[1]['obj'] if lemma in word.split(" ")])],
#             [synpos[0] for synpos in G.nodes.items()  if len([True for word in synpos[1]['obj'] if lemma in word.split(" ")])])


# def computePaths(G,firstSyns,secondSyns):
#     """
#         get shortest path between two Synsets:
#         Returns Ids of Nodes n Path and additionally the lsit of Names of Synsets on a Path
#     """
#     if nx.has_path(G,firstSyns, secondSyns):
#         path = nx.shortest_path(G,firstSyns,secondSyns)
#         names = [G.nodes[nd].get('obj', []) for nd in path]

#     else:
#         print(f"     Warning! No Path between {firstSyns[0]} and {secondSyns[0]}" )
#         path = [None]
#         names = [firstSyns,secondSyns]
    
#     return path,names


# def computePathWeights(path,G,avg1):
#     """
#         Returns computed weights in a Path Both as list and a single Number: 
#     """
#     allWeights = []
#     for i in range(len(path)-1):
#         allWeights.append(G.edges[path[i],path[i+1]]['weight'])
#     if avg1:
#         avgNum = len(allWeights)
#         hopP = 0.05*np.median*(allWeights)*(len(allWeights)-1)
#     else:
#         avgNum = 1;
#         hopP = 0
#     return sum(allWeights)/avgNum +hopP,allWeights



# #### Compute Every Path Variation between conjoined Synsets 

# def GetUtteranceWeights(G,lemmas,mode = "min",avg1=False):
#     """
#         gets Weight for Every Concept transition in an Utterance.
#         Computes Shortest Path in our Semantic Graph G: Computing it for every possible Synset combination between two lemmas

#         Returns - Dictionairy of all path specific
#                 - DataFrame   of Weights for every subsequent Concept Pairs with Paths and OnPath Synsets
#     """

#     if mode == "min":
#         aggr = lambda x: min(x);
#     elif mode == "max":
#         aggr = lambda x: max(x);
#     minimNames = []
#     minimPath = []
#     minimWeights = []
#     pathsDict = {}
#     outOfPlaceLemmas = []
#     for i in range(len(lemmas)-1):


#         lemmaFirst = tempL[i]
#         lemmaSecond = tempL[i+1]

#         contextEnvelopeBeg1 = 0
#         contextEnvelopeEnd1 = 5
#         if i >2 and i <=  len(tempL)-3 :
#             contextEnvelopeBeg1 = 0+i-2
#             contextEnvelopeEnd1 = 5+i-2
            
#             context = tempL[contextEnvelopeBeg1:contextEnvelopeEnd1]
#             contextSecond = context[contextEnvelopeBeg1+1:contextEnvelopeEnd1+1]
#             contextSecond.remove(contextSecond[2]) 

#             context.remove(context[2]) 
#         elif i > len(tempL)-3:
#             contextEnvelopeEnd1 = len(tempL)
#             contextEnvelopeBeg1 =  len(tempL)-5
#             context = tempL[contextEnvelopeBeg1:contextEnvelopeEnd1]
#             contextSecond = context[contextEnvelopeBeg1+1:contextEnvelopeEnd1+1]
#             contextSecond.remove(context[len(tempL) -i]) 
#             context.remove(context[ len(tempL) -i-1]) 

#         else:
#             context = tempL[contextEnvelopeBeg1:contextEnvelopeEnd1]
#             contextSecond = context[contextEnvelopeBeg1+1:contextEnvelopeEnd1+1]
#             contextSecond.remove(context[i+1]) 
#             context.remove(context[i]) 

#         #disambiguate_lemma(currentLemma, context, G, window=2, verbose=True)
#         synsFirst1,_ = disambiguate_lemma_fast(lemmaFirst, context, lemma2syn,G, window=2)
#         synsSecond1,_ = disambiguate_lemma_fast(lemmaSecond, contextSecond, lemma2syn,G, window=2)


#         synsFirst = (synsFirst1,lemma2synwords[synsFirst1])
#         synsSecond =  (synsSecond1,lemma2synwords[synsSecond1])
#         # Get better search with this Dictionairy:
#         # synsFirst = getNodeSympathsIDs(lemmas[i],G)
#         # synsSecond = getNodeSympathsIDs(lemmas[i+1],G)

#         SynsetPathsIds = []
#         SynsetPathsNames = []
#         SynsetPathWeights = []

#          # Handle missing synsets for either lemma
#         if len(synsFirst[1]) == 0 or len(synsSecond[1]) == 0:
#             SynsetPathsIds.append(None)
#             SynsetPathsNames.append([lemmas[i], lemmas[i+1]])
#         if len(synsFirst[1]) == 0:
#             outOfPlaceLemmas.append([lemmas[i]])
#             print(f"    No Synset found for 1st lemma: {lemmas[i]}.Omitting")
#             SynsetPathWeights.append(100)
#         elif len(synsSecond[1]) == 0:
#             print(f"    No Synset found for 2nd lemma: {lemmas[i+1]}. Setting distance to NAN for later !")
#             SynsetPathWeights.append(None)


#         computePaths(G,synsFirst,synsSecond)
#         for firstSyns in synsFirst[1]:

#             for secondSyns in synsSecond[1]:

#                 synp,namep = computePaths(G,firstSyns,secondSyns)
#                 SynsetPathsIds.append(synp)
#                 SynsetPathsNames.append(namep)
#                 if synp is not None:
#                     w,_  = computePathWeights(synp,G,avg1)
#                 else:
#                     w = None # Figure Out what to do with it 
#                 SynsetPathWeights.append(w)

#         pathsDict[f"{lemmas[i]}-{lemmas[i+1]}"]= pd.DataFrame([SynsetPathsIds,SynsetPathsNames,SynsetPathWeights],index=['PathIds','Synsets','Weights']).transpose()

#         minimWeights.append(aggr(SynsetPathWeights))
#         minimPath.append(SynsetPathsIds[SynsetPathWeights.index(aggr(SynsetPathWeights))])
#         minimNames.append(SynsetPathsNames[SynsetPathWeights.index(aggr(SynsetPathWeights))])

#     return pd.DataFrame([lemmas[:-1],lemmas[1:],minimWeights,minimPath,minimNames],index = ['firstEnt','secondEnt','Weights','Path','Synsets']).transpose(),pathsDict,outOfPlaceLemmas



# import networkx as nx
# import numpy as np
# from collections import defaultdict

# def disambiguate_lemma(lemma, context_lemmas, G, window=2, verbose=False):
#     """
#     Word-Sense Disambiguation (graph-based, simplified PPR-like version).

#     Parameters
#     ----------
#     lemma : str
#         The target lemma whose correct synset we want.
#     context_lemmas : list[str]
#         Nearby lemmas in the same utterance (left/right window).
#     G : nx.Graph
#         Your weighted WordNet graph: nodes = synset IDs, 
#         node['obj'] = list of lemmas; edge['weight'] = semantic cost.
#     window : int
#         How many neighbours on each side to treat as context.
#     verbose : bool
#         Print diagnostic info.

#     Returns
#     -------
#     best_synset : hashable
#         The ID of the synset in G chosen for the lemma.
#     scores : dict
#         Score assigned to every candidate synset.
#     """

#     dataSyns,candidates = getNodeSympathsIDs(lemma,G)

#     # ---- 1. collect candidate synsets for the lemma ----
#    # candidates = [n for n, data in G.nodes(data=True)
#    #               if lemma in data.get('obj', [])]
    
#     if not candidates:
#         if verbose:
#             print(f"[WSD] no synsets for {lemma}")
#         return None, {}

#     # ---- 2. gather context synsets (for neighbouring lemmas) ----
#     context_nodes = set()
#     for neigh in context_lemmas:
#         for n, data in G.nodes(data=True):
#             if neigh in data.get('obj', []):
#                 context_nodes.add(n)

#     # ---- 3. compute connectivity score to context ----
#     scores = defaultdict(float)
#     for cand in candidates:
#         total = 0.0
#         count = 0
#         for ctx in context_nodes:
#             if cand == ctx:
#                 continue
#             if nx.has_path(G, cand, ctx):
#                 # shortest path cost (lower = closer)
#                 path_len = nx.shortest_path_length(G, cand, ctx, weight='weight')
#                 total += 1.0 / (1.0 + path_len)   # convert to similarity
#                 count += 1
#         if count > 0:
#             scores[cand] = total / count
#         else:
#             scores[cand] = 0.0

#     # ---- 4. choose the synset with max average similarity ----
    
#     best_synset = max(scores, key=scores.get)
#     if verbose:
#         synsetSayings = candidates.index(best_synset)
#         print(f"[WSD] {lemma:15s} → {best_synset} (score={scores[best_synset]:.3f})")
#     return best_synset, scores


# _PAIR_DIST_CACHE = {}


# def pair_cost(G, a, b, khop=30, use_subgraph=True):
#     """
#     Weighted shortest-path COST between two *chosen* synset IDs a,b.
#     Returns float cost or None if no path.
#     """
#     if a is None or b is None:
#         return None
#     key = (a, b)
#     if key in _PAIR_DIST_CACHE:
#         return _PAIR_DIST_CACHE[key]

#     Guse = induce_khop_subgraph(G, {a, b}, k=khop) if use_subgraph else G
#     try:
#         d = nx.shortest_path_length(Guse, a, b, weight='weight')
#     except nx.NetworkXNoPath:
#         # try once on the full graph before giving up
#         try:
#             d = nx.shortest_path_length(G, a, b, weight='weight')
#         except nx.NetworkXNoPath:
#             d = None

#     _PAIR_DIST_CACHE[key] = d
#     return d



# import networkx as nx
# from functools import lru_cache

# def build_lemma_index(G):
#     """
#     Build lemma -> list[synset_id] once. Expects node['obj'] to be list of lemmas.
#     """
#     idx = {}
#     iddata = {}
#     for nid, data in G.nodes(data=True):
#         for lemma in data.get('obj', []):
#             idx.setdefault(lemma, []).append(nid)
#             iddata.setdefault(nid,[]).append(data)
#     return idx,iddata

# def induce_khop_subgraph(G, seeds, k=2):
#     """
#     Induce a subgraph containing nodes within k hops (unweighted hop count) of seeds.
#     Greatly reduces Dijkstra's search space.
#     """
#     visited = set(seeds)
#     frontier = set(seeds)
#     for _ in range(k):
#         nbrs = set()
#         for u in frontier:
#             nbrs.update(G.predecessors(u) if G.is_directed() else G.neighbors(u))
#             if G.is_directed():
#                 nbrs.update(G.successors(u))
#         frontier = nbrs - visited
#         visited |= frontier
#     return G.subgraph(visited).copy()

# def average_similarity_from_context(dist_map, candidates):
#     """
#     Convert distances to mean similarity for each candidate.
#     dist_map: dict[node -> distance] returned by multi-source Dijkstra
#     """
#     scores = {}
#     for c in candidates:
#         d = dist_map.get(c, None)
#         # multi-source dijkstra gives the best (min) distance from any context node
#         # We can use 1/(1+d) as similarity; if you want an average over all context nodes,
#         # switch to single-source for each context, but that defeats the purpose.
#         if d is None:
#             scores[c] = 0.0
#         else:
#             scores[c] = 1.0 / (1.0 + d)
#     return scores

# def disambiguate_lemma_fast(lemma, ctx_lemmas, lemma2syn, G,
#                             window=2, khop=2, use_subgraph=True):
#     """
#     Fast, corpus-free WSD:
#     - collect candidate synsets for target lemma
#     - collect candidate synsets for context window
#     - (optional) induce small k-hop subgraph around seeds
#     - run ONE multi-source Dijkstra from all context synsets
#     - score each candidate by 1/(1+dist)
#     """
#     # candidates for the target lemma
#     candidates = lemma2syn.get(lemma, [])
#     if not candidates:
#         return None, {}



#     # context synsets (union)
#     context_syns = set()
#     for w in ctx_lemmas:
#         context_syns.update(lemma2syn.get(w, []))
#     if not context_syns:
#         # no context evidence; default to first or any prior you may keep
#         return candidates[0], {c: 0.0 for c in candidates}

#     # optionally work on a tiny induced subgraph to speed up Dijkstra
#     Guse = G
#     if use_subgraph:
#         seeds = set(candidates) | set(context_syns)
#         Guse = induce_khop_subgraph(G, seeds, k=khop)

#     # multi-source Dijkstra once per lemma
#     # NOTE: weights must be "costs" (lower=closer)
#     dist = nx.multi_source_dijkstra_path_length(Guse, context_syns, weight='weight')

#     scores = average_similarity_from_context(dist, candidates)
#     best = max(scores, key=scores.get)
#     return best, scores
