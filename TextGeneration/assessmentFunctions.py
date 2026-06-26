
############################
##### HELPER FUNCTIONS #####
############################

#### === Preprocessing === ###



def loadStory(filename):
    """
        Loads tory Raw as well as getting read of labels
    """
    print(filename)
    f = open('.\\GeneratedStories\\' + filename,'r',encoding ='utf-8')
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




import nltk

def preprocess(text,lang):
    from nltk.tokenize import word_tokenize

    """
        Preprocessing, and tokenizing words with NLTK, as well as removing polish stopwords
        Polish stopwords are from external file.
        Then filtering Non-word characters.
    """

    # Tokenizing
    tokens = word_tokenize(text,language=lang)

    #Removing Stopwords
    if lang == "polish":
        f = open("./polish.stopwords.txt", "r", encoding='utf-8')
    else:
        f = open("./english.stopwords.txt", "r", encoding='utf-8')

    plstopwords = f.read().split("\n")
    filtered_tokens = [word for word in tokens if word.lower() not in plstopwords]

    # Filtering Non-words
    filtered_tokens = [word for word in filtered_tokens if word.lower() not in [',','.',':',';','?','!']]
    return filtered_tokens

def dictStemmer(text,lang):
    """
        For  Stemming of a WORD in Polish Or in English
    """
    from assessmentFunctions import preprocess
    tokens = preprocess(text,lang)

    if lang == "polish":
   
        ### Properly Stem and Lemmatize The Words:
        import morfeusz2 
        morf = morfeusz2.Morfeusz()

        stemmedStory = []
        for words in (tokens): 
            analysis = morf.analyse(words) 
            stemmedStory.append(analysis[len(analysis)-1][2][1])

        return stemmedStory
    else:
        from nltk.stem import PorterStemmer
        ps = PorterStemmer()
        stemmedStory = [ps.stem(word) for word in tokens]
        return stemmedStory

#### === POS Tagging === ###

def tagStoryPolish(Story1):
    """
        Part of Speech Tagger for Polish language using Morfeusz2
    """
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

    tokens = word_tokenize(" ".join(Story1),language='polish') # Word NLTK tokenizer 
    tokens
    SpeechTagged = {};
    for word in tokens:
        if word != '' and word != '\n':
            analysis = morf.analyse(word)
            dd = analysis[0][2][2].split(":")[0]
            #print(f"{analysis[0][2][0]} - {pos_abbrev[dd]}")
            if pos_abbrev[dd] not in list(SpeechTagged.keys()):
                SpeechTagged[pos_abbrev[dd]] = [analysis[0][2][0]]
            else:
                SpeechTagged[pos_abbrev[dd]].append(analysis[0][2][0])
    return SpeechTagged



def tagStoryEnglish(Story1):
    import spacy
    nlp = spacy.load("en_core_web_sm")

    """
        POS tag English words in a Story using SpaCy!
    """
    pos_abbrev = { # Add abbreviation to have a synergy with Morfeusz2 Output
        "NOUN": "N",
        "PROPN": "N",
        "ADJ": "ADJ",
        "ADV": "ADV",
        "NUM": "NUM",
        "VERB": "V",
        "AUX": "V",
        "PRON": "PRON",
        "ADP": "PREP",
        "CCONJ": "CONJ",
        "SCONJ": "CONJ",
        "PART": "PART",
        "INTJ": "INTERJ",
        "PUNCT": "PUNC",
        "SYM": "SYM",
        "X": "UNK",
        "DET": "DET"
    }

    doc = nlp(" ".join(Story1))

    SpeechTagged = {}

    for token in doc:
        if token.text.strip() == "":
            continue
        tag = pos_abbrev.get(token.pos_, "UNK")

        #print(f"{token.lemma_} - {tag}")

        if tag not in SpeechTagged:
            SpeechTagged[tag] = [token.lemma_]
        else:
            SpeechTagged[tag].append(token.lemma_)

    return SpeechTagged




##################################
####### READABILITY SCORES #######
##################################

def regex_pol_ipa():

    """Returns compiled regular expression that matches
    overlapping syllables in English words transcribed to IPA
    """
    import regex
    
    vowels = 'aɛiɨɔu' 
    special_vowels = "|".join(('ɛ̃','ɔ̃'))
    consonants = 'ɕʑʐɣɲʂxwlmvpɡŋszbkdrnjtf' ## To CHECK
    special_consonants = "|".join(('d͡z', 'd͡ʑ', 'd͡ʐ', 't͡ɕ', 't͡s', 't͡ʂ', 'ɡʲ', 'kʲ')) # needs to copy 'xʲ' and 'j̃' from real examples when I encounter them

    # syllable starts after beginning of word or previous vowel
    # consists of a vowel preceded/followed by 0 or more consonsants
    # ends before next vowel/diphtongue or end of word
    pattern_syllable = f"""(?=  # ? Look ahead to make sure to match a substring without removing it for later checkes
                           (?<=[{vowels}]|{special_vowels}|^) # ? to not return the result yet | <: says look behind 
                           ([{consonants}|{special_consonants}]* (?:[{vowels}]|{special_vowels}) [{consonants}|{special_consonants}]*)
                           (?:[{vowels}]|{special_vowels}|$)
                           ) 
                        """

    # add pattern to match words made of only consonants
    pattern_consonant = f"""(?<=^)(?:[{consonants}|{special_consonants}])+(?:$)"""

    syllables = regex.compile(pattern_syllable, regex.VERBOSE)
    consonants = regex.compile(pattern_consonant, regex.VERBOSE)

    return syllables, consonants



def count_syllables(word,lang="polish",cmu_dict=None):
    import re
    """Count syllables in Polish by correctly handling vowel clusters and glides."""

    word = word.lower().strip()

    if not word:
        return 0
    word = re.sub(r"[^a-z']", "", word)     # Remove punctuation

    
    if lang == "polish":
        # Polish vowels and semivowels/glides
        glidy = "łw"  # Glides to watch out for: 
                    # Glides are something that "glides" thorugh language so consonatns taht are a tad bit different
                    # they Glide between vowelas NOT starting a new SYllable
        # Step 1: Identify potential syllable splits using consonant-vowel structure
        # So we check wehther there is a Consonant BEFORE the vowel '[^aeiouyąęó]*, then we match the vowerl 1 or more
        # and lastly we get any consonant WITHOUT glides 
        wzorzec_sylab = re.findall(r'[^aeiouyąęó]*[aeiouyąęó]+[^aeiouyąęółw]*', word.lower())
        
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
    else: # TO DESCRIBE Wy it works!
        if word in cmu_dict:
            # Take first pronunciation
            pronunciation = cmu_dict[word][0]
            return len([phoneme for phoneme in pronunciation if phoneme[-1].isdigit()])

        # Fallback heuristic
        vowels = "aeiouy"
        count = 0
        prev_vowel = False

        for char in word:
            is_vowel = char in vowels
            if is_vowel and not prev_vowel:
                count += 1
            prev_vowel = is_vowel

        # Silent e
        if word.endswith("e") and count > 1:
            count -= 1

        return max(count, 1)


def FOGScore(text,lang,CMU_DICT):
    """ Computing Gunning-Fog Index. It is an estimation of Readability:
    
    0.4 * number of words/ number of sentences + 100* number of Words with more than 3 syllables / number of words 
    """
    from re import split,sub
    globalSylableCount = 0
    text = text.replace('\n',' ') # Remove new Lines

    # Get sentences (by splitting by . !? )
    sentences = split(r'[.!?]', text)
    sentences = [sentence.strip() for sentence in sentences if sentence] # strip empty sentences

    # get Words by splitting through " ", with removing of empty sequences
    words = dictStemmer(text,lang)
    # Ger Syllable count
    for i in words:
        if count_syllables(i,lang,CMU_DICT) >3:
            globalSylableCount += 1
    return 0.4*(len(words)/len(sentences) + 100*globalSylableCount/len(words))

def pisarekIndex(text,lang,CMU_DICT):
    
    """ Computing Pisarek Index. It is an estimation of Readability:
    206 - 60*(number of syllables / number of words) - number of words / number of sentences
    """
    globalSylableCount = 0
    for i in text.split(" "):
        globalSylableCount += count_syllables(i,lang,CMU_DICT)
    return 206 - 60*(globalSylableCount/len(text.split(" "))) - len(text.split(" "))/len(text.split("."))




##################################
####### LEXICAL DIVERSITY  #######
##################################


# Calculate lexical diversity
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


###############################
####### SHANNON ENTROP  #######
###############################



import numpy as np

def shannon(tokenList):
    from nltk.probability import FreqDist

    """
        Calculates Shannon Entropy, by using Frequency distribution of words in texts. 
        Requires preprocessed (Removed Stopwords, tokenized) text as a list of words. 
        First a frequency Distribution is gathered with list of frequencies Freq, then Entropy is caluculated with a formula:

                sum( N of all tokens/ F * log( N of all Tokens / F ))

        The reversed probability here gives us a positive outcome, not requiring - at the beginning
    """
    # Gives Frequency Distribution of Tokens as a Dictionary
    freq_dist = FreqDist(tokenList)
    freqs = []

    # Since we want a global entropy, we get list of Values
    for word in freq_dist:
        freqs.append(freq_dist[word])

    # to compute probability we simply divide each WordCount with tokens length 
    return sum(np.array(freqs)/len(tokenList) * np.log2(len(tokenList) / np.array(freqs)))



########################################
####### COMPARISON BETWEEN TEXTS #######
########################################

###### NOVELTY SCORE
def noveltyScore(tokens,corpus,model_name='.\\data\\cc.pl.300.bin'):
    import fasttext
    import fasttext.util

    model = fasttext.load_model(model_name)

    corpusDistance =  DG(tokens,model)
    storyDistance = DG(corpus,model)
    novelty =  2*abs(corpusDistance - storyDistance)

    return novelty


def DG(tokens,model):
    import fasttext
    import fasttext.util
    import numpy as np
    from sklearn.metrics.pairwise import cosine_similarity

    corpus_embedding =  [model.get_word_vector(word) for word in tokens]

    ### Memory Efficiency Improvement: For Loop:
    corpusCosSim_sum = np.zeros((len(corpus_embedding)))
    for i in range(len(corpus_embedding)):
        corpusCosSim_sum[i] = np.sum(1-cosine_similarity(corpus_embedding,corpus_embedding[i].reshape(1,-1)))

    np.fill_diagonal(corpusCosSim_sum, 0)
    corpusDistance = np.sum(corpusCosSim_sum)/(corpusCosSim_sum.shape[0] * (corpusCosSim_sum.shape[0] - 1))
    return corpusDistance

    ##### COSINE SIMILARITY

def cosineSim(text1,text2,lang):
    """
        Calculates Cosine Similarity between two texts. Can be used as a proxy to assess which kind of text to choose.
    
    """

    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity

    #Preprocess Words - Tokenize, and remove Stopwords
    tokens1 = preprocess(text1,lang)
    tokens2 = preprocess(text2,lang)

    # Create the TF-IDF vectors
    vectorizer = TfidfVectorizer()
    vector1 = vectorizer.fit_transform(tokens1)
    vector2 = vectorizer.transform(tokens2)

    # Mean Similarity ( between each word)
    similarity = np.sum(sum(cosine_similarity(vector1, vector2)))/np.shape(vector2)[0]
    return similarity

def bertCosineSim(text,model_name):
    """
    Computes the average pairwise cosine dissimilarity between token embeddings
    in a given Polish text using the 'dkleczek/bert-base-polish-uncased-v1' BERT model.

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
    from transformers import BertTokenizer, BertModel
    import torch

    # Load tokenizer and model (official version)
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertModel.from_pretrained(model_name,output_hidden_states=True) # Bert Does not give us Hidden States Unless Specifically Asked to

    ### Tokenize without return_offsets_mapping
    tokens = tokenizer(text, return_tensors="pt", padding=True, truncation=True,  max_length=300) # Padding and Truncation allows for joinign multiple Fragments without worry about messups

    ### Get Names of Each token to see to what they refer to.
    #tokenNames = tokenizer.tokenize(text)
    #print("Tokens:", tokenNames)

    ### Decoding Tokens
    #print("Token IDs:", tokens["input_ids"])
    #print("Tokens:", [tokenizer.decode([id]) for id in tokens["input_ids"][0]])
    attention_mask = tokens["attention_mask"].cpu().numpy()  # Shape: [batch_size, sequence_length]

    ### Get All Embeddings and Hidden States
    with torch.no_grad():
        outputs = model(**tokens)

        # Hidden states for all 12 layers - for All Layers we got a Tensor
    hidden_states = outputs.hidden_states  # Shape: (12 layers, batch_size, sequence_length, hidden_size) 
        # Batch Size - IF doing one sentence at a Time We got 1....
        # Sequence_length - Number of Langauge Tokens ([CLS] a begin token, Subwords (via Sub separation) - or Punctuations, [SEP] - end token)
        # hidden_size - what is the length of Embedding vector for each word?

    ### WARNING - BERT Considers a "Context" a single input that you provide it with... it may be a Sentence, but a whole thing BUT NO LONGER THAN 512 Tokens


    ### Layer 6 and 7th Focuses on Semantic Information Mostly: et those
    layer1 = hidden_states[6]
    layer2 = hidden_states[7]

    ### Option 1st: Avearaging Stories across Cosine Sim: 
    from sklearn.metrics.pairwise import cosine_similarity
    cosSim = [];
    avgSim = [];

    for i in range(layer1.shape[0]):

        # Average layers 6 and 7 for this fragment
        combined_embeddings = (layer1[i] + layer2[i]) / 2  # Shape: [sequence_length, hidden_size]

        # Mask to select only non-padded embeddings
        non_padded_embeddings = combined_embeddings[attention_mask[i] == 1]  # Shape: [actual_tokens, hidden_size]
        if non_padded_embeddings.shape[0] > 1:  # Ensure there are enough tokens to compute pairwise distances
            cosSim.append(1- cosine_similarity(non_padded_embeddings))  # Shape: [actual_tokens, actual_tokens]

            np.fill_diagonal(cosSim[i], 0)

            avgSim.append(np.sum(cosSim[i])/(cosSim[i].shape[0] * (cosSim[i].shape[0] - 1)))

    return avgSim



def bertSentenceCosineSim(text1, text2, model,aggregation="mean_pairs",mode=1):
    import torch
    from sentence_transformers import util
    """
    Quick BertCosine  Similarity Catered towards Sentence Embeddings. But Works also for Lemmas 
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
    


#####################
### General Tools ###
#####################

def getSubCorpus(source_dir = r"C:\Users\barak\Documents\GitHub\strategicMW\TextGeneration\data\NKJP3" ,destination_dir = ".\\filtered_books"):

    """
        Creates subcorpus from text that involves fairy tales

        TO DO: Make it more generall
    
    """
    from os import walk
    from os.path import join,exists,basename
    from re import compile,sub,DOTALL,IGNORECASE
    from  shutil import copytree
    import xml.etree.ElementTree as ET

    # Path to the extracted corpus
    source_dir = r"C:\Users\barak\Documents\GitHub\strategicMW\TextGeneration\data\NKJP3"  # Path to extracted NKJP corpus
    destination_dir = ".\\filtered_books"  # Destination for filtered books

    # Regex to find the <title> content (handling any whitespace or quotes) 
    #if the DOTALL flag has been specified, this matches any character including a newline.
    title_pattern = compile(r'<title>.*?"(.*?)".*?</title>', DOTALL | IGNORECASE)

    # Iterate through each folder
    for root, dirs, files in walk(source_dir):
        print(root)
        for folder in dirs:
            folder_path = join(root, folder)
            header_file = join(folder_path, "header.xml")

            # Check if header.xml exists
            if exists(header_file):
                try:
                    # Read the file as plain text
                    with open(header_file, 'r', encoding='utf-8') as f:
                        content = f.read()

                    # Search for the <title> tag
                    match = title_pattern.search(content)
                    if match:
                        title_text = match.group(1).strip()

                        if  "opowi" in title_text.lower() or "legend" in title_text.lower() or "baśń" in title_text.lower() or "baśn" in title_text.lower() or "bajk" in title_text.lower() or "nowel" in title_text.lower():
                            print(f"Title: {title_text} (In folder: {folder})")
                            book_dir = join(root,folder)

                            folderTitle  = sub('[:;,.?/\"\']', '', title_text)
                            destination_path = join(destination_dir, basename(folderTitle))
                            print(f"Copying book from: {book_dir} to {destination_path}")
                            
                            # Copy the entire book directory
                            copytree(book_dir, destination_path, dirs_exist_ok=True)
                except Exception as e:
                    print(f"Error processing {header_file}: {e}")

        # Break after first level of iteration (for control). Remove to scan all folders.
        break





def extract_base_forms(file_path):
    import xml.etree.ElementTree as ET
    from os.path import isfile
    namespaces = {
        'tei': 'http://www.tei-c.org/ns/1.0'
    }

    base_forms = []
    if isfile(file_path):

        tree = ET.parse(file_path)

        root = tree.getroot()

        # Find all <f name="base"> and extract the <string> content
        for base in root.findall(".//tei:f[@name='base']/tei:string", namespaces):
            if base.text:
                base_forms.append(base.text.strip())
        return base_forms

