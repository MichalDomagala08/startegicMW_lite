import numpy as np

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
    tokens1 = preprocess(text1)
    tokens2 = preprocess(text2)

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


