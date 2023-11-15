#Kian Fattahy 260978774
import nltk
import pandas as pd
import matplotlib.pyplot as plt

def fetchNltkTools():
    toolkitList = ['punkt', 'stopwords', 'wordnet', 'averaged_perceptron_tagger']
    for toolkit in toolkitList:
        nltk.download(toolkit)

def retrieveWordnetTag(tagInputSample):
    tagPrefixes = ['J', 'V', 'N', 'R']
    wordnetTags = [nltk.corpus.reader.wordnet.ADJ, nltk.corpus.reader.wordnet.VERB, nltk.corpus.reader.wordnet.NOUN, nltk.corpus.reader.wordnet.ADV]
    detectedTag = None
    for prefix in tagPrefixes:
        for tagType in wordnetTags:
            if tagInputSample.startswith(prefix):
                if tagType == wordnetTags[tagPrefixes.index(prefix)]:
                    detectedTag = tagType
                    break

        if detectedTag:
            break
    if detectedTag:
        return detectedTag
    else:
        return nltk.corpus.reader.wordnet.NOUN

def processStemming(sampleTexts):
    outputTexts = []
    for singleSample in sampleTexts:
        wordSamples = nltk.tokenize.word_tokenize(singleSample.lower())
        porterProcessor = nltk.stem.PorterStemmer()  
        processedWords = []
        for word in wordSamples:
                if word.isalpha():  
                    processedWords.append(porterProcessor.stem(word))
                    
        outputTexts.append(' '.join(processedWords))
    return outputTexts

def processLemmatization(sampleTexts):
    lemmatizedOutput = []
    for singleSample in sampleTexts:
        wordSamples = nltk.tokenize.word_tokenize(singleSample.lower())
        positionTags = nltk.pos_tag(wordSamples)
        
        wordnetProcessor = nltk.stem.WordNetLemmatizer()  
        lemmatizedWords = []


        for word, position in positionTags:
                if word.isalpha():  
                    lemmatizedWords.append(wordnetProcessor.lemmatize(word, pos=retrieveWordnetTag(position)))
                    
        lemmatizedOutput.append(' '.join(lemmatizedWords))
    return lemmatizedOutput

def openDocument(documentPath):
    with open(documentPath, 'r') as doc:
        documentContent = doc.readlines()
    return [line for line in documentContent] 

def gatherDataset():
    realStatements = openDocument("facts.txt")
    fakeStatements = openDocument("fakes.txt")
    
    statementGroup = []
    for realStatement in realStatements:
        statementGroup.append((realStatement, 1))
        
    fakeGroup = []
    for singleFake in fakeStatements:
        fakeGroup.append((singleFake, 0))
    fullDataset = statementGroup + fakeGroup
    return [item[0] for item in fullDataset], [item[1] for item in fullDataset]


def visualizeEvaluationResults(evaluationResultsList):
    pandasDataSet = pd.DataFrame(evaluationResultsList)

    for rowIndex in range(pandasDataSet.shape[0]):

        for colName in pandasDataSet.columns:
            if isinstance(pandasDataSet.at[rowIndex, colName], (int, float)):
                pandasDataSet.at[rowIndex, colName] = "{:.4f}".format(pandasDataSet.at[rowIndex, colName])

    fig, axisObj = plt.subplots(figsize=(12, 8))
    axisObj.axis('off')

    cellTextList = []
    for _, row in pandasDataSet.iterrows():
        currentRowList = []

        for colName in pandasDataSet.columns:
            currentRowList.append(row[colName])
        cellTextList.append(currentRowList)
    tableObj = axisObj.table(cellText=cellTextList, colLabels=pandasDataSet.columns, cellLoc='center', loc='center')
    tableObj.auto_set_font_size(False)
    for _ in range(5):
        tableObj.set_fontsize(12)
    for index in range(len(pandasDataSet.columns)):
        tableObj.auto_set_column_width(col=index)

    axisObj.set_title('Evaluation Results', fontsize=16)

    plt.tight_layout()
    plt.savefig('pandasDataSetTable.png', bbox_inches='tight')
    plt.close()
    print(pandasDataSet)