#Kian Fattahy 260978774

from sklearn.preprocessing import FunctionTransformer
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.naive_bayes import MultinomialNB
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
import nltk
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt

def fetchNltkTools():
    toolkitList = ['punkt', 'stopwords', 'wordnet', 'averaged_perceptron_tagger']
    for toolkit in toolkitList:
        nltk.download(toolkit)

def retrieveWordnetTag(tagInputSample):
    tagPrefixes = ['J', 'V', 'N', 'R']
    wordnetTags = [nltk.corpus.reader.wordnet.ADJ, nltk.corpus.reader.wordnet.VERB, nltk.corpus.reader.wordnet.NOUN, nltk.corpus.reader.wordnet.ADV]
    detectedTag = None
    #go through and find pos
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

def processLemmatization(sampleTexts):
    lemmatizedOutput = []
    for singleSample in sampleTexts:

        wordSamples = nltk.tokenize.word_tokenize(singleSample.lower())
        positionTags = nltk.pos_tag(wordSamples)
        
        wordnetProcessor = nltk.stem.WordNetLemmatizer()  
        lemmatizedWords = []


        for word, position in positionTags:
                if word.isalpha():  
                    #use the pos to lemmatize appropriately
                    lemmatizedWords.append(wordnetProcessor.lemmatize(word, pos=retrieveWordnetTag(position)))
                    
        lemmatizedOutput.append(' '.join(lemmatizedWords))

    return lemmatizedOutput

def processStemming(sampleTexts):
    outputTexts = []

    for singleSample in sampleTexts:
        wordSamples = nltk.tokenize.word_tokenize(singleSample.lower())
        porterProcessor = nltk.stem.PorterStemmer()  
        processedWords = []
        #stem each word individually
        for word in wordSamples:
                if word.isalpha():  
                    processedWords.append(porterProcessor.stem(word))
                    
        outputTexts.append(' '.join(processedWords))
    return outputTexts

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

    #generate stout and table for the report
    axisObj.set_title('Evaluation Results', fontsize=16)

    plt.tight_layout()
    plt.savefig('pandasDataSetTable.png', bbox_inches='tight')
    plt.close()
    print(pandasDataSet)















#grabbing stopwords for preprocessing
nltkStopWordsList = nltk.corpus.stopwords.words('english')

#preprocessing tools 
stemmingTransformerTool = FunctionTransformer(processStemming, validate=False)
lemmatizingTransformerTool = FunctionTransformer(processLemmatization, validate=False)

def executeModelTraining(dataSetTrainInput, dataSetTestInput, dataSetTrainOutput, dataSetTestOutput):

    preprocessStrategyList = [None, "stem", "lemmatize", "bigrams"]

    #model configurations
    modelConfigurationList = [
        ("SVM", SGDClassifier(loss='hinge', random_state=2, max_iter=1000), {'modelClassifier__alpha': [0.0001, 0.001, 0.01], 'modelClassifier__penalty': ['l1', 'l2']}),
        ("Naive Bayes", MultinomialNB(), {'modelClassifier__alpha': [0.1, 0.5, 1]}),
        ("Logistic Regression", LogisticRegression(random_state=2, max_iter=1000), {'modelClassifier__C': [0.01, 0.1, 1, 10]})
    ]

    evaluationSummaryList = []
    preprocessingTools = {"stem": stemmingTransformerTool, "lemmatize": lemmatizingTransformerTool}
    for preprocessStrategy in preprocessStrategyList:
        if preprocessStrategy in preprocessingTools:
            activePreprocessorTool = preprocessingTools[preprocessStrategy]
        else:
            activePreprocessorTool = None

        #bigram handling
        if preprocessStrategy == "bigrams":
            tokenRangeTuple = (1, 2)
        else:
            tokenRangeTuple = (1, 1)

        #loop through for each model to optimize hyperparameters
        for modelName, modelInstance, modelParameters in modelConfigurationList:
            #conditional pipeline based on preprocessor
            if activePreprocessorTool:
                modelFlowSequence = Pipeline([
                    ('preprocessor', activePreprocessorTool),
                    ('vectorizationTool', CountVectorizer(ngram_range=tokenRangeTuple, stop_words=nltkStopWordsList)),
                    ('tfidfApplication', TfidfTransformer()),
                    ('modelClassifier', modelInstance)
                ])
            else:
                modelFlowSequence = Pipeline([
                    ('vectorizationTool', CountVectorizer(ngram_range=tokenRangeTuple, stop_words=nltkStopWordsList)),
                    ('tfidfApplication', TfidfTransformer()),
                    ('modelClassifier', modelInstance)
                ])

            #gridsearch optimization
            gridOptimizer = GridSearchCV(modelFlowSequence, param_grid=modelParameters, cv=5, scoring='f1', n_jobs=-1)
            gridOptimizer.fit(dataSetTrainInput, dataSetTrainOutput)

            
            optimalParameters = gridOptimizer.best_params_
            optimalCvPerformance = gridOptimizer.best_score_

            refinedModel = gridOptimizer.best_estimator_
            modelPredictions = refinedModel.predict(dataSetTestInput)
            testingPerformance = f1_score(dataSetTestOutput, modelPredictions)

            #results added for this model
            myDict = {
                'Processing Approach': preprocessStrategy if preprocessStrategy else "baseline",
                'Selected Model': modelName,
                'Optimal Parameters': str(optimalParameters),
                'Best CV F1 Score': optimalCvPerformance,
                'Testing F1 Score': testingPerformance
            }

            evaluationSummaryList.append(myDict)

    return evaluationSummaryList


if __name__ == "__main__":
    fetchNltkTools()
    entireInputData, entireOutputData = gatherDataset()
    dataSetTrainInput, dataSetTestInput, dataSetTrainOutput, dataSetTestOutput = train_test_split(entireInputData, entireOutputData, test_size=0.25, random_state=2)
    evaluationSummaryList = executeModelTraining(dataSetTrainInput, dataSetTestInput, dataSetTrainOutput, dataSetTestOutput)
    visualizeEvaluationResults(evaluationSummaryList)
