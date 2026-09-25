# Script Name: Function_Calls
# Author: Joshua (Jay) Wimhurst, Marco Kemperman
# Date Created: 10/17/2023
# Date Last Edited: 25/09/2026

################################ DESCRIPTION ##################################
# Functions in the Preprocessing_and_Topic_Modeling_Functions script are called
# here, first extracting desired main texts iteratively, saving them to the
# database and then performing topic modeling on the desired texts
# NOTE: When adding new texts to Document Details.xlsx, they must be fully
# documented before attempting to pre-process them!
###############################################################################
# Necessary packages
import fitz
from natsort import natsorted

# Change filepath on Line 19 to match where Model Materials is located
filepath = "....."
stopwordsFilePath = filepath + "Stopwords.csv"

def count_chars(value):
    if isinstance(value, list):
        return sum(len(item) for item in value)
    return len(value)

# =============================================================================
#                         PRE-PROCESSING FUNCTIONS
# =============================================================================

# Ask user whether to pre-process text first or go straight to topic modeling
from Preprocessing_and_Topic_Modeling_Functions import preprocessText
yesNo = preprocessText.yesNo

# If the user said yes to pre-processing, then run the functions below
if yesNo == "Y":
    
    # Create the list of PDFs to pre-process, calling on the filepath to the 
    # Document Details database as an argument
    from Preprocessing_and_Topic_Modeling_Functions import pdfFileList
    fileList = pdfFileList(filepath + "Document Details.xlsx")
    print(f"[PDFfiles] Number of files: {len(fileList)}\n", flush=True)
    if len(fileList) == 0:
        print("[PDFfiles] WARNING: No files to process. Exiting.", flush=True)
    else:
        # List to be filled and added to the "Preprocessed Text" column in the
        # Document Details database
        finalTexts = []
            
        # The following functions from the pre-processing script are called 
        # iteratively to fill the empty list above with the main text from each PDF
        for file in natsorted(fileList):
                
            # Convert the PDF document into text, using the filepath to each PDF
            # document as an argument
            from Preprocessing_and_Topic_Modeling_Functions import pdfToText
            print(f"[Text] Processing: {file}", flush=True)
            text = pdfToText(fitz.open(filepath + "PDFs/" + file))
            print(f"[Text]   After pdfToText: {count_chars(text)} chars", flush=True)

            # Removal of any lines of PDF text that do not contribute to the main text
            from Preprocessing_and_Topic_Modeling_Functions import delUnwantedLines
            text = delUnwantedLines(text)
            print(f"[Text]   After delUnwantedLines: {count_chars(text)} chars", flush=True)
                
            # Removal of alphanumeric characters within lines that also do not
            # contribute to the main text
            from Preprocessing_and_Topic_Modeling_Functions import delInsideLines
            text = delInsideLines(text)
            print(f"[Text]   After delInsideLines: {count_chars(text)} chars", flush=True)
                
            # Tokenize the text so that any remaining undesired words and characters
            # can be removed more precisely
            from Preprocessing_and_Topic_Modeling_Functions import tokenizeAndRemove
            text = tokenizeAndRemove(text,stopwordsFilePath)
            print(f"[Text]   After tokenizeAndRemove: {count_chars(text)} chars", flush=True)
                
            # Add the pre-processed text to the empty list (finalTexts)
            finalTexts.append(text)

        # Once the loop ends, pre-processed texts are added and saved to the
        # Document Details database
        from Preprocessing_and_Topic_Modeling_Functions import appendAndSave
        database_fresh = appendAndSave(filepath + "Document Details.xlsx",finalTexts)
            
        # The database is finally reopened from the CSV companion for the LDA
        # algorithm training.
        from Preprocessing_and_Topic_Modeling_Functions import openDocumentDetails
        database_fresh = openDocumentDetails(filepath + "Document Details.csv")
        print("[DocumentDetails] Preprocessing complete!", flush=True)
    
# =============================================================================
#                   LATENT DIRICHLET ALLOCATION FUNCTIONS
# =============================================================================

# # If the user said no to pre-processing, only the openDocumentDetails
# # function from above is used to open the database as a pandas dataframe
else:
    from Preprocessing_and_Topic_Modeling_Functions import openDocumentDetails
    database_fresh = openDocumentDetails(filepath + "Document Details.csv")
    print("[DocumentDetails] Opened without preprocessing.", flush=True)

# Shared analysis steps, regardless of whether preprocessing ran.
from Preprocessing_and_Topic_Modeling_Functions import calculateSpatialScores
spatialScoreTables = calculateSpatialScores(database_fresh, filepath)
from Preprocessing_and_Topic_Modeling_Functions import plotSpatialChoropleths
plotSpatialChoropleths(spatialScoreTables, filepath)
from Preprocessing_and_Topic_Modeling_Functions import analyzeTopicTrends
# Provide terms for topic trend analysis; change as desired
terms = ["terms", "words"]
analyzeTopicTrends(terms, database_fresh, filepath) 

# # Extract the desired texts from the Document Details database based on
# # user input criteria (Province/Country/Continent/Period/All)
from Preprocessing_and_Topic_Modeling_Functions import textSelection
textsForTraining = textSelection(database_fresh)

# Create the corpus that will be used to train the LDA algorithm, also
# specifying the n-gram size with user input
from Preprocessing_and_Topic_Modeling_Functions import createCorpus
trainingCorpus = createCorpus(textsForTraining,filepath)

# # Train the Latent Dirichlet Allocation (LDA) algorithm and provide user 
# # inputs for performing later sensitivity analysis on model output
from Preprocessing_and_Topic_Modeling_Functions import trainLDAAlgorithm
trainedModel = trainLDAAlgorithm(trainingCorpus,filepath)

# Use the trained LDA algorithm to create word clouds that show the frequency 
# of n-grams within topics, assess document-topic densities, and create word
# webs showing the pairwise occurrence of the commonest n-grams within documents
from Preprocessing_and_Topic_Modeling_Functions import evaluateTrainedModel
evaluateTrainedModel(trainedModel,filepath)

# # Write all end-user decisions and other model outputs not presented in 
# # map/chart form to a separate text file
from Preprocessing_and_Topic_Modeling_Functions import writeTextFile
writeTextFile(filepath, yesNo)

# Copy all the model outputs into a sub-folder of their own, with the
# sub-folder's name reflecting the decisions made by the user
from Preprocessing_and_Topic_Modeling_Functions import moveToSubFolder
moveToSubFolder(filepath, yesNo)
