 Author: Joshua (Jay) Wimhurst
# Date Created: 10/10/2023
# Date Last Edited: 5/7/2025

################################ DESCRIPTION ##################################
# This script contains all pre-processing and topic modeling functions. These
# functions select files for pre-processing, extract main text from each of
# the texts of interest to the user, and finally enlists a Latent Dirichlet
# Allocation algorithm to construct maps of common words and phrases within the
# text. Imperfections in text pre-processing come from inconsistent PDF
# formatting between journal houses and government repositories. However, the
# outcome is a quantitative and visual summary of common topics pertaining to
# socio-environmental system challenges facing of the Mississippi River Basin.
###############################################################################

# Pip install necessary packages that are not standard Google Colab packages
!pip install PyMuPDF
!pip install gensim
!pip install netgraph
!pip install WordCloud

# Necessary modules (some may have to be installed first)
import csv
import fitz
import gensim.corpora as corpora
import itertools
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import os
import pandas as pd
import re
import seaborn as sns
import shutil
import string
from collections import Counter
from gensim import models
from itertools import combinations
from netgraph import Graph, get_circular_layout, get_bundled_edge_paths
from nltk.tokenize import WhitespaceTokenizer
from nltk.stem import WordNetLemmatizer
from operator import itemgetter
from PIL import Image
from sklearn.feature_extraction.text import CountVectorizer
from tqdm import tqdm
from wordcloud import WordCloud

# =============================================================================
#                         PRE-PROCESSING FUNCTIONS
# =============================================================================

########################## RUN TEXT PRE-PROCESSING ############################

# User is asked whether to pre-process any text first, or progress
# immediately to topic modeling
def preprocessText(values,message):
    while True:
        x = input(message)
        if x in values:
            preprocessText.yesNo = x
            break
        else:
            print("Invalid value: options are " + str(values))
    return preprocessText
preprocessText(["Y","N"],'''\nWould you like to pre-process the PDFs first '''
               '''before topic modeling them? (Y/N): \n''')

########################### CREATE PDF FILE LIST ##############################

# Create a list of all PDF file names from which main text will be
# extracted and pre-processed
def pdfFileList(databaseFilepath):

    # User input for whether pre-processing is performed on all or only new PDFs
    # IMPORTANT: Document details must be added to the database first!!!!
    def preprocessNewOnly(values,message):
        while True:
            x = input(message)
            if x in values:
                preprocessNewOnly.yesNo = x
                break
            else:
                print("Invalid value: options are " + str(values))
        return preprocessNewOnly
    preprocessNewOnly(["Y","N"],'''\nWould you like to only pre-process texts '''
                  '''that you haven't added to the database yet? (Y/N): \n''')
    # Assign user input to variable for use in later functions
    global yesNo
    yesNo = preprocessNewOnly.yesNo

    # The Document Details database (authors, year, spatial extent,
    # search criteria, main text) is opened as a pandas dataframe; assigned
    # globally for later use
    global database
    database = pd.read_csv(databaseFilepath, index_col = 0, encoding="ANSI")

    # Only new files that haven't been pre-processed yet have their names kept
    if yesNo == "Y":
        fileList = []
        for i in range(len(database)):
            if str(database["Filename"][i]) != "nan" and str(database["Preprocessed Text"][i]) == "nan":
                fileList.append(database["Filename"][i])

    # Otherwise use all filenames, derived from filepath to the PDFs
    else:
        fileList = database["Filename"].dropna().tolist()
        print("User has decided to pre-process all PDF documents.")

    return fileList