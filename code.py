# Author: Joshua (Jay) Wimhurst
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

########################## PDF TO TEXT CONVERSION ############################

# The original PDF file must be converted into text
def pdfToText(fileToConvert):
    pdfPages = []
    # Iterate through each page in the document
    for page in fileToConvert:
        # Headers and footers also deleted based on falling outside each
        # page's bounding box
        rect = page.rect
        height = 50
        clip = fitz.Rect(20, height, rect.width-20, rect.height-height)
        # Extract the text from each page and add to the empty pdfPages list
        text = page.get_text(clip=clip, flags=fitz.TEXT_PRESERVE_LIGATURES)
        pdfPages.append(text)
    # List of page indices
    pageNumbers = list(range(len(pdfPages)))
    
    # Variable placeholder for contents page
    contentsPages = []    
    # This loop catches every use of the word "Contents" in the main text. Will
    # catch the Table of Contents pages and delete them from the list of page
    # numbers (a specific condition is provided for the Elsevier journals; the
    # other conditions catch use in the main text)
    for i in range(len(pdfPages)):    
        if (re.search(r'\bContents\b', pdfPages[i]) or
            re.search(r'\bCONTENTS\b', pdfPages[i]) or
            re.search(r'\bC O N T E N T S\b', pdfPages[i]) or
            re.search(r'\bTABLEOFCONTENTS\b', pdfPages[i]) or
            re.search(r'\bContents lists available\b'.lower(), pdfPages[i].lower())):
            # If the numbers aren't consecutive, then "Contents" also appeared 
            # in the main text and the loop breaks
            if len(contentsPages) >= 1:
                if i-1 == contentsPages[-1]:
                    contentsPages.append(i)
                else:
                    break
            else:
                contentsPages.append(i)
    # Delete contents page numbers if they only occurred in the main text,
    # assuming that a page number at least one third the size of the total
    # page count is in the main text
    contentsPages = [x for x in contentsPages if (x <= max(pageNumbers)/3)]
    
    # Delete all page numbers prior to (and including) the contents page, but
    # not if it's an Elsevier journal
    if len(contentsPages) > 0 and "Contents lists available" not in pdfPages[max(contentsPages)]:  
        pdfPages = pdfPages[max(contentsPages)+1:]
        pageNumbers = pageNumbers[max(contentsPages)+1:]

    # Same again but this time catching every use of the word "Abstract" in
    # the main text. All page numbers before the page on which the Abstract
    # occurs must be deleted. Usage of the word "abstract" in the main text
    # is skipped
    abstractPages = []
    for i in range(len(pdfPages)):
        if (re.search(r'\bAbstract\b', pdfPages[i]) or
            re.search(r'\bABSTRACT\b', pdfPages[i]) or
            re.search(r'\bA B S T R A C T\b', pdfPages[i]) or
            re.search(r'\ba b s t r a c t\b', pdfPages[i])):
            # Don't save if the word Abstract if it occurs in the second half
            # of the main text
            if i+min(pageNumbers) <= max(pageNumbers)*0.4:
                abstractPages.append(i)

    # Delete all content that comes before the Abstract on the same page,
    # then delete all prior pages
    if len(abstractPages) > 0:
        if "Abstract" in pdfPages[min(abstractPages)]:
            pdfPages[min(abstractPages)] = re.split("Abstract",pdfPages[min(abstractPages)])[1]  
        if "ABSTRACT" in pdfPages[min(abstractPages)]:
            pdfPages[min(abstractPages)] = re.split("ABSTRACT",pdfPages[min(abstractPages)])[1]  
        if "A B S T R A C T" in pdfPages[min(abstractPages)]:
            pdfPages[min(abstractPages)] = re.split("A B S T R A C T",pdfPages[min(abstractPages)])[1]  
        if "a b s t r a c t" in pdfPages[min(abstractPages)]:
            pdfPages[min(abstractPages)] = re.split("a b s t r a c t",pdfPages[min(abstractPages)])[1]  
            
        pdfPages = pdfPages[min(abstractPages):]
        pageNumbers = pageNumbers[min(abstractPages):]
        
    # If there is no abstract, then the same code instead catches every use of
    # "Introduction" in the main text, and deletes page numbers before its
    # first usage after the Contents page. Also must account for the word
    # often appearing in the References list and not as a subheading
    else:
        introPages = []
        for i in range(len(pdfPages)):
            if (re.search(r'\bIntroduction\b', pdfPages[i]) or
                re.search(r'\bINTRODUCTION\b', pdfPages[i]) and not
               re.search(r'\bReferences\b', pdfPages[i]) and not
               re.search(r'\bREFERENCES\b', pdfPages[i])):
               # Introduction may appear in the References list but on a 
               # subsequent page if the list is long enough
               if i+min(pageNumbers) <= max(pageNumbers)*0.8:
                   introPages.append(i)
        # Delete everything before the Intro that's on the same page,
        # then all prior pages
        if len(introPages) > 0:
            if "Introduction" in pdfPages[min(introPages)]:
                pdfPages[min(introPages)] = re.split("Introduction",pdfPages[min(introPages)])[1]
            if "INTRODUCTION" in pdfPages[min(introPages)]:
                pdfPages[min(introPages)] = re.split("INTRODUCTION",pdfPages[min(introPages)])[1]
            pdfPages = pdfPages[min(introPages):]
            pageNumbers = pageNumbers[min(introPages):]
    
    # Next find every occurrence of the word "References" or "Literature Cited"
    # in the main text. Looking to delete all page numbers after the final 
    # usage of that word/phrase.
    referencePages = []
    for i in range(len(pdfPages)):
        if (re.search(r'\bReferences\b', pdfPages[i]) or 
        re.search(r'\bREFERENCES\b', pdfPages[i]) or 
        re.search(r'\br e f e r e n c e s\b', pdfPages[i]) or  
        re.search(r'\bR E F E R E N C E S\b', pdfPages[i]) or   
        re.search(r'\bCited Literature\b', pdfPages[i]) or
        re.search(r'\bCITED LITERATURE\b', pdfPages[i]) or
        re.search(r'\bLiterature Cited\b', pdfPages[i]) or
        re.search(r'\bLiterature cited\b', pdfPages[i]) or
        re.search(r'\bLITERATURE CITED\b', pdfPages[i]) or
        re.search(r'\bLiterature reviewed\b', pdfPages[i]) or
        re.search(r'\bLITERATURE REVIEWED\b', pdfPages[i]) or
        re.search(r'\bReferences Cited\b', pdfPages[i]) or
        re.search(r'\bREFERENCES CITED\b', pdfPages[i]) or
        re.search(r'\bBibliography\b', pdfPages[i]) or
        re.search(r'\bBIBLIOGRAPHY\b', pdfPages[i]) or
        re.search(r'\bSelect Bibliography\b', pdfPages[i]) or
        re.search(r'\bSELECT BIBLIOGRAPHY\b', pdfPages[i])):
            referencePages.append(i)
            
    # Delete all pages following the title of the References section, assumed 
    # to be the largest recorded page number in the list
    if len(referencePages) > 0:
        pdfPages = pdfPages[:max(referencePages)+1]
        pageNumbers = pageNumbers[:max(referencePages)+1]
        # Delete the contents of the References themselves as well
        if "References" in pdfPages[max(referencePages)]:
            pdfPages[max(referencePages)] = re.split("References",pdfPages[max(referencePages)])[0]
        if "REFERENCES" in pdfPages[max(referencePages)]:
            pdfPages[max(referencePages)] = re.split("REFERENCES",pdfPages[max(referencePages)])[0]
        if "R E F E R E N C E S" in pdfPages[max(referencePages)]:
            pdfPages[max(referencePages)] = re.split("R E F E R E N C E S",pdfPages[max(referencePages)])[0]
        if "r e f e r e n c e s" in pdfPages[max(referencePages)]:
            pdfPages[max(referencePages)] = re.split("r e f e r e n c e s",pdfPages[max(referencePages)])[0]
        if "Cited Literature" in pdfPages[max(referencePages)]:
            pdfPages[max(referencePages)] = re.split("Cited Literature",pdfPages[max(referencePages)])[0]
        if "CITED LITERATURE" in pdfPages[max(referencePages)]:
            pdfPages[max(referencePages)] = re.split("CITED LITERATURE",pdfPages[max(referencePages)])[0]
        if "Literature Cited" in pdfPages[max(referencePages)]:
            pdfPages[max(referencePages)] = re.split("Literature Cited",pdfPages[max(referencePages)])[0]
        if "Literature cited" in pdfPages[max(referencePages)]:
            pdfPages[max(referencePages)] = re.split("Literature cited",pdfPages[max(referencePages)])[0]
        if "LITERATURE CITED" in pdfPages[max(referencePages)]:
            pdfPages[max(referencePages)] = re.split("LITERATURE CITED",pdfPages[max(referencePages)])[0]
        if "References Cited" in pdfPages[max(referencePages)]:
            pdfPages[max(referencePages)] = re.split("References Cited",pdfPages[max(referencePages)])[0]
        if "REFERENCES CITED" in pdfPages[max(referencePages)]:
            pdfPages[max(referencePages)] = re.split("REFERENCES CITED",pdfPages[max(referencePages)])[0]
        if "Bibliography" in pdfPages[max(referencePages)]:
            pdfPages[max(referencePages)] = re.split("Bibliography",pdfPages[max(referencePages)])[0]
        if "BIBLIOGRAPHY" in pdfPages[max(referencePages)]:
            pdfPages[max(referencePages)] = re.split("BIBLIOGRAPHY",pdfPages[max(referencePages)])[0]
        if "Select Bibliography" in pdfPages[max(referencePages)]:
            pdfPages[max(referencePages)] = re.split("Select Bibliography",pdfPages[max(referencePages)])[0]
        if "SELECT BIBLIOGRAPHY" in pdfPages[max(referencePages)]:
            pdfPages[max(referencePages)] = re.split("SELECT BIBLIOGRAPHY",pdfPages[max(referencePages)])[0]

    # Delete all Appendices as well. These are usually removed by deleting
    # everything after the References list, but documents do not always
    # contain References, and sometimes Appendices come first. 
    appendixPages = []
    for i in range(len(pdfPages)):
        if (re.search(r'\bAppendix\b', pdfPages[i]) or
            re.search(r'\bAPPENDIX\b', pdfPages[i])):
            # Similar to "Abstract", sometimes "Appendix" occurs in the main
            # text, so skip its usage by only looking for it toward the end
            # of the document
            if i+min(pageNumbers) >= max(pageNumbers)*0.8:
                appendixPages.append(i)
    # Delete all pages following the first Appendix
    if len(appendixPages) > 0:
        pdfPages = pdfPages[:min(appendixPages)+1]
        pageNumbers = pageNumbers[:min(appendixPages)+1]
        # Delete the contents of the Appendix itself as well
        if "Appendix" in pdfPages[min(appendixPages)]:
            pdfPages[min(appendixPages)] = re.split("Appendix",pdfPages[min(appendixPages)])[0]
        if "APPENDIX" in pdfPages[min(appendixPages)]:
            pdfPages[min(appendixPages)] = re.split("APPENDIX",pdfPages[min(appendixPages)])[0]
    
    # Delete the Acknowledgements as well (UK and US English spellings given).
    # Acknowledgements typically occur late in the text, but their page
    # numbers are ignored if they occur early to avoid main text deletion
    acknowPages = []
    for i in range(len(pdfPages)):
        if (re.search(r'\bAcknowledgements\b', pdfPages[i]) or
        re.search(r'\bACKNOWLEDGEMENTS\b', pdfPages[i]) or
        re.search(r'\bA C K N O W L E D G E M E N T S\b', pdfPages[i]) or
        re.search(r'\bAcknowledgments\b', pdfPages[i]) or
        re.search(r'\bACKNOWLEDGMENTS\b', pdfPages[i]) or
        re.search(r'\bA C K N O W L E D G M E N T S\b', pdfPages[i]) or
        re.search(r'\bAcknowledgement\b', pdfPages[i]) or
        re.search(r'\bACKNOWLEDGEMENT\b', pdfPages[i]) or
        re.search(r'\bA C K N O W L E D G E M E N T\b', pdfPages[i]) or
        re.search(r'\bAcknowledgment\b', pdfPages[i]) or
        re.search(r'\bACKNOWLEDGMENT\b', pdfPages[i]) or
        re.search(r'\bA C K N O W L E D G M E N T\b', pdfPages[i])):
            acknowPages.append(i)
            
    # The contents of the Acknowledgements themselves must also be deleted    
    if len(acknowPages) > 0:
        if "Acknowledgements" in pdfPages[max(acknowPages)]:
            pdfPages[max(acknowPages)] = re.split("Acknowledgements",pdfPages[max(acknowPages)])[0]
        if "ACKNOWLEDGEMENTS" in pdfPages[max(acknowPages)]:
            pdfPages[max(acknowPages)] = re.split("ACKNOWLEDGEMENTS",pdfPages[max(acknowPages)])[0]
        if "Acknowledgments" in pdfPages[max(acknowPages)]:
            pdfPages[max(acknowPages)] = re.split("Acknowledgments",pdfPages[max(acknowPages)])[0]
        if "ACKNOWLEDGMENTS" in pdfPages[max(acknowPages)]:
            pdfPages[max(acknowPages)] = re.split("ACKNOWLEDGMENTS",pdfPages[max(acknowPages)])[0]
        if "Acknowledgement" in pdfPages[max(acknowPages)]:
            pdfPages[max(acknowPages)] = re.split("Acknowledgement",pdfPages[max(acknowPages)])[0]
        if "ACKNOWLEDGEMENT" in pdfPages[max(acknowPages)]:
            pdfPages[max(acknowPages)] = re.split("ACKNOWLEDGEMENT",pdfPages[max(acknowPages)])[0]
        if "Acknowledgment" in pdfPages[max(acknowPages)]:
            pdfPages[max(acknowPages)] = re.split("Acknowledgment",pdfPages[max(acknowPages)])[0]
        if "ACKNOWLEDGMENT" in pdfPages[max(acknowPages)]:
            pdfPages[max(acknowPages)] = re.split("ACKNOWLEDGMENT",pdfPages[max(acknowPages)])[0]
            
    # Take the same approach to Author Contributions sections, deleting their
    # contents if found
    authorPages = []
    for i in range(len(pdfPages)):
        if (re.search(r'\bAuthor Contributions\b', pdfPages[i]) or
        re.search(r'\bAuthor contributions\b', pdfPages[i]) or
        re.search(r'\bAUTHOR CONTRIBUTIONS\b', pdfPages[i]) or
        re.search(r'\bAUTHOR INFORMATION\b', pdfPages[i]) or
        re.search(r'\bAuthor contribution statement\b', pdfPages[i]) or
        re.search(r'\bCRediT authorship contribution statement\b', pdfPages[i])):
            authorPages.append(i)
    if len(authorPages) > 0:
        if "Author Contributions" in pdfPages[max(authorPages)]:
            pdfPages[max(authorPages)] = re.split("Author Contributions",pdfPages[max(authorPages)])[0]
        if "Author contributions" in pdfPages[max(authorPages)]:
            pdfPages[max(authorPages)] = re.split("Author contributions",pdfPages[max(authorPages)])[0]
        if "AUTHOR CONTRIBUTIONS" in pdfPages[max(authorPages)]:
            pdfPages[max(authorPages)] = re.split("AUTHOR CONTRIBUTIONS",pdfPages[max(authorPages)])[0]
        if "AUTHOR INFORMATION" in pdfPages[max(authorPages)]:
            pdfPages[max(authorPages)] = re.split("AUTHOR INFORMATION",pdfPages[max(authorPages)])[0]
        if "Author contribution statement" in pdfPages[max(authorPages)]:
            pdfPages[max(authorPages)] = re.split("Author contribution statement",pdfPages[max(authorPages)])[0]
        if "CRediT authorship contribution statement" in pdfPages[max(authorPages)]:
            pdfPages[max(authorPages)] = re.split("CRediT authorship contribution statement",pdfPages[max(authorPages)])[0]

    # Same again if there exists a Data Availability Statement
    dataPages = []
    for i in range(len(pdfPages)):
        if (re.search(r'\bData availability statement\b', pdfPages[i]) or
        re.search(r'\bData Availability Statement\b', pdfPages[i]) or
        re.search(r'\bDATA AVAILABILITY STATEMENT\b', pdfPages[i])):
            dataPages.append(i)
    if len(dataPages) > 0:
        if "Data availability statement" in pdfPages[max(dataPages)]:
            pdfPages[max(dataPages)] = re.split("Data availability statement",pdfPages[max(dataPages)])[0]
        if "Data Availability Statement" in pdfPages[max(dataPages)]:
            pdfPages[max(dataPages)] = re.split("Data Availability Statement",pdfPages[max(dataPages)])[0]
        if "DATA AVAILABILITY STATEMENT" in pdfPages[max(dataPages)]:
            pdfPages[max(dataPages)] = re.split("DATA AVAILABILITY STATEMENT",pdfPages[max(dataPages)])[0]

    # Same again but if there exists a Declaration of Competing Interest
    declarePages = []
    for i in range(len(pdfPages)):
        if (re.search(r'\bDeclaration of conflicting interests\b', pdfPages[i]) or
        re.search(r'\bDeclaration of Conflicting Interests\b', pdfPages[i]) or
        re.search(r'\bDeclaration of conflicting interest\b', pdfPages[i]) or
        re.search(r'\bDeclaration of Competing Interest\b', pdfPages[i]) or 
        re.search(r'\bDeclaration of Competing interest\b', pdfPages[i]) or 
        re.search(r'\bDeclaration of competing interest\b', pdfPages[i]) or
        re.search(r'\bDeclarations\b', pdfPages[i]) or
        re.search(r'\bDisclosure statement\b', pdfPages[i]) or
        re.search(r'\bConflicts of Interest\b', pdfPages[i]) or
        re.search(r'\bCONFLICT OF INTEREST\b', pdfPages[i])):
            declarePages.append(i)
            
    if len(declarePages) > 0:
        if "Declaration of conflicting interests" in pdfPages[max(declarePages)]:
            pdfPages[max(declarePages)] = re.split("Declaration of conflicting interests",pdfPages[max(declarePages)])[0]
        if "Declaration of Conflicting Interests" in pdfPages[max(declarePages)]:
            pdfPages[max(declarePages)] = re.split("Declaration of Conflicting Interests",pdfPages[max(declarePages)])[0]
        if "Declaration of competing interest" in pdfPages[max(declarePages)]:
            pdfPages[max(declarePages)] = re.split("Declaration of competing interest",pdfPages[max(declarePages)])[0]
        if "Declaration of Competing Interest" in pdfPages[max(declarePages)]:
            pdfPages[max(declarePages)] = re.split("Declaration of Competing Interest",pdfPages[max(declarePages)])[0]
        if "Declaration of Competing interest" in pdfPages[max(declarePages)]:
            pdfPages[max(declarePages)] = re.split("Declaration of Competing interest",pdfPages[max(declarePages)])[0]
        if "Declarations" in pdfPages[max(declarePages)]:
            pdfPages[max(declarePages)] = re.split("Declarations",pdfPages[max(declarePages)])[0]
        if "Disclosure statement" in pdfPages[max(declarePages)]:
            pdfPages[max(declarePages)] = re.split("Disclosure statement",pdfPages[max(declarePages)])[0]
        if "Conflict of Interest" in pdfPages[max(declarePages)]:
            pdfPages[max(declarePages)] = re.split("Conflict of Interest",pdfPages[max(declarePages)])[0]
        if "CONFLICT OF INTEREST" in pdfPages[max(declarePages)]:
            pdfPages[max(declarePages)] = re.split("CONFLICT OF INTEREST",pdfPages[max(declarePages)])[0]

    # Join the pages together as one long string and then split on new line
    # (\n) characters
    extractedText = ", ".join(pdfPages).splitlines()

    return extractedText
    
############################ DELETE UNWANTED LINES ############################

# Lines of the extracted text that do not have a meaning pertinent to the main 
# text content must be deleted. Examples include table rows, stray 
# letters/numbers/characters on their own lines, empty lines, and errors in
# PDF to text extraction
def delUnwantedLines(extractedText):
    
    # Delete leading and trailing whitespace from each line first
    whiteStrip = (x.strip() for x in extractedText)
    extractedText = [line for line in whiteStrip if line]
    
    # Truncate all whitespace on each line to a single space
    extractedText = [" ".join(line.split()) for line in extractedText]
    
    # Delete comma and space at the start of some lines; these indicated 
    # detected new pages for tables and headers when the PDF to text 
    # conversion happened
    extractedText = [line.removeprefix(", ") for line in extractedText]

    # Use list comprehension to delete unwanted lines
    
    # 1) Remove lines only one character in length, likely to be page numbers,
    #    super/subscript characters, and table cell text
    extractedText = [line for line in extractedText if not len(line) == 1]
    
    # 2) Remove lines that consist solely of numbers, since these are also
    #    likely to be page numbers, super/subscript, and cell text
    def isFloat(s):
        try:
            float(s)
            return True
        except ValueError:
            return False
    extractedText = [line for line in extractedText if not isFloat(line)]
    
    # 3) Remove lines containing an "=" sign, since these are almost certainly
    #    equations and thus not essential to the main text's meaning
    extractedText = [line for line in extractedText if "=" not in line]
    
    # 4) Remove lines made up only of numbers, whitespace, and punctuation,
    #    which are likely the start of bullet pointed lists, table cells,
    #    and numbers broken up by commas
    def isSpace(s): 
        if ' ' in s: 
           return True
        else: 
           return False
    extractedText = [line for line in extractedText if not all(x.isdigit() or isSpace(x) or x in string.punctuation for x in line)]

    # 5) Remove any line that contains the following characters, since these
    #    are almost always author affiliations/details and headers/footers that
    #    were missed during the initial extraction
    extractedText = [line for line in extractedText if "doi:" not in 
                      line.lower() and "http" not in line.lower() and "www." 
                      not in line.lower() and "journal of" not in line.lower()
                      and "@" not in line and ".gov" not in line.lower()
                      and ", USA" not in line and "10.10" not in
                      line.lower() and "10.11" not in line.lower() and "©" not
                      in line.lower() and "Department of" not in line.lower() 
                      and "e-mail" not in line.lower()
                      and "fax" not in line.lower()
                      and "Tel:" not in line.lower()
                      and "all rights reserved" not in line.lower()
                      and "phone" not in line.lower()]

    # 6) Remove lines that are duplicates, since these are very likely table
    #    rows from the same/different tables, accidental duplications of the 
    #    main text from the PDF to text conversion, and can also remove any 
    #    remaining headers and footers
    extractedText = [i for n, i in enumerate(extractedText) if i not in extractedText[:n]]
    
    # To finish, PDF documents often use ligature characters, which should be
    # converted into standard Latin characters
    extractedText = [line.replace("ﬂ","fl") for line in extractedText]
    extractedText = [line.replace("ﬀ","ff") for line in extractedText]
    extractedText = [line.replace("ﬁ","fi") for line in extractedText]
    extractedText = [line.replace("ﬃ","ffi") for line in extractedText]
    extractedText = [line.replace("ﬄ","ffl") for line in extractedText]
    # Punctuation should also be standardized for later removal and inclusion
    extractedText = [line.replace("–","-") for line in extractedText]
    extractedText = [line.replace("‐","-") for line in extractedText]
    extractedText = [line.replace("‑","-") for line in extractedText]
    # Do the same thing with accented characters that appear frequently
    extractedText = [line.replace("ñ","n") for line in extractedText]
    extractedText = [line.replace("é","e") for line in extractedText]
    return extractedText

########################### DELETE TEXT INSIDE LINES ##########################

# Parts of text inside each line are now deleted too. This includes numbers,
# text inside parentheses, and any remaining unwanted text
def delInsideLines(extractedText):
    
    # All numbers in each line are deleted first
    extractedText = [re.sub(r'[0-9]+', '', line) for line in extractedText]
    
    # All text inside parentheses is deleted next
    extractedText = [re.sub('\(.*?\)','', line) for line in extractedText]
    # Parentheses often stretch across lines, so delete everything before a
    # closing parenthesis
    for i in range(len(extractedText)):
        if ")" in extractedText[i]:
            extractedText[i] = extractedText[i].split(")")[1]
        if "(" in extractedText[i]:
            extractedText[i] = extractedText[i].split("(")[0]
  
    # Hyphens at the end of lines almost always represent a single word written
    # across two lines. Each time this happens, the two lines are joined
    # together and the hyphen is deleted. Ignore floating hyphens.
    for line in extractedText:
        if line.endswith("-") and not line.endswith(" -"):
            index = extractedText.index(line)
            try:
               extractedText[index] = extractedText[index][:-1] + extractedText[index+1]
               extractedText.remove(extractedText[index+1])
            # Don't attempt if the final line of the main text ends with a hyphen
            except:
                break

    # Any text in each line that isn't a Latin character, period, hyphen, or 
    # whitespace is deleted
    extractedText = [re.sub(r'[^a-zA-Z .-]', '', line) for line in extractedText]
    
    # Delete any remaining lines that are a repeat of another line in the text;
    # this particularly targets repeating table rows once numbers and text in
    # parentheses have been removed
    extractedText = [line for line,elem in itertools.groupby(extractedText)]
    extractedText = '\n'.join(extractedText)    
    
    return extractedText

########################### TOKENIZE AND REMOVE ###############################

# By first defining new lines by sentences and then tokenizing the text, other
# undesired lines and words can be removed, such as stopwords, in-text 
# citations, people's names, and remaining unwanted sections.
def tokenizeAndRemove(extractedText,stopwordsFilePath):
    
    # Split text on periods instead of new line characters, so each list
    # element is now a sentence from the main text
    extractedText = extractedText.replace("\n"," ").split(".")
    
    # Remove any white space that was left behind by the within-line text
    # deletion in the previous function
    extractedText = [" ".join(line.split()) for line in extractedText]
    
    # Delete sentences that are four characters or shorter. These are usually
    # sentences that were either deleted completely by the previous functions, 
    # units, or table cell entries
    extractedText = [line for line in extractedText if not len(line) <= 4]
    
    # Delete sentences that start with a specific word(s) that represents an
    # unwanted section that still exists in the main text
    extractedText = [line for line in extractedText if not line.startswith("All authors have read and agreed")
                      and not line.startswith("All rights reserved")
                      and not line.startswith("ARTICLE HISTORY")
                      and not line.startswith("Author Contributions")
                      and not line.startswith("BioOne sees sustainable")
                      and not line.startswith("Citation")
                      and not line.startswith("Commercial inquiries")
                      and not line.startswith("Contents list available")
                      and not line.startswith("Correspondence to")
                      and not line.startswith("Corresponding author")
                      and not line.startswith("Data Availability Statement")
                      and not line.startswith("Declaration of Competing Interest")
                      and not line.startswith("Declaration of conflicting interest")
                      and not line.startswith("Funding")
                      and not line.startswith("Full Terms")
                      and not line.startswith("Informed Consent Statement")
                      and not line.startswith("Institutional Review Board")
                      and not line.startswith("Journal of")
                      and not line.startswith("Key Points")
                      and not line.startswith("KEY WORDS")
                      and not line.startswith("Key Words")
                      and not line.startswith("Key words")
                      and not line.startswith("Keywords")
                      and not line.startswith("No part of this periodical")
                      and not line.startswith("Open Access")
                      and not line.startswith("Page number")
                      and not line.startswith("Posted online")
                      and not line.startswith("Published in")
                      and not line.startswith("Published online")
                      and not line.startswith("Submit your article")
                      and not line.startswith("Supplemental Material")
                      and not line.startswith("SUPPLEMENTARY MATERIAL")
                      and not line.startswith("Supplementary Information")
                      and not line.startswith("Supporting information")
                      and not line.startswith("This manuscript was submitted on")
                      and not line.startswith("Your use of this PDF")]
 
    # Rejoin and then tokenize the text
    extractedText = ' '.join(extractedText) 
    tokens = WhitespaceTokenizer().tokenize(extractedText)
    
    # If a token ends with a hyphen, the next token is almost always the 
    # remainder of the same word. Join these tokens together
    for token in tokens:
        if token.endswith("-") and token.count("-") == 1 and len(token) > 1:
            index = tokens.index(token)
            try:
                tokens[index] = tokens[index][:-1] + tokens[index+1]
                tokens.remove(tokens[index+1])
            # Don't attempt if the final token ends with a hyphen
            except:
                break
    
    citedList = []
    # If the tokens "et" and "al" appear as subsequent tokens, append them
    # as well as the previous word (the name of the person being cited)
    for i in range(len(tokens)):
        if tokens[i] == "al" and tokens[i-1] == "et":
            citedList.append(tokens[i-2])
            citedList.append(tokens[i-1])
            citedList.append(tokens[i])
    # Delete these in-text citations from the list of tokens
    tokens = [token for token in tokens if token not in citedList]
    
    # A collection of stopwords are added to a list by reading its csv file first
    def readInCsv(csvFile):
            with open(csvFile, 'r', encoding = 'utf-8', errors = "ignore") as fp:
                reader = csv.reader(fp, delimiter = ',', quotechar = '"')
                dataRead = [row for row in reader]
            return dataRead
    def getStopwords():
        stopwords = readInCsv(stopwordsFilePath)
        stopwords = [word[0] for word in stopwords]
        return stopwords
    # Calling the above functions fills the list of stopwords
    stopwords = getStopwords()

    # Remove these stopwords from the tokenized text
    tokens = [token for token in tokens if token not in stopwords]
    
    # The remaining tokens must be lemmatized as the final pre-processing step
    lemmatizer = WordNetLemmatizer()
    
    # Lemmatization is performed on nouns, verbs, adjectives, and
    # adverbs separately
    lemmaN = [lemmatizer.lemmatize(word, 'n') for word in tokens]
    lemmaNV = [lemmatizer.lemmatize(word, 'v') for word in lemmaN]
    lemmaNVA = [lemmatizer.lemmatize(word, 'a') for word in lemmaNV]
    lemmaNVAR = [lemmatizer.lemmatize(word, 'r') for word in lemmaNVA]
  
    text = " ".join(lemmaNVAR)    
    return text

########################## APPEND AND SAVE TEXT ###############################

# Once the text has been pre-processed, it must be added to the Preprocessed
# Text column in the Document Details database
def appendAndSave(savedDatabase,finalTexts):

    # If pre-processing all documents, blank list elements must be added to the 
    # finalTexts list, corresponding to absent texts in the database
    if yesNo == "N":
        for i in range(len(database)):    
            if str(database["Filename"][i]) == "nan":
                finalTexts.insert(i,"")
    # If pre-processing only new documents, add them to the end of the already
    # pre-processed texts
    else:
        finalTexts = database["PreprocessedText"][:-len(finalTexts)].tolist() + finalTexts
    
    # Add the pre-processed text to the database and save the result
    database["PreprocessedText"] = finalTexts
    database.to_csv(savedDatabase)
    
    return database
