####################################
### Neccessary Import Statements ###
####################################
import pandas as pd
from spacy.tokens import Doc
import psycopg2
import test
import re

########################
### Load in the Data ###
########################
def large_data_extractor():
    data = pd.read_csv("../data/GMB_dataset.txt", sep="\t", encoding="latin1").drop(
        columns="Unnamed: 0"
    )
    # Make sure you run this in your "src" folder.

    # Note that this data was downloaded from https://www.kaggle.com/shoumikgoswami/annotated-gmb-corpus/home.
    # Super big shout out to them.

    ###############################
    ### Extract all of entities ###
    ###############################
    # Get a list of all the tags for every single row.
    tags_list = data.Tag.tolist()

    # Now determine where the entities occur.
    list_of_entity_dfs = []
    # this is a list that stores all of the sub-DataFrames of the original DataFrame
    # that make up the entities. This is mainly done just in case you want to keep
    # track of information like the sentence ID of the entity, the POS tags for each
    # token in the entity, etc...
    found_a_B_tag = False
    # We're just starting the search so it makes sense that would be initialized to
    # False.
    starting_index, ending_index = 0, 0
    for index, tag in enumerate(tags_list):
        if not found_a_B_tag and tag == "O":
            # If you're at a point where you've run into a series of "O" tags since you
            # are in the meat of the text.
            pass
        elif "B-" in tag:
            # If you have finally found the beginning of an entity.
            found_a_B_tag = True
            starting_index = index
            # We need to keep track of where to begin the entity.
        if found_a_B_tag and tag == "O":
            # If you have reached the end of the entity. Notice how we are NOT checking
            # for "I"-tags. We don't have to and we choose not to since it allows for
            # flexibility to handle entities that are either comprised of one token or
            # multiple.
            found_a_B_tag = False
            # We have to revert this back to False since we are done with the entity
            # that we found.
            ending_index = index
            # We will be going UP to this index when saving this entity in our list.

            df_to_append = data[starting_index:ending_index:]
            assert df_to_append.iloc[0].Tag[0:2:] == "B-"
            # if the first token of this entity isn't marked at the beginning, then
            # we know that we did something wrong.
            if len(df_to_append) > 1:
                # if we have an entity that is made up of more than token.
                assert df_to_append["Sentence #"].std() == 0
                # this is checking whether or not all of the tokens that comprise the
                # entity are in the same sentence. They have to be since there is no
                # way that an entity can span more than one sentence.
            list_of_entity_dfs.append(df_to_append)

    list_of_entity_strs = [
        entity_compiler(entity_df) for entity_df in list_of_entity_dfs
    ]
    # also compile the tags for each of these entities
    return (data, list_of_entity_strs)

def entity_compiler(entity_df, *args, **kwargs):
    """
    """
    to_return = (None, None, 0)
    ###
    entity_str_to_format = (len(entity_df) - 1) * "{} " + "{}"
    to_return = (
        entity_str_to_format.format(*entity_df.Word.tolist()),
        entity_df.Tag.iloc[0][2::],
        entity_df["Sentence #"].iloc[0],
    )
    return to_return

###########################
### Cleans the HTML Data ###
###########################
def remove_html_tags(html_text):
    # removes anything between <> and any other unneeded html tags
    cleanr = re.compile("<.*?>|&([a-z0-9]+|#[0-9]{1,6}|#x[0-9a-f]{1,6});")
    # cleans the html text
    cleantext = re.sub(cleanr, "", html_text)
    # returns the cleaned version of the text
    return cleantext
# ----------------------------------------------------------------------------------------------------------------------
"""
geo = Geographical Entity
org = Organization
per = Person
gpe = Geopolitical Entity
tim = Time indicator
art = Artifact
eve = Event
nat = Natural Phenomenon
"""

def entity_label_mapper(label):
    """
    Purpose
    -------

    Parameters
    ----------

    Returns
    -------

    References
    ----------
    1. https://spacy.io/api/annotation
    """
    if label == "per":
        mapped_label = "PERSON"
    elif label == "org":
        mapped_label = "ORG"
    elif label == "eve" or label == "nat":
        mapped_label = "EVENT"
    elif label == "geo":
        mapped_label = "LOC"
    elif label == "art":
        mapped_label = "WORK_OF_ART"
    elif label == "tim":
        mapped_label = "TIME"
    elif label == "gpe":
        mapped_label = "GPE"
    return mapped_label


# ---------------------------------------------------------------------------------------------------------------------
def article_doc_creator(words_iter):
    call_next = True
    spaces = []
    token = next(words_iter)[1]
    article_text = ""
    inside_quote = False
    # Concats article text together and determines whether or not their should be spaces in between tokens
    while True:
        article_text += token
        try:
            if call_next:
                next_token = next(words_iter)[1]  # the next row in the series
            call_next = True
            if next_token == '"' and not inside_quote:
                inside_quote = True
                article_text += " " + next_token
                token = next(words_iter)[1]
                token = next(words_iter)[1]
            elif next_token == '"' and inside_quote:
                inside_quote = False
                article_text += next_token + " "
                token = next(words_iter)[1]
            elif next_token not in [",", "'", ":", ".", "-", "'s"] and token not in [
                "'",
                "-",
                '"',
            ]:
                try:
                    next_next_token = next(words_iter)[1]
                    # # the next row in the series
                    if next_next_token == "." and token == ".":
                        spaces.append(False)
                    else:
                        article_text += " "
                    token = next_token
                    next_token = next_next_token
                    call_next = False
                except StopIteration:
                    article_text += " "
                    token = next_token
            else:
                if next_token == "'" and token.endswith("s"):
                    try:
                        article_text += next_token + " "
                        token = next(words_iter)[1]
                        continue
                    except StopIteration:
                        article_text += next_token
                        break
                spaces.append(False)
                token = next_token
        except StopIteration:
            spaces.append(False)
            break
    # returns a doc objects containing all article tokens and entities
    return article_text


# ----------------------------------------------------------------------------------------------------------------------
"""
Inputs:
golden_annotation: A list of tuples or lists that contain the entity text and the label
ex. [(China, GPE), (Tom Hanks, PERSON)]
nlp: SpaCy model

Outputs:
gold_truth_list: A list of all gold truth entities as Span Objects
gold_truth_dict: A dict of all gold truth entities as Span Objects (keys) and corresponding label (value)
"""


def gold_truth_creator(golden_annotation, nlp):
    """
    Purpose
    -------

    Parameters
    ----------

    Returns
    -------

    References
    ----------
    1.
    """
    ent_position = []
    i, j = 0, 0
    spaces = []
    words = []
    # splits annotated entities into individual tokens
    for ann_entity in golden_annotation:
        # strip the annotation
        split_commas = ann_entity[0].replace(",", " , ")
        split_colon = split_commas.replace(":", " : ")
        split_apostrophe = split_colon.replace("'", " ' ")
        split_dash = split_apostrophe.replace("-", " - ")
        split_period = split_dash.replace(".", " . ")
        tokens = split_period.split()
        j += len(tokens)
        token_iter = iter(tokens)
        ent_position.append([i, j])
        i = j
        # Creates the words and spaces lists to be used in creating the annotated data doc
        call_next = True
        token = next(token_iter)
        while True:
            words.append(token)
            try:
                if call_next:
                    next_token = next(token_iter)  # the next row in the series
                call_next = True
                if next_token not in [",", "'", ":", ".", "-", "'s"] and token not in [
                    "'",
                    "-",
                    '"',
                ]:
                    try:
                        next_next_token = next(token_iter)
                        # # the next row in the series
                        if next_next_token == "." and token == ".":
                            spaces.append(False)
                        else:
                            spaces.append(True)
                        token = next_token
                        next_token = next_next_token
                        call_next = False
                    except StopIteration:
                        spaces.append(True)
                        token = next_token
                else:
                    if next_token == "'" and token.endswith("s"):
                        try:
                            spaces.append(False)
                            words.append(next_token)
                            spaces.append(True)
                            token = next(token_iter)
                            continue
                        except StopIteration:
                            spaces.append(False)
                            words.append(next_token)
                            spaces.append(False)
                            break
                    spaces.append(False)
                    token = next_token
            except StopIteration:
                spaces.append(False)
                break
    annotated_data_doc = Doc(nlp.vocab, words=words, spaces=spaces)
    ground_truth_dict = {}
    ground_truth_list = []
    i = 0
    for start, end in ent_position:
        ann_ent = annotated_data_doc[start:end]
        ground_truth_list.append(ann_ent)
        ground_truth_dict[ann_ent] = golden_annotation[i][1]
        i += 1
    return (ground_truth_list, ground_truth_dict)

# ----------------------------------------------------------------------------------------------------------------------
def connect_to_db(database, hostname, port, userid, passwrd):
    # create string
    conn_string = "host={} port={} dbname={} user={} password={}".format(
        hostname, port, database, userid, passwrd
    )
    # connect to the database with the connection string
    conn = psycopg2.connect(conn_string)
    # commits all queries you execute
    conn.autocommit = True
    cursor = conn.cursor()
    return conn, cursor

#-----------------------------------------------------------------------------------------------------------------------

# @st.cache( persist = True, hash_funcs = {preshed.maps.PreshMap : lambda x: hash(x), cymem.cymem.Pool: lambda x: hash(x)} )
def test_clean():
    annotated_entities, article_text, found_entities = test.create_test_text()
    return (annotated_entities, article_text, found_entities)
