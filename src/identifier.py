####################################
### Neccessary Import Statements ###
####################################
import pandas as pd
import numpy as np
import re
from connect_bucket import Connect_Bucket
import data_cleaning
from spacy.matcher import Matcher

##############################################
### Write the Class That Identifies Errors ###
##############################################
class Error_Identifier:
    """
    Class Purpose
    -------------
    To identify and categorize errors in recognized entity from SpaCy model and output a
    Pandas DF of categorical data associated with error types.

    Attributes
    ----------
    identified_errors_table - (DataFrame)
    error_example - (Dict)

    Features:
    url - (str)

    frag - (int) count of fragmentatation errors found for a given article
    sos_frag - (int) count of sos frag errors for a given article
    num_frag
    title_colon_frag
    title_prefix_frag

    concat
    sos_concat
    noun_ent_concat
    company_product_concat
    conj_adv_concat
    interior_ent_concat
    contractional_concat
    comma_list_concat
    comma_ent_concat
    sports_concat
    athlete_pos_concat
    team_score_rank_concat
    hyphen_concat
    colon_concat
    diseases_frag()

    Init Parameters
    ---------------

    References
    ----------
    1.
    """

    table_schema = [
        'art_title',
        'tp',
        'fp',
        'fn',
        'frag',
        'sos_frag',
        'num_frag',
        'title_colon_frag',
        'title_prefix_frag',
        'concat',
        'sos_concat',
        'noun_ent_concat',
        'conj_adv_concat',
        'interior_ent_concat',
        'contractional_concat',
        'comma_list_concat',
        'comma_ent_frag',
        'sports_concat',
        'athlete_pos_concat',
        'team_score_rank_concat',
        'hyphen_concat',
        'one_hyphen_concat',
        'colon_concat',
        'diseases_frag',
        'spurious',
        'missing'
    ]
    identified_errors_table = pd.DataFrame(columns = table_schema)
    # error_examples = {
    # 	'sos_frag': [],
    # 	'num_frag': [],
    # 	'title_colon_frag': [],
    # 	'title_prefix_frag': [],
    # 	'sos_concat': [],
    # 	'noun_ent_concat': [],
    # 	'conj_adv_concat': [],
    # 	'interior_ent_concat': [],
    # 	'contractional_concat': [],
    # 	'comma_list_concat': [],
    # 	'comma_ent_concat': [],
    # 	'sports_concat': [],
    # 	'athlete_pos_concat': [],
    # 	'team_score_rank_concat': [],
    # 	'hyphen_concat': [],
    # 	'one_hyphen_concat': [],
    # 	'colon_concat': [],
    # 	'diseases_frag': []
    # }
    # creates an instance of Connect_Bucket that can be used to access the s3 bucket
    connect_bucket = Connect_Bucket()
    # directly obtaining the saved error examples
    error_examples = connect_bucket.get_err_examples()
    honorifics = [
        'Mr',
        'Mrs',
        'Miss',
        'Ms',
        'Dr',
        'Professor',
        'Rabbi',
        'Canon',
        'Dame',
        'Chief',
        'Sister',
        'Brother',
        'Reverend',
        'Major',
        'Sir',
        'Lord',
        'Lady',
        'Mx',
        'St',
        'Saint',
        'Cantor',
        'Chancellor',
        'President'
    ]
    # Creates a cursor object that allows us to create tables and query into existing tables using SQL language
    conn, cursor = data_cleaning.connect_to_db(database="XXXXXXX",
                                               hostname="XXXXXXX",
                                               port="XXXX",
                                               userid="XXXXXXXX",
                                               passwrd="XXXXXXXX")

    # ----------------------------------------------------------------------------------------------------------------------

    def __init__(self, nlp, article_text, article_url, article_title, annotated_entities, language, found_entities=None):
        """
        Parameters:
        * NLP SpaCy model
        * Dictionary of annotated entities (keys) and labels (values)
        * Article Text
        * found_entities (list of found entity strings)

        Attributes:
        * self.article_doc: Contains all article tokens (SpaCy Doc Object)
        * self.found_entity_list: List of all entities found by the model (entities are SpaCy Span objects)
        * self.ground_truth_dict: Ground truth dict of entities for article (keys) and labels (values)
        """
        # The title and url for the article, can be used for indexing into dataframe
        self.title = article_title
        self.url = article_url
        self.language = language

        row_schema = {
            'art_title' : article_title,
            'tp': 0,
            'fp': 0,
            'fn': 0,
            'frag': 0,
            'sos_frag': 0,
            'num_frag': 0,
            'title_colon_frag': 0,
            'title_prefix_frag': 0,
            'concat': 0,
            'sos_concat': 0,
            'noun_ent_concat': 0,
            'conj_adv_concat': 0,
            'interior_ent_concat': 0,
            'contractional_concat': 0,
            'comma_list_concat': 0,
            'comma_ent_frag': 0,
            'sports_concat': 0,
            'athlete_pos_concat': 0,
            'team_score_rank_concat': 0,
            'hyphen_concat': 0,
            'one_hyphen_concat': 0,
            'colon_concat': 0,
            'diseases_frag': 0,
            'spurious': 0,
            'missing': 0
        }
        # Adds an additional row of data for new article instance
        article_row = pd.DataFrame(row_schema, columns = row_schema.keys(), index = [article_url])
        Error_Identifier.identified_errors_table = Error_Identifier.identified_errors_table.append(article_row)
        # SpaCy Doc object that contains all tokens for article text
        self.article_doc = nlp(article_text)
        # if found_entities weren't provided then default to finding entities using Vanilla SpaCy trained on
        # en_core_web_md
        if not found_entities:
            # A list of found entities
            self.found_entity_list_minus_mapped = self.article_doc.ents
        # if found entities were provided then find corresponding span objects using found_entity_creator
        else:
            # A list of found entities
            self.found_entity_list_minus_mapped = self.entity_creator(found_entities, nlp, self.article_doc)
        # A sorted list of entities that were found by the model for the given article (List of SpaCy Span Objects)
        self.found_entity_list_sorted = sorted(self.found_entity_list_minus_mapped, key= lambda ent: ent.text)
        # A list of annotated entities (Span Objects)
        self.ground_truth_list = self.entity_creator(annotated_entities, nlp, self.article_doc)
        # A sorted list of annotated entities
        self.ground_truth_sorted = sorted(self.ground_truth_list, key= lambda ann_ent: ann_ent.text)
        # A dictionary of all entities and their corresponding error-type
        self.error_types_dictionary = {}
        for ent in self.found_entity_list_minus_mapped:
            start_index = len(self.article_doc[:ent.start].text)
            end_index = len(self.article_doc[:ent.end].text)
            self.error_types_dictionary[ent] = [start_index, end_index, "SPURIOUS"]
# ----------------------------------------------------------------------------------------------------------------------
    # Kaelan

    def entity_creator(self, entities_list, nlp, doc):
        pass

# ----------------------------------------------------------------------------------------------------------------------
    # Kaelan

    def is_frag(self, ent, possible_ground_truth):
        pass

# ----------------------------------------------------------------------------------------------------------------------
    # Kaelan

    def is_sos_frag(self, ent, ground_truth_ent):
        pass

# ----------------------------------------------------------------------------------------------------------------------
    # Kaelan

    def is_num_frag(self, ent, ground_truth_ent):
        pass

# ----------------------------------------------------------------------------------------------------------------------
    # Kaelan

    def is_title(self, text):
        pass

# ----------------------------------------------------------------------------------------------------------------------
    # Kaelan

    def is_title_colon_frag(self, ent, ground_truth_ent):
       pass

# ----------------------------------------------------------------------------------------------------------------------
    # Kaelan

    def is_title_prefix_frag(self, ent, ground_truth_ent):
       pass


# ----------------------------------------------------------------------------------------------------------------------
    # Sebastian/Kaelan

    def is_concat(self, ent, possible_ground_truth):
        pass

# ----------------------------------------------------------------------------------------------------------------------
    # Sebastian/Kaelan

    def conj_adv_concat(self, ent, ground_truth_ent):
        pass

# ----------------------------------------------------------------------------------------------------------------------
    # Sebastian

    def sos_concat(self, ent, ground_truth_ent):
        pass
# ----------------------------------------------------------------------------------------------------------------------
    # Sebastian

    def athlete_pos_concat(self, ent, ground_truth_ent):
        pass

# ----------------------------------------------------------------------------------------------------------------------
    # Sebastian

    def colon_concat(self, ent, ground_truth_ent):
        pass
# ----------------------------------------------------------------------------------------------------------------------
    # Sebastian

    def comma_list_concat(self, ent, ground_truth_ent):
        pass

# ----------------------------------------------------------------------------------------------------------------------
    # Sophia

    def hyphen_concat(self, ent, ground_truth_ent):
        pass

# ----------------------------------------------------------------------------------------------------------------------
    # Sophia

    def one_hyphen_concat(self, ent, ground_truth_ent):
        pass

# ----------------------------------------------------------------------------------------------------------------------
    # Sophia

    def contractional_concat(self, ent, ground_truth_ent):
        pass

    # ----------------------------------------------------------------------------------------------------------------------
    # Sophia

    def sports_concat(self, ent, ground_truth_ent):
        pass

    # ----------------------------------------------------------------------------------------------------------------------
    # Ivy
    
    def noun_ent_concat(self, ent, ground_truth_ent):
        pass

    # ----------------------------------------------------------------------------------------------------------------------
    # Ivy/Kaelan
    
    def interior_ent_concat(self, ent, similar_truth_ents):
        pass

    # ----------------------------------------------------------------------------------------------------------------------
    # Ivy
    def comma_ent_frag(self, ent, ground_truth_ent):
        pass

    # ----------------------------------------------------------------------------------------------------------------------
    # Ivy
    def team_score_rank_concat(self, ent, ground_truth_ent):
        pass

    # ----------------------------------------------------------------------------------------------------------------------
    ## Ivy

    def diseases_frag(self, ent, ground_truth_ent):
        pass

    # ----------------------------------------------------------------------------------------------------------------------
    # The container function that calls all subsequent identifier functions for all identified entities
    def main(self):
        """
        Purpose
        -------
        This method acts as the engine for the identifier class with several functions:
        1. First it begins by finding all TP pairs of found entities and corresponding annotated entities
        2. For each remaining unmapped (erroneous) found entity, the main function finds all similar annotated entities and passes
        the entity and these similar annotated entities to the top-most layer of our error identification hierarchy (is_frag
        and is_concat)
        3. If a fragment or concatenation is found, the main function passes the now mapped (and fragmented or concatenated)
        entity and it's corresponding annotated entity to the more granular fragmentation or concatenation functions
        (depending on if it is a fragment or concatenation) to more specifically label the error
        4. If the error type was not a fragmentation or a concatenation the main function will label the remaining entities
        and annotated entities as spurious or missing


        Method Parameters
        -----------------

        Returns
        -------

        References
        ----------

        """
        similar_ent_ann = {}
        index = 0
        for ent in self.found_entity_list_sorted:
            while index < len(self.ground_truth_sorted):
                ann_ent = self.ground_truth_sorted[index]
                if ent.text > ann_ent.text:
                    index += 1
                    continue
                elif ent.text == ann_ent.text:
                    self.found_entity_list_minus_mapped.remove(ent)
                    index += 1
                    self.ground_truth_list.remove(ann_ent)
                    Error_Identifier.identified_errors_table.at[self.url, 'tp'] += 1
                    self.error_types_dictionary[ent][2] = 'CORRECT'
                    break
                else:
                    break

        # now look through proposed entities that have NOT been mapped yet.
        for ent in self.found_entity_list_minus_mapped:
            similar_ent_ann[ent] = []
            for ann_ent in self.ground_truth_list:
                # still going through all possible pairs (just in case).
                similarity_threshold = 0.65
                if ent.similarity(
                        ann_ent) > similarity_threshold or ent.text in ann_ent.text or ann_ent.text in ent.text:
                    similar_ent_ann[ent].append(ann_ent)
            frag = self.is_frag(ent, similar_ent_ann[ent])
            if frag[0]:
                '''
                print('fragment error: {}'.format(ent))
                print('Portion missing: {}'.format(frag[1]))
                print('\n')
                '''
                # Call all more granular frag functions
                if self.language == 'English':
                    sos_frag = self.is_sos_frag(ent, frag[1])
                    self.is_title_prefix_frag(ent, frag[1])
                    self.diseases_frag(ent, frag[1])
                    self.is_num_frag(ent, frag[1])
                    self.is_title_colon_frag(ent, frag[1])
                    self.comma_ent_frag(ent, frag[1])
                if self.language == 'Japanese':
                    self.is_num_frag(ent, frag[1])
                    self.is_title_colon_frag(ent, frag[1])
                continue
            concat = self.is_concat(ent, similar_ent_ann[ent])
            if concat[0]:
                '''
                print('concat error: {}'.format(ent))
                print('Portion added: {}'.format(concat[1]))
                print('\n')
                '''
                if self.language == 'English':
                    # All more specific concat functions
                    is_interior_ent_concat = self.interior_ent_concat(ent, similar_ent_ann[ent])
                    self.hyphen_concat(ent, concat[1])
                    self.one_hyphen_concat(ent, concat[1])
                    self.contractional_concat(ent, concat[1])
                    self.sports_concat(ent, concat[1])
                    self.conj_adv_concat(ent, concat[1])
                    self.sos_concat(ent, concat[1])
                    self.athlete_pos_concat(ent, concat[1])
                    self.colon_concat(ent, concat[1])
                    self.comma_list_concat(ent, concat[1])
                    # noun_ent_concat called when the error is not interior_ent_concat err or comma_ent_concat err
                    if not is_interior_ent_concat:
                        is_noun_ent_concat = self.noun_ent_concat(ent, concat[1])
                    # team_score_rank_concat is called when the ent has any ints included
                    if any([char.isdigit() for char in ent.text]):
                        self.team_score_rank_concat(ent, concat[1])
                if self.language == 'Japanese':
                    self.interior_ent_concat(ent, similar_ent_ann[ent])
                    self.colon_concat(ent, concat[1])
                    self.hyphen_concat(ent, concat[1])
                    self.one_hyphen_concat(ent, concat[1])
                    self.noun_ent_concat(ent, concat[1])
                continue

            Error_Identifier.identified_errors_table.at[self.url, 'spurious'] += 1
            Error_Identifier.identified_errors_table.at[self.url, 'fp'] += 1
        print("article not processed before, processing it now")
        Error_Identifier.identified_errors_table.at[self.url, 'missing'] = len(self.ground_truth_list)
        for ann_ent in self.ground_truth_list:
            start_index = len(self.article_doc[:ann_ent.start].text)
            end_index = len(self.article_doc[:ann_ent.end].text)
            self.error_types_dictionary[ann_ent.text] = [start_index, end_index, "MISSING"]
        Error_Identifier.identified_errors_table.at[self.url, 'fn'] = len(self.ground_truth_list)
        # save the data to pickle file in our bucket
        Error_Identifier.connect_bucket.save_identified_errors(self.url,
                                                               Error_Identifier.identified_errors_table.loc[
                                                                   self.url])
        # save the error examples in our bucket
        Error_Identifier.connect_bucket.save_err_examples(Error_Identifier.error_examples)
        return sorted([[start, end, type] for start, end, type in self.error_types_dictionary.values()], key=lambda ent: ent[0])
# ----------------------------------------------------------------------------------------------------------------------
