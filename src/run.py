####################################
### Neccessary Import Statements ###
####################################
from identifier import Error_Identifier
from spacy import displacy
import data_cleaning

# load in the code for our tool.
import spacy

# to instantiate an NLP object.
import psycopg2

# to make a connection to our database.
import streamlit as st

# to create the UI
import pandas as pd
import numpy as np
import random

# to perform data manipulation
import plotly.graph_objects as go
import plotly.express as px

# to create our graphics.

# Creates a cursor object that allows us to create tables and query into existing tables using SQL language
conn, cursor = data_cleaning.connect_to_db(
    database="XXXXXX",
    hostname="XXXXXX",
    port="XXXX",
    userid="XXXXXX",
    passwrd="XXXXXX",
)

#############
### TESTS ###
#############
def test_run(nlp):
    article_url = "www.test2.com"
    article_title = "Don't Get Testy"
    ## Must check whether the article has already been processed first !!!!!!!
    ## if so, return the saved data directly instead of call the error identifier class
    existed = Error_Identifier.connect_bucket.if_existed(article_url)
    '''if existed is not None:
        print("Article already processed !")
        return (existed, Error_Identifier.connect_bucket.get_err_examples())''' #, (error_type_list, article_title, article_text))
    annotated_entities, article_text, found_entities = data_cleaning.test_clean()
    identifier = Error_Identifier(
        nlp, article_text, article_url, article_title, annotated_entities, language_selected, found_entities=found_entities
    )
    error_type_list = identifier.main()
    return (Error_Identifier.identified_errors_table, Error_Identifier.error_examples, (error_type_list, article_title, article_text))


def test_big_run(nlp, force_reprocess=False):
    data, list_of_entity_strs = data_cleaning.large_data_extractor()
    i = 2800
    j = 2850
    reached_end = False
    dataframes = []
    while True:
        if j > data["Sentence #"].max():
            j = int(data["Sentence #"].max()) + 1
            reached_end = True
        # pull the data the corresponds to the chunk that we are looking at.
        chunk_data = data[
            np.logical_and(data["Sentence #"] < j, data["Sentence #"] >= i)
        ]
        words_iter = chunk_data["Word"].iteritems()
        chunk_text = data_cleaning.article_doc_creator(words_iter)
        chunk_ann_ents = [tup[0] for tup in list_of_entity_strs if tup[2] in range(i, j)]
        # check to see if the data that we are looking at has already been analyzed
        # and thus we can just load in the result.
        existed = Error_Identifier.connect_bucket.if_existed("sentence start #: {}".format(i))
        need_instance = True
        if existed is not None and not force_reprocess:
            # if the article has been processed before.
            print("Article already processed !")
            # put the existing data into the dataframe directly
            ## put the existing data into the dataframe directly
            dataframes.append(existed)
            need_instance = False
        if need_instance:
            my_instance = Error_Identifier(
                nlp,
                chunk_text,
                "sentence start #: {}".format(i),
                "sentence end #: {}".format(j - 1),
                chunk_ann_ents,
                language_selected
            )
            my_instance.main()
        i = j
        j += 50
        if reached_end:
            break
    dataframes.append(Error_Identifier.identified_errors_table)
    return pd.concat(dataframes), Error_Identifier.error_examples


##########################
### Evaluation Metrics ###
##########################
def recall(dataframe):
    true_positive = dataframe["tp"].sum()  # true positive score from dataframe
    false_negative = dataframe["fn"].sum()  # false negative score from dataframe
    return true_positive / (true_positive + false_negative)


# Outputs the models overall precision
def precision(dataframe):
    # calculates the total number of true positives
    tp = dataframe["tp"].sum()
    # calculates the total number of false positives
    fp = dataframe["fp"].sum()
    return tp / (tp + fp)


def f_score(df):
    p = precision(df)
    r = recall(df)
    return 2 * (p * r) / (p + r)


######################
### Run Everything ###
######################
st.title("Named Entity Recognition Error Analysis")

LANGUAGES = ['English', 'Janpanese', 'Spanish']
"""
Select the language of your article(s)
"""
language_selected = st.selectbox('', LANGUAGES)


nlp = spacy.load("en_core_web_md")
# This dataframe should be all you need. Hope it helps with testing :)
dataframe, error_examples, displacy_info = test_run(nlp)

"""
Hello there! Good news, the data file was found and everything ran successfully through **Vanilla spaCy**. Would you like to re-process using a different NER model/system?
"""
reprocess_button_obj = st.button(label="Reprocess?")
if reprocess_button_obj:
    # if the user would like to reprocess their shit.
    """
  ***You have indicated that you would like for the article to be re-processed. Please indicate which NER model/system you would like to re-process with.***
  """
    user_input_0 = st.checkbox("Vanilla spaCy")
    user_input_1 = st.checkbox("Other")
    if user_input_1:
        st.text_input("Please enter the name of your method.")
    """
  ***Reprocessing has begun with selected option.***
  """
    dataframe, error_examples = test_big_run(cursor, nlp, force_reprocess=True)
    """
  ***Reprocessing has completed!***
  """

st.header("Analysis")
st.subheader("Summary Statistics")

a = np.array(
    [[dataframe["tp"].sum(), dataframe["fp"].sum()], [dataframe["fn"].sum(), "N/A"]]
)
confusion = pd.DataFrame(
    data=a,
    index=["Predicted Positive", "Predicted Negative"],
    columns=["Actually Positive", "Actually Negative"],
)
st.table(confusion)

p = precision(dataframe)
r = recall(dataframe)
f = f_score(dataframe)

st.write(
    "{} articles were analyzed. Precision = {:.3f}, Recall = {:.3f}, F1 Score = {:.3f}".format(
        len(dataframe.index), p, r, f
    )
)

st.subheader("Error Categorization")
st.write("Total Fragmentations/Concatenations Overall")
labels = ["Fragmentation", "Concatenation"]
frag_count = dataframe["frag"].sum()
concat_count = dataframe["concat"].sum()
values = [frag_count, concat_count]
pie = go.Figure(data=[go.Pie(labels=labels, values=values)])
st.write(pie)

articles = dataframe.index.values

frag_counts = dataframe["frag"].tolist()
concat_counts = dataframe["concat"].tolist()

st.write("The Fragmentation/Concatenation Errors by Article")
fig = go.Figure(
    data=[
        go.Bar(name="Fragmentations", x=articles, y=frag_counts),
        go.Bar(name="Concatenations", x=articles, y=concat_counts),
    ]
)
# Change the bar mode
fig.update_layout(barmode="stack")
st.write(fig)

st.subheader("Error Correlation")
dataframe = dataframe[
    [
        "sos_frag",
        "num_frag",
        "title_colon_frag",
        "title_prefix_frag",
        "sos_concat",
        "noun_ent_concat",
        "conj_adv_concat",
        "interior_ent_concat",
        "contractional_concat",
        "comma_list_concat",
        "comma_ent_frag",
        "sports_concat",
        "athlete_pos_concat",
        "team_score_rank_concat",
        "hyphen_concat",
        "one_hyphen_concat",
        "colon_concat",
        "diseases_frag",
    ]
]
dataframe = dataframe.astype(int)
heat_data = dataframe.corr(method="pearson")
error_types = [
    "SOS Frag",
    "Numerical Frag",
    "Title Colon Frag",
    "Title Prefix Frag",
    "SOS Concat",
    "Noun Entity Concat",
    "Conjunctive Adverb Concat",
    "Interior Entity Concat",
    "Contractional Concat",
    "Comma List Concat",
    "Comma Entity Frag",
    "Sports Concat",
    "Athlete Position Concat",
    "Team Score Rank Concat",
    "Hyphen Concat",
    "One Hyphen Concat",
    "Colon Concat",
    "Disease Frag",
]
fig = px.imshow(heat_data, x=error_types, y=error_types)
fig.update_xaxes(side="top")
st.write(fig)

st.subheader("Article Distribution by Error")
st.write("View the distribution of articles by occurrences of specific errors.")
option = st.selectbox(
    "Which error type would you like to view?",
    (
        "SOS Frag",
        "Numerical Frag",
        "Title Colon Frag",
        "Title Prefix Frag",
        "SOS Concat",
        "Noun Entity Concat",
        "Conjunctive Adverb Concat",
        "Interior Entity Concat",
        "Contractional Concat",
        "Comma List Concat",
        "Comma Entity Frag",
        "Sports Concat",
        "Athlete Position Concat",
        "Team Score Rank Concat",
        "Hyphen Concat",
        "One Hyphen Concat",
        "Colon Concat",
        "Disease Frag",
    ),
)
# option names to dataframe column names
df_option_mapper = {
    "SOS Frag": "sos_frag",
    "Numerical Frag": "num_frag",
    "Title Colon Frag": "title_colon_frag",
    "Title Prefix Frag": "title_prefix_frag",
    "SOS Concat": "sos_concat",
    "Noun Entity Concat": "noun_ent_concat",
    "Conjunctive Adverb Concat": "conj_adv_concat",
    "Interior Entity Concat": "interior_ent_concat",
    "Contractional Concat": "contractional_concat",
    "Comma List Concat": "comma_list_concat",
    "Comma Entity Frag": "comma_ent_frag",
    "Sports Concat": "sports_concat",
    "Athlete Position Concat": "athlete_pos_concat",
    "Team Score Rank Concat": "team_score_rank_concat",
    "Hyphen Concat": "hyphen_concat",
    "One Hyphen Concat": "one_hyphen_concat",
    "Colon Concat": "colon_concat",
    "Disease Frag": "diseases_frag",
}
mapped_error = df_option_mapper[option]
error_series = dataframe[mapped_error]
hist_counts, hist_edges = np.histogram(
    a=error_series, bins=np.arange(error_series.min(), error_series.max() + 2)
)
# see https://numpy.org/doc/stable/reference/generated/numpy.histogram.html for documentation on this function.
# See https://plotly.com/python/bar-charts/
# hist_counts
# hist_edges
# error_series
histo = px.bar(
    data_frame=pd.DataFrame({"counts": hist_counts, "edges": hist_edges[:-1:]}),
    x="edges",
    y="counts",
    opacity=0.65,
    text="counts",
    width=1000,
    height=900,
    labels={"counts": "Num. of Articles", "edges": "Freq. of Error Type"},
)
histo.update_traces(
    marker_color="rgb(195, 62, 227)",
    marker_line_color="rgb(0, 0, 0)",
    marker_line_width=0.75,
)
histo.update_layout(
    font={"family": "Times New Roman", "size": 18, "color": "black"},
    title={
        "text": "Distribution of Error Type Over Articles",
        "y": 0.975,
        "x": 0.5,
        "xanchor": "center",
        "yanchor": "top",
    },
)
st.write(histo)

st.subheader("Error Frequency")
st.write(
    "View the frequency of error by type and hover over to see examples of each error from the article."
)
totals = dataframe.sum()
errors = ["None"] * 18
rand_error_example_ind = int(random.uniform(0, 1) * 3)
i = 0
for error_type in error_examples:
    if len(error_examples[error_type]) == 3:
        errors[i] = error_examples[error_type][rand_error_example_ind]
    elif len(error_examples[error_type]) > 0:
        errors[i] = error_examples[error_type][0]
    i += 1
frame = {
    "Error Type": list(df_option_mapper.keys()),
    "Count": totals,
    "Example": errors,
}
result = pd.DataFrame(frame)
# See https://plotly.com/python/figure-labels/
bar_chart = px.bar(
    frame,
    x="Error Type",
    y="Count",
    hover_data=["Example"],
    opacity=0.65,
    text="Count",
    width=1000,
    height=900,
    labels={"Count": "Total Count"},
)
bar_chart.update_layout(
    xaxis_tickangle=-45,
    font={"family": "Times New Roman", "size": 18, "color": "black"},
    title={
        "text": "Total Counts of Errors Across all Articles",
        "y": 0.975,
        "x": 0.5,
        "xanchor": "center",
        "yanchor": "top",
    },
)
bar_chart.update_traces(
    marker_color="rgb(158, 202, 225)",
    marker_line_color="rgb(8, 48, 107)",
    marker_line_width=0.5,
)
st.write(bar_chart)

entity_list, title, text = displacy_info
ents = []
for entity in entity_list:
    ents.append({"start": entity[0], "end": entity[1], "label": entity[2]})
entities = [{"text": text, "ents": ents, "title": "Example Article: " + title}]

opts = {'colors': {"SOSFRAG": '#ff7a4a', "NUMFRAG": "#ff9670", "TITLECOLONFRAG": "#ffbaa1",
                   "TITLEPREFIXFRAG": "#ffc800", "COMMAENTFRAG": "#ffde66", "DISEASEFRAG": "#ffea9e",
                   "SOSCONCAT": "#61ffe7", "NOUNENTITYCONCAT": "#bafff5", "CONJADVCONCAT": "#00fbff",
                   "INTERIORENTCONCAT": "#d9feff", "CONTRACTIONALCONCAT": "#00ccff",
                   "COMMALISTCONCAT": "#83c4fc", "SPORTSCONCAT": "#d2e9fc", "ATHLETEPOSCONCAT": "#3b59ad",
                   "TEAMSCORECONCAT": "#3369ff", "HYPHENCONCAT": "#8065f7", "ONEHYPHENCONCAT": "#bfb0ff",
                   "COLONCONCAT": "#e7c5fa", "CORRECT": "#50FB19", "MISSING": "#EB3434", "SPURIOUS": "#b7b6b8",
                   "MULTIFRAG": "#ff007f", "MULTICONCAT": "#ff007f", "FRAGMENT": "#FF4500", "CONCAT": "#0073CF"}}
html = displacy.render(entities, style="ent", options = opts, manual=True)
html = html.replace("\n\n","\n")
HTML_WRAPPER = """<div style="display: inline-block; padding: 1rem">{}</div>"""
st.write(HTML_WRAPPER.format(html),unsafe_allow_html=True)
