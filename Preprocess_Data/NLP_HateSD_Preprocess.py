import re
import emoji
import fasttext
from huggingface_hub import hf_hub_download

# Download and load the fastText model for language identification
model_path = hf_hub_download(repo_id="facebook/fasttext-language-identification", filename="model.bin")
model = fasttext.load_model(model_path)

def preprocess_comment(comment):
    """ Function to clean a comment """
    #  Remove punctuation (except necessary ones, ex: f**k, sh!t, @$$)
    comment = comment.replace('\n', ' ')
    comment = re.sub(r'[^\w\s*?!@$&]', '', comment)

    #  Remove emojis
    comment = emoji.replace_emoji(comment, "")

    # Replace numbers with corresponding letters
    number_to_letter_map = {
        '0': 'o',
        '1': 'l',
        '3': 'e',
        '4': 'a',
        '5': 's',
        '7': 't',
        '9': 'g'
    }
    comment = ''.join([number_to_letter_map.get(c, c) if c.isdigit() else c for c in comment])

    return comment


def preprocess_comments(comments):

    def is_english(comment):
        """ Function to filter non-English comments """
        # Get the top 2 language present
        prediction = model.predict(comment, k=2)
        for label in prediction[0]:
            if label == "__label__eng_Latn":
                return True
        return False  # Return False if no Eng identified

    cleaned_comments = []
    for comment in comments:
        # Split the comment into words and check if it contains at least 2 words
        words = comment.split()
        cleaned_comment = preprocess_comment(comment)
        if len(words) >= 2 and is_english(cleaned_comment):
            cleaned_comments.append(cleaned_comment)

    return cleaned_comments