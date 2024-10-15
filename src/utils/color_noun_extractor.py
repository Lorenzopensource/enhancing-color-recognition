import spacy

# Load the spacy model for noun recognition
nlp = spacy.load("en_core_web_sm")

def color_noun_extractor(colors, caption):
    """
    Extracts color-noun pairs from the given caption.
    :param colors: A list of valid colors.
    :param caption: The caption string.
    :return: A list of tuples containing (color, noun, position) for valid pairs.
    """
    pairs = []
    doc = nlp(caption)

    # Check for color-noun pairs in the caption
    for i, token in enumerate(doc):
        if token.text in colors:  # Check if the token is a color
            # Ensure the next token is a noun
            if i < len(doc) - 1 and doc[i + 1].pos_ == "NOUN":
                noun = doc[i + 1].lemma_  # Get the singular version of the noun
                pairs.append((token.text, noun, i))  # Store color, noun, and position

    return pairs  # Return all valid pairs found

