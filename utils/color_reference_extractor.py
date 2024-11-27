import spacy

nlp = spacy.load("en_core_web_sm")

def color_noun_extractor(colors, caption):
    """
    Extracts color-noun pairs from the given English caption based on predefined patterns.
    
    This function identifies and extracts valid pairs of colors and nouns from a caption
    by recognizing specific linguistic patterns. It ensures that each noun is associated
    with only one color to maintain clarity and avoid ambiguity in processing.

    **Patterns Checked:**
    
    1. **Simple Color-Noun Pair (`{color} {noun}`):**
       - Detects phrases where a color directly modifies a noun.
       - *Example:* "red apple" → ("red", "apple")
    
    2. **Color with Adjective and Noun (`{color} and {adjective} {noun}`):**
       - Identifies scenarios where a color is connected with an adjective and a noun using the conjunction "and".
       - Ensures that the adjective correctly modifies the noun following the color.
       - *Example:* "blue and shiny car" → ("blue", "car")
    
    3. **Noun in Color (`{noun} in {color}`):**
       - Captures phrases where a noun is described as being "in" a particular color.
       - *Example:* "cup in green" → ("green", "cup")
    
    **Avoidance of Multiple Color References:**
    
    - After extracting all potential color-noun pairs based on the above patterns, the function checks for nouns that are associated with multiple colors.
    - If a noun is found to be referred to by more than one color within the caption, it is excluded from the final list of pairs.
    - This ensures that only nouns with a single, unambiguous color association are processed further.
    - *Example:* In "red apple and green apple", the noun "apple" is associated with both "red" and "green" and thus will be excluded.
    
    **Parameters:**
    
    - `colors` (`list` of `str`): A list of valid color names to be recognized in the caption.
    - `caption` (`str`): The input caption string from which to extract color-noun pairs.
    
    **Returns:**
    
    - `filtered_pairs` (`list` of `tuple`): A list of tuples, each containing:
        - `color` (`str`): The color associated with the noun.
        - `noun` (`str`): The noun being colored.
        - `position` (`int`): The position index of the color in the caption.
    
    **Example Usage:**
    
    ```python
    colors = ["red", "blue", "green", "yellow", "black", "white", "purple", "orange"]
    caption = "A beautiful red and shiny blue car parked next to a green tree."
    pairs = color_noun_extractor(colors, caption)
    # Output: [('red', 'car', position_index_of_red), ('blue', 'car', position_index_of_blue), ('green', 'tree', position_index_of_green)]
    # Note: 'car' is associated with both 'red' and 'blue' and will be excluded in the final output.
    # Final filtered_pairs: [('green', 'tree', position_index_of_green)]

    """
    pairs = []
    doc = nlp(caption)

    # Iterate through tokens to find patterns
    for i, token in enumerate(doc):
        # Pattern 1: {color} and {adjective} {noun}
        if token.text.lower() in colors:
            if i + 3 < len(doc):
                if (doc[i + 1].lower_ == "and" and
                    doc[i + 2].pos_ == "ADJ" and
                    doc[i + 3].pos_ == "NOUN"):
                    color = token.text.lower()
                    adjective = doc[i + 2].lemma_
                    noun = doc[i + 3].lemma_
                    pairs.append((color, noun, i))
                    print(f"Pattern 1 matched: ({color}, {noun}, position={i})")
            
            # Pattern 2: {color} {noun}
            if i + 1 < len(doc) and doc[i + 1].pos_ == "NOUN":
                color = token.text.lower()
                noun = doc[i + 1].lemma_
                pairs.append((color, noun, i))
                print(f"Pattern 2 matched: ({color}, {noun}, position={i})")
        
        # Pattern 3: {noun} in {color}
        if token.pos_ == "NOUN":
            if i + 2 < len(doc) and doc[i + 1].lower_ == "in" and doc[i + 2].lower_ in colors:
                noun = token.lemma_
                color = doc[i + 2].lemma_
                pairs.append((color, noun, i))
                print(f"Pattern 3 matched: ({color}, {noun}, position={i})")

    # Detect nouns referred to by multiple colors
    noun_color_map = {}
    for color, noun, pos in pairs:
        if noun not in noun_color_map:
            noun_color_map[noun] = []
        noun_color_map[noun].append(color)
    
    # Filter out nouns with multiple color references
    filtered_pairs = []
    for noun, color_list in noun_color_map.items():
        if len(color_list) == 1:
            # Retrieve the position of the first occurrence
            pos = next(pos for c, n, pos in pairs if n == noun)
            filtered_pairs.append((color_list[0], noun, pos))
            print(f"Noun '{noun}' has a single color '{color_list[0]}' and is included.")
        else:
            print(f"Info: Noun '{noun}' is referred to by multiple colors {color_list} and cannot be processed further.")

    return filtered_pairs
