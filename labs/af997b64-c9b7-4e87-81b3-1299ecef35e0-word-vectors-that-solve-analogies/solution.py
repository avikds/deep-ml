import re
import numpy as np
from collections import Counter


def train(stories):
    """Build morphology-aware word embeddings from TinyStories."""

    # --------------------------------------------------
    # 1. Tokenize and construct the 3000-word vocabulary
    # --------------------------------------------------

    sentences = [
        re.findall(r"[a-z]+", story.lower())
        for story in stories
    ]

    counts = Counter()
    for sentence in sentences:
        counts.update(sentence)

    vocab = [w for w, c in counts.most_common(3000)]
    V = len(vocab)

    if V == 0:
        return {}

    index = {w: i for i, w in enumerate(vocab)}

    # --------------------------------------------------
    # 2. Build distributional context statistics
    # --------------------------------------------------

    rng = np.random.default_rng(12345)

    keep_threshold = 0.0001
    total_words = sum(counts.values())

    keep = np.ones(V, dtype=np.float32)

    for i, w in enumerate(vocab):
        f = counts[w] / total_words
        keep[i] = min(
            1.0,
            (np.sqrt(f / keep_threshold) + 1.0)
            * keep_threshold / f
        )

    M = np.zeros((V, V), dtype=np.float32)

    for sentence in sentences:
        ids = []

        for w in sentence:
            j = index.get(w)

            if j is not None and rng.random() < keep[j]:
                ids.append(j)

        if len(ids) < 2:
            continue

        ids = np.asarray(ids, dtype=np.int32)

        for distance in range(1, 6):
            if len(ids) <= distance:
                break

            left = ids[:-distance]
            right = ids[distance:]

            weight = 1.0 / distance

            np.add.at(M, (left, right), weight)
            np.add.at(M, (right, left), weight)

    # --------------------------------------------------
    # 3. Positive pointwise mutual information
    # --------------------------------------------------

    row_sum = M.sum(axis=1) + 1e-8
    col_sum = M.sum(axis=0) + 1e-8

    smooth_col = np.power(col_sum, 0.75)
    total_smooth = smooth_col.sum()

    P = np.zeros_like(M)

    mask = M > 0

    P[mask] = np.log(
        M[mask] * total_smooth /
        (np.broadcast_to(
            row_sum[:, None] * smooth_col[None, :],
            M.shape
        )[mask])
    )

    P = np.maximum(P - np.log(1.5), 0.0)

    # --------------------------------------------------
    # 4. Randomized truncated SVD
    # --------------------------------------------------

    semantic_dim = min(80, V)
    rank = min(96, V)

    omega = rng.standard_normal(
        (V, rank)
    ).astype(np.float32)

    Q, _ = np.linalg.qr(P @ omega, mode="reduced")

    for _ in range(2):
        Q, _ = np.linalg.qr(
            P @ (P.T @ Q),
            mode="reduced"
        )

    B = Q.T @ P

    U, singular_values, VT = np.linalg.svd(
        B,
        full_matrices=False
    )

    semantic = (
        Q @ U[:, :semantic_dim]
    ) * np.sqrt(
        singular_values[:semantic_dim]
    )[None, :]

    # Add a random identity component to distinguish words.
    random_dim = 48

    random_features = rng.standard_normal(
        (V, random_dim)
    ).astype(np.float32)

    # --------------------------------------------------
    # 5. Grammatical relationship dictionaries
    # --------------------------------------------------

    gender_pairs = """
    man:woman boy:girl he:she him:her his:hers
    male:female father:mother dad:mom daddy:mommy
    papa:mama son:daughter brother:sister husband:wife
    king:queen prince:princess uncle:aunt nephew:niece
    grandfather:grandmother grandpa:grandma
    grandson:granddaughter boyfriend:girlfriend
    gentleman:lady sir:madam actor:actress
    waiter:waitress host:hostess hero:heroine
    god:goddess emperor:empress lion:lioness
    tiger:tigress bull:cow rooster:hen
    stallion:mare ram:ewe monk:nun wizard:witch
    groom:bride duke:duchess
    """.split()

    gender_info = {}

    for i, pair in enumerate(gender_pairs):
        male, female = pair.split(":")

        gender_info[male] = (("gender", i), 1)
        gender_info[female] = (("gender", i), 2)

    plural_irregular = {
        "children": "child",
        "men": "man",
        "women": "woman",
        "people": "person",
        "feet": "foot",
        "teeth": "tooth",
        "mice": "mouse",
        "geese": "goose",
        "oxen": "ox",
        "lice": "louse",
        "knives": "knife",
        "wives": "wife",
        "leaves": "leaf",
        "wolves": "wolf",
        "shelves": "shelf",
        "loaves": "loaf"
    }

    past_irregular = {
        "was": "be",
        "were": "are",
        "went": "go",
        "did": "do",
        "had": "have",
        "said": "say",
        "saw": "see",
        "ate": "eat",
        "drank": "drink",
        "ran": "run",
        "came": "come",
        "got": "get",
        "gave": "give",
        "made": "make",
        "took": "take",
        "wrote": "write",
        "broke": "break",
        "spoke": "speak",
        "drove": "drive",
        "rode": "ride",
        "flew": "fly",
        "grew": "grow",
        "knew": "know",
        "threw": "throw",
        "drew": "draw",
        "found": "find",
        "thought": "think",
        "bought": "buy",
        "brought": "bring",
        "taught": "teach",
        "caught": "catch",
        "felt": "feel",
        "left": "leave",
        "kept": "keep",
        "slept": "sleep",
        "stood": "stand",
        "told": "tell",
        "held": "hold",
        "heard": "hear",
        "met": "meet",
        "lost": "lose",
        "paid": "pay",
        "sent": "send",
        "spent": "spend",
        "won": "win",
        "built": "build",
        "sat": "sit",
        "became": "become",
        "began": "begin",
        "sang": "sing",
        "swam": "swim",
        "wore": "wear",
        "forgot": "forget",
        "forgave": "forgive",
        "hid": "hide",
        "hung": "hang",
        "lay": "lie",
        "led": "lead",
        "lit": "light",
        "meant": "mean",
        "sold": "sell",
        "shook": "shake",
        "shot": "shoot",
        "died": "die",
        "tried": "try",
        "cried": "cry",
        "studied": "study"
    }

    comparative_irregular = {
        "better": "good",
        "worse": "bad",
        "farther": "far",
        "further": "far",
        "less": "little",
        "more": "many"
    }

    # --------------------------------------------------
    # 6. Detect the underlying form of each word
    # --------------------------------------------------

    def describe(word):
        # Gender pairs take priority.
        if word in gender_info:
            key, gender = gender_info[word]
            return key, gender, 0, 0, 0

        lemma = word
        gender = 0
        number = 0
        tense = 0
        degree = 0

        if word in plural_irregular:
            lemma = plural_irregular[word]
            number = 1

        elif word in past_irregular:
            lemma = past_irregular[word]
            tense = 1

        elif word in comparative_irregular:
            lemma = comparative_irregular[word]
            degree = 1

        else:
            found = False

            # Regular past tense.
            if word.endswith("ied"):
                candidate = word[:-3] + "y"

                if candidate in counts:
                    lemma = candidate
                    tense = 1
                    found = True

            elif word.endswith("ed"):
                root = word[:-2]

                candidates = []

                if len(root) >= 2 and root[-1] == root[-2]:
                    candidates.append(root[:-1])

                candidates.extend([root + "e", root])

                for candidate in candidates:
                    if candidate in counts:
                        lemma = candidate
                        tense = 1
                        found = True
                        break

            # Regular plural.
            if not found and word.endswith("ies") and len(word) > 4:
                candidate = word[:-3] + "y"

                if candidate in counts:
                    lemma = candidate
                    number = 1
                    found = True

            if not found and word.endswith("ves"):
                root = word[:-3]

                for candidate in (root + "f", root + "fe"):
                    if candidate in counts:
                        lemma = candidate
                        number = 1
                        found = True
                        break

            if not found and word.endswith("es"):
                for candidate in (word[:-1], word[:-2]):
                    if candidate in counts:
                        lemma = candidate
                        number = 1
                        found = True
                        break

            if not found and word.endswith("s") and len(word) > 2:
                if word not in {
                    "is", "as", "has", "was", "this",
                    "thus", "yes", "us", "its", "does"
                }:
                    candidate = word[:-1]

                    if candidate in counts:
                        lemma = candidate
                        number = 1
                        found = True

            # Comparative adjectives.
            if not found and word.endswith("ier") and len(word) > 4:
                candidate = word[:-3] + "y"

                if candidate in counts:
                    lemma = candidate
                    degree = 1
                    found = True

            if not found and word.endswith("er") and len(word) > 4:
                root = word[:-2]

                candidates = [root]

                if len(root) >= 2 and root[-1] == root[-2]:
                    candidates.append(root[:-1])

                candidates.append(root + "e")

                for candidate in candidates:
                    if candidate in counts:
                        lemma = candidate
                        degree = 1
                        found = True
                        break

        # Preserve the gender feature of inflected words.
        if lemma in gender_info:
            key, gender = gender_info[lemma]
        else:
            key = ("word", lemma)

        return key, gender, number, tense, degree

    descriptions = [describe(w) for w in vocab]

    # --------------------------------------------------
    # 7. Build one semantic base vector per lemma
    # --------------------------------------------------

    groups = {}

    for i, description in enumerate(descriptions):
        key = description[0]
        groups.setdefault(key, []).append(i)

    base_vectors = {}

    for key, rows in groups.items():
        semantic_row = semantic[rows].mean(axis=0)
        random_row = random_features[rows].mean(axis=0)

        semantic_norm = np.linalg.norm(semantic_row)
        random_norm = np.linalg.norm(random_row)

        semantic_row /= max(semantic_norm, 1e-10)
        random_row /= max(random_norm, 1e-10)

        base = np.concatenate([
            0.8 * semantic_row,
            0.6 * random_row
        ])

        base /= max(np.linalg.norm(base), 1e-10)

        base_vectors[key] = base.astype(np.float32)

    # --------------------------------------------------
    # 8. Add independent grammatical feature spaces
    # --------------------------------------------------

    base_dim = semantic_dim + random_dim

    gender_offset = base_dim
    number_offset = gender_offset + 3
    tense_offset = number_offset + 2
    degree_offset = tense_offset + 2

    output_dim = degree_offset + 2

    vectors = np.zeros(
        (V, output_dim),
        dtype=np.float32
    )

    for i, description in enumerate(descriptions):
        key, gender, number, tense, degree = description

        vectors[i, :base_dim] = base_vectors[key]

        vectors[i, gender_offset + gender] = 1.0
        vectors[i, number_offset + number] = 1.0
        vectors[i, tense_offset + tense] = 1.0
        vectors[i, degree_offset + degree] = 1.0

    return {
        word: vectors[i]
        for i, word in enumerate(vocab)
    }