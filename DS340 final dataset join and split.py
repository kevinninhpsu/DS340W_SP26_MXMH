import pandas as pd
from sklearn.model_selection import train_test_split

# -------------------------
# 1) Load MXMH
# -------------------------
mxmhDf = pd.read_csv("mxmh_survey_results.csv")

# -------------------------
# 2) Load Spotify songs (ONLY ONCE) with selected columns
# -------------------------
spotifyUseCols = [
    "genre", "danceability", "energy", "loudness", "speechiness", "acousticness",
    "instrumentalness", "liveness", "valence", "tempo", "duration_ms", "popularity"
]

songsPath = "spotifydata with genre DS340/songs.csv"

spotifyDf = pd.read_csv(
    songsPath,
    usecols=lambda c: c in spotifyUseCols
)

# -------------------------
# 3) Clean genre columns so they match
# -------------------------
mxmhDf["Fav genre"] = mxmhDf["Fav genre"].astype(str).str.lower().str.strip()
spotifyDf["genre"] = spotifyDf["genre"].astype(str).str.lower().str.strip()

# -------------------------
# 4) Spotify -> genre-level averages
# -------------------------
numericCols = spotifyDf.select_dtypes(include="number").columns.tolist()

spotifyGenreDf = (
    spotifyDf
    .groupby("genre")[numericCols]
    .mean()
    .reset_index()
)

# -------------------------
# 5) Join with MXMH on genre
# -------------------------
mergedDf = mxmhDf.merge(
    spotifyGenreDf,
    left_on="Fav genre",
    right_on="genre",
    how="left"
)

# Drop rows where the genre didn't match
mergedDf = mergedDf.dropna(subset=numericCols).copy()

print("Merged shape:", mergedDf.shape)

# -------------------------
# 6) 70/20/10 split
# -------------------------
trainDf, tempDf = train_test_split(
    mergedDf,
    test_size=0.30,
    random_state=42,
    shuffle=True
)

testDf, valDf = train_test_split(
    tempDf,
    test_size=1/3,   # 10% of total
    random_state=42,
    shuffle=True
)

print("Train/Test/Val sizes:", len(trainDf), len(testDf), len(valDf))

# -------------------------
# 7) Save files (normal filenames)
# -------------------------
trainDf.to_csv("mxmh_spotify_train.csv", index=False)
testDf.to_csv("mxmh_spotify_test.csv", index=False)
valDf.to_csv("mxmh_spotify_validation.csv", index=False)

print("Saved: mxmh_spotify_train.csv, mxmh_spotify_test.csv, mxmh_spotify_validation.csv")