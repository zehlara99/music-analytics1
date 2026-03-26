# I Analyzed the 30 Most-Streamed Songs on Spotify — Here's What the Data Actually Says

*A Python-powered deep dive into audio signals, lyrics NLP, and clustering algorithms — applied to the songs that broke streaming records.*

---

## How This Was Built

Before we get to the findings, here's what actually happened under the hood. This isn't a blog post based on opinion — every number below came from running code on the actual audio files and lyrics.

**Step 1 — Define the songs.** We started with the 30 most-streamed songs of all time on Spotify (as of early 2026), from Blinding Lights (5.34B streams) down to Cruel Summer (3.29B).

**Step 2 — Download the audio.** Using `yt-dlp` (a Python library), we searched YouTube for each song and downloaded the audio as MP3 files. No manual clicking — the script searches, finds the top result, and saves it automatically.

**Step 3 — Analyze the sound.** Using `librosa`, a Python audio analysis library, we extracted features from the first 60 seconds of each MP3:
- **Tempo (BPM)** — how fast is the beat?
- **Key** — what musical key is the song in (C, D, A major, etc.)?
- **Energy (RMS)** — how loud/intense is the signal on average?
- **Spectral centroid** — where does the "brightness" of the sound sit?
- **MFCCs** — 13 numbers that describe the timbral texture of the sound (think of it as a sonic fingerprint)
- **Harmonic vs. percussive energy** — how much is melody vs. drums?

**Step 4 — Fetch the lyrics.** Using `lyricsgenius` (Genius API), we downloaded the full lyrics for each song and saved them as text files.

**Step 5 — Analyze the lyrics.** Using `VADER` (from NLTK) — a sentiment analysis tool originally built for social media — we scored every song's lyrics from -1 (extremely negative) to +1 (extremely positive). We also measured word count, vocabulary diversity, and average word length.

**Step 6 — Find patterns.** Using `scikit-learn`, we applied K-means clustering on 16 combined audio + lyric features to let the algorithm group songs by similarity — no genre labels, no human input. Just math.

All of this runs in a single Jupyter notebook (~300 lines of Python). Now, here's what it found.

---

## The Findings

---

### 1. One song is on a completely different planet

![Tempo Comparison](visuals/tempo_comparison.png)

**Blinding Lights — The Weeknd: 5.34 billion streams.**

The #2 song (Shape of You, Ed Sheeran) has 4.83B. That's a 500 million stream gap — bigger than the entire gap between positions #2 and #12. No other song on this list comes close. It's not just the most-streamed song ever. It's statistically alone at the top.

---

### 2. The "sweet spot" tempo for a hit is 118 BPM

![Tempo Comparison Chart](visuals/tempo_comparison.png)

The average tempo across all 30 songs is **118 BPM**. That's the classic pop-dance range — fast enough to feel energetic, slow enough to sing along with. Most songs sit between 86–130 BPM. This isn't a coincidence: human walking pace is ~100 BPM, and music in that zone syncs naturally with how our bodies move.

---

### 3. Ed Sheeran alone owns 10% of this list — in three different genres

Ed Sheeran has three songs in the top 30: Shape of You (#2, 4.83B), Perfect (#9, 3.89B), and Photograph (#28, 3.35B). The remarkable part? When the clustering algorithm ran — with no knowledge of who made what — it placed the three songs in **three separate clusters**: dance-pop, soft indie-rock, and retro/classic. One artist, three completely different sonic identities.

---

### 4. "Perfect" clocks in at 185 BPM — but sounds like a slow ballad

The audio analysis measured Ed Sheeran's *Perfect* at **184.57 BPM** — the highest tempo on the entire list. But it sounds slow and gentle. Why? Beat-tracking algorithms sometimes double-count subdivisions in songs with complex rhythms. The "felt" tempo is closer to 92 BPM. This is a good reminder: the data tells you what the signal looks like, not always what it sounds like.

---

### 5. The most "positive" lyrics belong to a song about luxury and ego

![Sentiment Comparison](visuals/sentiment_comparison.png)

*Starboy* by The Weeknd scored a **VADER sentiment compound of 0.9998** — nearly the maximum possible. The lyrics are dominated by materialism and self-aggrandizement: cars, women, money, fame. The AI reads it as overwhelmingly positive because the words themselves are technically upbeat. Lesson: sentiment analysis measures word polarity, not emotional depth.

---

### 6. The angriest song has 3.79 billion streams

*Believer* by Imagine Dragons scored **-0.9913** — the most negative sentiment in the entire dataset. The song is about pain, adversity, and struggle. Yet it's #11 on the all-time chart. Emotional intensity, not happiness, is what makes people hit replay. The data strongly suggests that *how hard a song hits* matters more than whether it makes you feel good.

---

### 7. Dance Monkey is the most repetitive song — and it worked

*Dance Monkey* by Tones and I has a **lexical diversity of 0.160** — the lowest on the list. That means only 16% of its words are unique. Nearly everything is repeated. 3.41 billion streams later, the earworm strategy is clearly valid. Repetition isn't laziness — it's engineering.

---

### 8. The most "literate" hit is a love song you've probably cried to

*Say You Won't Let Go* by James Arthur has a **lexical diversity of 0.420** — the highest on the list, more than 2.5× the vocabulary variety of Dance Monkey. Thoughtful, wordy songwriting with 3.56 billion streams proves you don't have to dumb it down to connect with a massive audience.

---

### 9. Two songs from completely different artists are musical twins

![Similarity Matrix](visuals/similarity_matrix.png)

*The Night We Met* (Lord Huron) and *Another Love* (Tom Odell) have a **cosine similarity score of 0.775** — the highest pair-wise score in the entire 30×30 similarity matrix. Two sparse, melancholic piano-led songs about loss, written years apart by artists from different countries. The algorithm, looking only at audio features, found them nearly identical. Your playlist algorithm probably already knew this.

---

### 10. A major is the most common key in these hits

![Key Distribution](visuals/key_distribution.png)

4 of the 30 songs are in **A major** — more than any other key. Music theory has long held that A major feels bright and sits comfortably in the human singing range. The data from the top 30 most-streamed songs ever backs this up. If you're writing a hit, start in A.

---

### 11. The quietest song is nearly silent by audio measurements

*Another Love* by Tom Odell has an **RMS energy score of 0.057** — compared to 0.328 for *As It Was* (Harry Styles), the most energetic song on the list. That's nearly **6× less audio energy**. A sparse piano and a quiet vocal. And still: 3.49 billion streams. The data proves that restraint can be just as powerful as volume.

---

### 12. Python found 5 hidden "tribes" in mainstream pop — with no genre labels

![PCA Clusters](visuals/clusters_pca.html)

K-means clustering grouped all 30 songs into 5 distinct clusters using only math — no genre tags, no human input:

| Cluster | Songs |
|---|---|
| **Emotional Ballads** | Someone You Loved, lovely, The Night We Met, Another Love, Say You Won't Let Go, Take Me To Church |
| **Soft Indie Rock** | Sweater Weather, Yellow, Riptide, Perfect, I Wanna Be Yours, Die With A Smile |
| **High-Energy Modern Pop** | Blinding Lights, Starboy, As It Was, BIRDS OF A FEATHER |
| **Dance / Pop** | Shape of You, Cruel Summer, Heat Waves, Believer, Dance Monkey, Closer + 5 more |
| **Retro / Classic** | Every Breath You Take, Photograph, rockstar |

The algorithm essentially reinvented genre classification from raw audio. It's not perfect — but it's surprisingly close to how a human music fan would group these songs.

---

### 13. Billie Eilish lives in two completely different sonic worlds

Billie Eilish has two songs on this list. *lovely* (with Khalid) landed in the **Emotional Ballads cluster** — slow, sparse, intimate. *BIRDS OF A FEATHER* landed in the **High-Energy Modern Pop cluster** alongside Blinding Lights and Starboy. The same artist, placed at opposite ends of the sonic spectrum by the same algorithm. Her range is not just artistically impressive — it's mathematically verified.

---

### 14. A song from 1983 has 3.38 billion Spotify streams

*Every Breath You Take* by The Police was recorded before CDs were mainstream. It has been streaming on a platform that didn't exist until 2006, racking up billions of plays from listeners who weren't born when it was released. The clustering algorithm placed it in the "Retro/Classic" group alongside Post Malone's *rockstar* and Ed Sheeran's *Photograph* — songs that share its mid-tempo, melodic, minimalist DNA. Some sonic formulas are simply timeless.

---

### 15. The Weeknd has two songs in the top 5 — and they sound almost identical

*Blinding Lights* (#1, 5.34B) and *Starboy* (#4, 4.44B) both landed in the same cluster — High-Energy Modern Pop. The cosine similarity matrix confirms they score high on shared features: bright spectral signature, strong percussive energy, mid-high tempo, and maximum positive sentiment. Years apart, different albums, different collaborators (Daft Punk on Starboy). Same sonic DNA. That kind of consistency isn't accidental — it's a brand.

---

## All Visuals

Here's a summary of every chart generated by the analysis:

| Visual | What it shows |
|---|---|
| [Waveforms](visuals/waveforms.png) | Audio waveform for all 30 songs |
| [Spectrograms](visuals/spectrograms.png) | Frequency-over-time heat maps for all songs |
| [Word Clouds](visuals/wordclouds.png) | Most-used words per song |
| [Tempo Comparison](visuals/tempo_comparison.png) | BPM ranked bar chart |
| [Key Distribution](visuals/key_distribution.png) | Which musical keys dominate |
| [Sentiment Comparison](visuals/sentiment_comparison.png) | Positive vs negative per song |
| [Correlation Heatmap](visuals/correlation_heatmap.png) | How all features relate to each other |
| [Similarity Matrix](visuals/similarity_matrix.png) | Which songs sound most alike |
| [Sentiment vs Energy (interactive)](visuals/sentiment_vs_energy.html) | Bubble chart: mood vs intensity |
| [Radar Chart (interactive)](visuals/radar_chart.html) | 6-feature audio profile per song |
| [PCA Clusters (interactive)](visuals/clusters_pca.html) | The 5 song tribes visualized |
| [Streams vs Complexity (interactive)](visuals/streams_vs_complexity.html) | Do simpler songs get more streams? |
| [Scatter Matrix (interactive)](visuals/scatter_matrix.html) | All features vs all features |

---

*All data generated using Python (librosa, NLTK/VADER, scikit-learn, lyricsgenius, plotly). Audio sourced from YouTube via yt-dlp. Lyrics via Genius API. Stream counts from Spotify charts, early 2026.*
