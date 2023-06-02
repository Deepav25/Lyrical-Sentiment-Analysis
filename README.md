# Lyrical-Sentiment-Analysis
This repo is dedicated to the Lyrical Sentiment Analysis group project for COMPSCI760 at the University of Auckland. 

Here is how to use the `idiom_processing.py` file (uses pytorch). The actual `idiom_processing` function takes 2 arguments: the `data` we are going to be using (which is the `reduced_dataset.csv` file that contains 3000 lyrics (2000 training, 500 validation, 500 testing)) and the `idioms` we are going to be using (the `idiom.csv` dataset). With the idioms processed, they are all put into the `reduced_dataset.csv` file for later use. This code looks for idioms in song lyrics, removes them from the songs, and puts them into the same row as the song they came from. Their sentiments are analyzed separately from the song (which now doesn't have idioms) and then both the song and the idioms sentiment is combined to get a more accurate sentiment label. 

The idiom library came from this github: https://github.com/LowriWilliams/Idiom_Sentiment_Analysis
