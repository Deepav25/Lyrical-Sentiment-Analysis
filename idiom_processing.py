def extract_idioms(data, idioms):
    idiom_vector = []
    for song in data['LYRICS']:

        phrase_vector = []
        song_length = len(song)
        for phrase in idioms:
            if phrase[1] in song:
                phrase_count = song.count(phrase)
                phrase_vector.append[phrase[1]]
                # phrase_vector.append[phrase[1], phrase[3], phrase[4], phrase[5]]
                song.replace(phrase, ' ', phrase_count)

        # Ideally we would have an algorithm that weighted
        # the volume of idioms in each song to how much
        # sentiment they contribute to the song overall

    idiom_vector.append(phrase_vector)
    assert len(idiom_vector) == len(data)
    data['IDIOMS'] = idiom_vector # Create new column to store data