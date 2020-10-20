from text_preprocessing.text_normalizer import normalize_corpus

sample_text = ("US unveils world's most powerful supercomputer, beats China. " 
               "The US has unveiled the world's most powerful supercomputer called 'Summit', " 
               "beating the previous record-holder China's Sunway TaihuLight. With a peak performance "
               "of 200,000 trillion calculations per second, it is over twice as fast as Sunway TaihuLight, "
               "which is capable of 93,000 trillion calculations per second. Summit has 4,608 servers, "
               "which reportedly take up the size of two tennis courts.")

preprocessed_text = normalize_corpus([sample_text])
