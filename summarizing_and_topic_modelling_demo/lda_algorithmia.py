import Algorithmia

from text_format_util import get_text_from_wikipedia, read_book_and_format

# raw_text, formatted_text = get_text_from_wikipedia("https://en.wikipedia.org/wiki/Tsavo_Man-Eaters")
raw_text, formatted_text = read_book_and_format("summarizing_and_topic_modelling/egyptian-history.txt")

input = {
    "docsList": [
        raw_text
    ],
    "mode": "quality"
}
client = Algorithmia.client('sim5PFpwXqCgBemsacGD/Gjn5aY1')
algo = client.algo('nlp/LDA/1.0.0')
algo.set_options(timeout=300)  # optional
print(algo.pipe(input).result)
