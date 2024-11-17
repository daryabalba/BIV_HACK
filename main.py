from tqdm import tqdm

import pandas as pd
import torch
from torch import Tensor
import os
import nltk
import rarfile
import re
import spacy

nltk.download('stopwords')

from nltk.corpus import stopwords
from transformers import pipeline, FlaubertForSequenceClassification, FlaubertTokenizer


class Preprocessor:
    def __init__(self, df, common_stops, spec_stops):
        self.df = self.replace_sums(self.drop_axes(df.copy()))
        self.stop_words = common_stops
        self.stop_words.extend(spec_stops)

    @staticmethod
    def drop_axes(df: pd.DataFrame) -> pd.DataFrame:
        return df.drop(['date'], axis=1)

    @staticmethod
    def replace_sums(df: pd.DataFrame) -> pd.DataFrame:
        sum_int = []
        for _, val in df['sum'].items():
            sum_int.append(int(re.sub(r'[-\.,]\d\d', '', val)))
        df['sum'] = sum_int
        return df

    @staticmethod
    def put_tags(text: str) -> str:
        # плейсхолдер дат
        text = re.sub(r'\d{1,4}[\./]\d{2}[\./]\d\d{2,4}\s?г?(ода)?\.?', ' DATE_PLACEHOLDER ', text)
        text = re.sub(
            r'\d{1,2}\s?(января|февраля|марта|апреля|мая|июня|июля|августа|сентября|октября|ноября|декабря)\s?\d\d{2,4}\s?г?(ода)?\.?',
            ' DATE_PLACEHOLDER ', text)
        # тэг адреса
        text = re.sub(r'по\sадресу\sг\.[-\w\s]+,\s[-\.\w\s]+,\sд\.\s?[-/\d]+', ' ADDRESS_TAG ', text)
        text = re.sub(r'[^о]г\.\s\s?[-\w]+', ' ADDRESS_TAG ', text)
        text = re.sub(r'\(россия, г\.? \w+\)', ' ADDRESS_TAG ', text)
        # плейсхолдеры счетов, договоров и сумм
        text = re.sub(r'(согл\.)?\s{0,2}[^(ра)]сч\.?[её]?т?[а-я]{0,2}\s№?\s?[-/\.\w\d]+', ' ACCOUNT_PLACEHOLDER ', text)
        # договора
        text = re.sub(r'дог(овор)?[а-я]{0,2}\.?\s?№?[-/\.\w\d]+\d[-/\.\w\d]+', ' CONTRACT_PLACEHOLDER ', text)
        text = re.sub(r'№[-/\.\w\d]+', ' CONTRACT_PLACEHOLDER ', text)
        text = re.sub(r'\d+\.\d+-\w+', ' CONTRACT_PLACEHOLDER ', text)
        text = re.sub(r'сумма\s?\d*[-\.,]?\d*', ' SUM_PLACEHOLDER ', text)
        # тэги с и без ндс
        text = re.sub(r'в\s?т(ом)?\.?\s?ч(исле)?\.?\s?ндс[-\.,\s\d\(\)%]+', ' YES_NDS_TAG ', text)
        text = re.sub(r'без\s(уч)?\.?[её]?(та)?\s?ндс', ' NO_NDS_TAG ', text)
        text = re.sub(r'ндс\sне\sоблагается', ' NO_NDS_TAG ', text)
        # тэг количества
        text = re.sub(r'[\d\s](л|мл|кг|г|мг|м|см|мм|шт)[\.,\s]', ' AMOUNT_TAG ', text)
        return text

    def clean_text(self, text: str) -> str:
        # пробелы перед заглавными для устранения слитности
        text = re.sub(r'([А-Я][а-я])', r' \1', text)
        text = text.lower()
        text = self.put_tags(text)
        # замена точек на пробелов для устранения слитности
        text = re.sub(r'\.', ' ', text)
        # устранение пунктуации
        text = re.sub(r'[-/]', ' ', text)
        text = re.sub(r'[^_\sА-Яа-яA-Za-z]', '', text)
        return text

    @staticmethod
    def remove_placeholders(text: str) -> str:
        return re.sub(r'[A-Z]+_PLACEHOLDER', '', text)

    def clean_and_tokenize(self, texts: list) -> list[str]:
        clean_texts = []
        for text in texts:
            clean_texts.append(re.sub(r'\s+', ' ', self.remove_placeholders(self.clean_text(text))))
        return clean_texts

    @staticmethod
    def lemmatize(texts: list[str]) -> list[list[str]]:
        lemmas = []
        nlp = spacy.load('ru_core_news_sm-3.8.0/ru_core_news_sm/ru_core_news_sm-3.8.0', disable=['parser', 'ner'])
        for text in tqdm(nlp.pipe(texts)):
            lemmas.append([token.lemma_ for token in text])
        return lemmas

    def remove_stop_words(self, tokens: list[list[str]]) -> list[list[str]]:
        clean_tokens = []
        for text in tokens:
            clean_tokens.append([i for i in text if i not in self.stop_words and len(i) > 1])
        return clean_tokens

    def pipeline(self) -> None:
        clean = self.clean_and_tokenize(self.df['text'].to_list())
        tokens = self.lemmatize(clean)
        clean_tokens = self.remove_stop_words(tokens)
        clean_strings = [' '.join(i) for i in clean_tokens]
        self.df['text'] = clean_strings

    def get_df(self) -> pd.DataFrame:
        return self.df


def main():
    df_train = pd.read_csv('data/payments_training.tsv', sep='\t', names=['id', 'date', 'sum', 'text', 'label'])
    df_main = pd.read_csv('data/payments_main.tsv', sep='\t', names=['id', 'date', 'sum', 'text'])

    prep_train = Preprocessor(df_train, stopwords.words('russian'), ['оплата'])
    prep_train.pipeline()
    train_clean = prep_train.get_df()

    prep_main = Preprocessor(df_main, stopwords.words('russian'), ['оплата'])
    prep_main.pipeline()
    main_clean = prep_main.get_df()

    if not os.path.exists('./preprocessed/'):
        os.makedirs('./preprocessed/')

    train_clean.to_csv('preprocessed/train_clean.csv', index=False)
    main_clean.to_csv('preprocessed/main_clean.csv', index=False)

    # with rarfile.RarFile('best_model.rar') as rar:
    #     rar.extractall('best_model')

    model = FlaubertForSequenceClassification.from_pretrained('vchemsmisl/biv_hack_model')
    tokenizer = FlaubertTokenizer.from_pretrained('moctarsmal/bank-transactions-statements-classification')

    classification = pipeline('text-classification',
                              model=model,
                              tokenizer=tokenizer)

    classified_texts = classification(main_clean['text'].tolist())
    labels = [clf['label'] for clf in classified_texts]

    main_clean['labels'] = labels
    df_clean = main_clean.drop(columns=['text', 'sum'], axis=1)
    df_clean.to_csv('output.tsv', sep='\t', index=False, header=False)


if __name__ == '__main__':
    main()
