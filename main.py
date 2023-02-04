import argparse
import logging
import os
import sys
import spacy
import srsly
from spacy.tokens import Doc
from spacy.training.iob_utils import spans_from_biluo_tags, offsets_to_biluo_tags

NLP = spacy.load("en_core_web_sm")


def parse_spacy_to_bert_format(source_file: str, result_file: str):
    """
    The function converting dataset from spacy-NER to BERT-NER format
    :param source_file: file contain spacy dataset
    :param result_file: result in bert format
    """
    logging.info(f"Start converting dataset from spacy-NER to BERT-NER format...")
    spacy_data = srsly.read_jsonl(os.getcwd() + '/' + source_file)
    with open(os.getcwd() + '/' + result_file, 'w') as f:
        for data in spacy_data:
            print(len(data['text']))
            doc = NLP.make_doc(data['text'])
            tags = offsets_to_biluo_tags(doc, data['label'])
            entities = spans_from_biluo_tags(doc, tags)
            doc.ents = entities
            for token in doc:
                f.write(f'{token.text} {token.ent_iob_}-{token.ent_type_}\n')
            f.write('\n')
    logging.info(f"work detection algorithm")


def parse_bert_to_spacy_format(source_file: str, result_file: str):
    """
    The function converting dataset from BERT-NER  to spacy-NER format
    :param source_file: file contain bert  dataset
    :param result_file: result in spacy format
    """
    logging.info(f"Start converting dataset from BERT-NER format  to spacy-NER format...")
    file = open(os.getcwd() + '/' + source_file, 'r')
    iob_data = file.readlines()

    text = {}
    iob_tags = {}
    text_id = 0
    for line in iob_data:
        if line != '\n':
            splitted_line = line.split(' ')
            if text.get(text_id):
                text[text_id].append(splitted_line[0])
            else:
                text[text_id] = [splitted_line[0]]
            if iob_tags.get(text_id):
                iob_tags[text_id].append(splitted_line[1].replace('\n', ''))
            else:
                iob_tags[text_id] = [splitted_line[1].replace('\n', '')]
        else:
            text_id += 1
    for line_id, line in text.items():
        doc = Doc(NLP.vocab, words=line, ents=iob_tags[line_id])
        labels = [{'start': ent.start_char, 'end': ent.end_char, 'label': ent.label_} for ent in doc.ents]
        srsly.write_jsonl(os.getcwd() + '/' + result_file, [{'id': line_id, 'text': doc.text, 'label': labels}],
                          True, False)
    logging.info(f"work detection algorithm")


def main(options):
    if options.spacy_to_bert:
        parse_spacy_to_bert_format(source_file=options.source, result_file=options.result)

    elif options.bert_to_spacy:
        parse_bert_to_spacy_format(source_file=options.source, result_file=options.result)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="Dataset Converter",
        description="This program converts spacy-NER dataset to BERT-NER dataset and conversely")
    parser.add_argument("-s", "--source", action="store",
                        help="Path to source file for parsing")
    parser.add_argument("-r", "--result", action="store",
                        help="Path to result file of parsing")
    parser.add_argument("--spacy_to_bert", action="store", type=bool, default=False,
                        help="Flag of using spacy to bert dataset parsing")
    parser.add_argument("--bert_to_spacy", action="store", type=bool, default=False,
                        help="Flag of using bert to spacy dataset parsing")
    parser.add_argument("--dry", action="store_true", default=False,
                        help="Flag of dry run. If True, use log level - DEBUG")
    parser.add_argument("-l", "--log", action="store", default=None,
                        help="File name of log file")
    args = parser.parse_args()

    logging.basicConfig(filename=args.log, level=logging.INFO if not args.dry else logging.DEBUG,
                        format='[%(asctime)s] %(levelname).1s %(message)s', datefmt='%Y.%m.%d %H:%M:%S')

    logging.info(f"Dataset Converter started with options: {args.__dict__}")
    try:
        main(args)
    except Exception as e:
        logging.exception(f"Unexpected error: {e}", exc_info=True)
        sys.exit(1)
