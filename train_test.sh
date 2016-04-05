python2.7 train_models.py implicit_conll_ff_train
python2.7 neural_discourse_parser.py en conll16st-en-03-29-16-blind-test . output_en 
python2.7 ../conll16st/tira_sup_eval.py conll16st-en-03-29-16-blind-test output_en output_en

python2.7 train_models.py implicit_conll_zh_ff_train
python2.7 neural_discourse_parser.py zh conll16st-zh-03-22-2016-blind-test . output_zh
python2.7 ../conll16st/tira_sup_eval.py conll16st-zh-03-22-2016-blind-test output_zh output_zh
