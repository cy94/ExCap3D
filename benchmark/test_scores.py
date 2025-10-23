from nltk.translate.meteor_score import meteor_score
from torchmetrics.text import BLEUScore, ROUGEScore
from torchmetrics.text.bert import BERTScore
from pprint import pprint
from codetiming import Timer
from nltk import word_tokenize

from benchmark.cider import Cider

def main():
    # eval single pred against multiple GT
    pred = 'hello there how are you'
    gt = [f'hello there how are you{i}' for i in range(10)]

    pred2 = 'how are you doing?'
    gt2 = [f'how are you doing?{i}' for i in range(10)]

    pred3 = 'the sun is shining'
    gt3 = [f'the sun is shining' for i in range(10)]

    rouge = ROUGEScore()
    bleu4 = BLEUScore()
    cider = Cider()
    bertscore = BERTScore(device='cuda')

    print('Rouge:', rouge([pred], [gt])['rougeL_fmeasure']) # tensor
    print('B4:', bleu4([pred], [gt])) # tensor
    meteor_gt = [word_tokenize(gt_cap) for gt_cap in gt]
    meteor_pred = word_tokenize(pred)
    print('Meteor:', meteor_score(meteor_gt, meteor_pred)) # float

    cider_gts = {'test_key': gt, 'test_key2': gt2, 'test_key3': gt3}
    cider_preds = {'test_key': [pred], 'test_key2': [pred2], 'test_key3': [pred3]}
    cider_avg, cider_scores = cider.compute_score(cider_gts, cider_preds)
    print('Cider avg:', cider_avg) # float
    print('Cider scores:', cider_scores) # list of floats

    bert_scores = bertscore([pred], gt)['f1']


if __name__ == '__main__':
    main()
